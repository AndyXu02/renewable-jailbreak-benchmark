"""
Main experiment runner.

Orchestrates: load data -> apply attack -> apply scenario -> apply defense -> run model -> save output.

Usage:
    # Tiny milestone (1 model, manual seed, direct scenario, no defense)
    python run_inference.py --phase tiny --model qwen3-8b

    # Phase 1: Baseline characterization
    python run_inference.py --phase phase1_baseline --model all

    # Single model, specific config
    python run_inference.py --phase phase2_scenarios --model llama3.1-8b --attacks A0_direct A2_persona
"""

import argparse
import json
import os
import time
import datetime
from pathlib import Path
from typing import Optional

import requests
import re

from config import (
    MODELS, MODELS_LITE, FORMAL_MODELS, PHASES, DATA_PROCESSED,
    HARMFUL_BEHAVIORS_V2, RUNS_DIR, DEFAULT_DECODING,
)
from utils import load_jsonl, save_jsonl, generate_run_id, create_experiment_record, stratified_sample
from attacks import apply_attack
from scenarios import apply_scenario
from defenses import apply_defense, aggregate_perturbation_responses


# ============================================================
# Model Inference Backends
# ============================================================
class OllamaBackend:
    """Inference via Ollama REST API."""
    
    def __init__(self, model_id: str, api_base: str = "http://localhost:11434", no_think: bool = False):
        self.model_id = model_id
        self.api_base = api_base
        self.chat_url = f"{api_base}/api/chat"
        self.no_think = no_think
    
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        """Send chat completion request to Ollama."""
        is_gpt_oss = self.model_id.startswith("gpt-oss")
        num_predict = max_tokens
        if is_gpt_oss:
            # gpt-oss emits a separate reasoning trace; guarantee enough budget
            # so the short final answer (e.g. a single judge digit) is produced.
            num_predict = max(max_tokens, 1024)

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "seed": seed,
            },
        }
        if self.no_think:
            # Current Ollama versions expose thinking control as a top-level
            # request option; a textual /no_think suffix is not reliable.
            payload["think"] = False
        elif is_gpt_oss:
            # Bounded reasoning effort for the judge; keeps latency reasonable.
            payload["think"] = "low"
        
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            raw_response = message.get("content", "")
            # If the final channel is empty (reasoning consumed the budget),
            # fall back to the reasoning trace so a score can still be parsed.
            if is_gpt_oss and not raw_response.strip():
                raw_response = message.get("thinking", "")
            # Strip Qwen3 thinking tags
            clean_response = re.sub(r'<think>.*?</think>\s*', '', raw_response, flags=re.DOTALL).strip()
            return {
                "response": clean_response,
                "model": data.get("model", self.model_id),
                "total_duration": data.get("total_duration"),
                "eval_count": data.get("eval_count"),
                "error": None,
            }
        except requests.exceptions.RequestException as e:
            return {
                "response": "",
                "model": self.model_id,
                "error": str(e),
            }


class OpenAICompatibleBackend:
    """Inference via OpenAI-compatible API (DeepSeek, OpenAI, Together, etc.)."""
    
    def __init__(self, model_id: str, api_base: str, api_key: str = None):
        self.model_id = model_id
        self.api_base = api_base.rstrip("/")
        # Load API key: explicit > env var > .env file
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("DEEPSEEK_API_KEY="):
                        self.api_key = line.split("=", 1)[1].strip()
                        break
        self.chat_url = f"{self.api_base}/chat/completions"
    
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        """Send chat completion request to OpenAI-compatible API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        
        last_error = None
        for attempt in range(4):
            try:
                resp = requests.post(
                    self.chat_url, json=payload, headers=headers, timeout=120
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                if not content.strip():
                    last_error = RuntimeError(
                        f"Empty final content at max_tokens={payload['max_tokens']}"
                    )
                    if attempt < 3:
                        # Reasoning APIs can consume the entire completion budget
                        # before emitting user-visible final content. Retry with a
                        # larger budget rather than recording an empty response as
                        # a model refusal.
                        payload["max_tokens"] = min(payload["max_tokens"] * 2, 8192)
                        time.sleep(2 ** attempt)
                        continue
                    return {
                        "response": "",
                        "model": data.get("model", self.model_id),
                        "usage": data.get("usage"),
                        "error": str(last_error),
                    }
                return {
                    "response": content.strip(),
                    "model": data.get("model", self.model_id),
                    "usage": data.get("usage"),
                    "error": None,
                }
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2 ** attempt)

        return {
            "response": "",
            "model": self.model_id,
            "error": str(last_error),
        }


class HuggingFaceBackend:
    """Inference via HuggingFace Transformers pipeline (local GPU/CPU)."""
    
    def __init__(self, model_id: str, device: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        print(f"  Loading model: {model_id} ...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        dtype = torch.float16 if device != "cpu" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
        if device in ("mps", "cuda"):
            self.model = self.model.to(device)
        
        self.device = device
        print(f"  Model loaded on {device}")
    
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        """Generate response using HuggingFace model."""
        import torch
        
        try:
            # Apply chat template
            if self.tokenizer.chat_template:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback: concatenate messages
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    text += f"<|{role}|>\n{content}\n"
                text += "<|assistant|>\n"
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # Set seed
            torch.manual_seed(seed)
            
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return {
                "response": response.strip(),
                "model": self.model_id,
                "error": None,
            }
        except Exception as e:
            return {
                "response": "",
                "model": self.model_id,
                "error": str(e),
            }


class MLXBackend:
    """Inference via MLX-LM, optimized for Apple Silicon."""

    def __init__(self, model_id: str):
        from mlx_lm import load

        print(f"  Loading MLX model: {model_id} ...")
        self.model_id = model_id
        self.is_gemma4 = "gemma-4" in model_id.lower() or "gemma4" in model_id.lower()
        model_config = None
        # Gemma 4 HF repos currently expose the text model as gemma4_unified,
        # while mlx-lm implements the text path as gemma4. The project only
        # evaluates text prompts, so the multimodal weights can be ignored by
        # the local mlx-lm sanitizer patch in .venv_mlx312.
        if self.is_gemma4:
            model_config = {"model_type": "gemma4"}
        self.model, self.tokenizer = load(model_id, model_config=model_config)
        print("  MLX model loaded")

    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages using the model tokenizer when available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if self.is_gemma4:
                    # With enable_thinking=False, the Gemma4 HF template ends
                    # in an empty thought channel. Appending a final channel
                    # after it makes mlx-lm decode the literal word "thought"
                    # before the real answer. Replace the empty channel instead.
                    empty_thought = "<|channel>thought\n<channel|>"
                    if prompt.endswith(empty_thought):
                        prompt = prompt[:-len(empty_thought)] + "<|channel>final\n"
                    elif not prompt.endswith("<|channel>final\n"):
                        prompt += "<|channel>final\n"
                return prompt
            except Exception:
                pass

        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}\n"
        return text + "<|assistant|>\n"

    def _clean_response(self, response: str) -> str:
        """Remove chat-template control channels from MLX text output."""
        if not self.is_gemma4:
            return response.strip()

        text = response
        if "<|channel>final\n" in text:
            text = text.split("<|channel>final\n", 1)[1]
        elif "<|channel>thought" in text and "<channel|>" in text:
            text = text.split("<channel|>", 1)[1]
        if "<channel|>" in text:
            before, after = text.split("<channel|>", 1)
            # Legacy append-style prompts can decode as
            # "thought\n<channel|><actual final>". Treat the bare channel name
            # as control text, never as the model response.
            text = after if before.strip().lower() in {"", "thought", "final"} else before
        text = re.sub(r"<\|channel>(?:thought|final)\n?", "", text)
        text = text.replace("<channel|>", "")
        return text.strip()

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
        **kwargs,
    ) -> dict:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        try:
            prompt = self._format_messages(messages)
            sampler = make_sampler(temp=temperature)
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                verbose=False,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            clean_response = self._clean_response(response)
            if self.is_gemma4 and not clean_response:
                # A small number of prompts terminate immediately after the
                # forced final-channel sentinel. Retry with Gemma's native
                # empty-thought generation suffix, which reliably emits the
                # refusal/final content instead of a channel-only decode.
                final_suffix = "<|channel>final\n"
                fallback_prompt = prompt
                if prompt.endswith(final_suffix):
                    fallback_prompt = (
                        prompt[:-len(final_suffix)]
                        + "<|channel>thought\n<channel|>"
                    )
                fallback_response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=fallback_prompt,
                    verbose=False,
                    max_tokens=max_tokens,
                    sampler=sampler,
                )
                clean_response = self._clean_response(fallback_response)
                if not clean_response:
                    return {
                        "response": "",
                        "model": self.model_id,
                        "error": "Gemma4 produced empty/control-only output after channel fallback",
                    }
            return {
                "response": clean_response,
                "model": self.model_id,
                "error": None,
            }
        except Exception as e:
            return {
                "response": "",
                "model": self.model_id,
                "error": str(e),
            }


def get_backend(model_key: str, backend_override: Optional[str] = None):
    """Instantiate the appropriate backend for a model."""
    cfg = MODELS[model_key]
    if backend_override == "mlx":
        return MLXBackend(cfg.model_id)
    if backend_override:
        raise ValueError(f"Unsupported backend override: {backend_override}")
    if cfg.backend == "ollama":
        return OllamaBackend(cfg.model_id, cfg.api_base, no_think=cfg.no_think)
    elif cfg.backend == "huggingface":
        return HuggingFaceBackend(cfg.model_id)
    elif cfg.backend == "mlx":
        return MLXBackend(cfg.model_id)
    elif cfg.backend == "openai_compatible":
        return OpenAICompatibleBackend(cfg.model_id, cfg.api_base)
    else:
        raise NotImplementedError(f"Backend '{cfg.backend}' not yet implemented.")


# ============================================================
# Multi-turn Inference
# ============================================================
def run_multiturn(backend, messages: list[dict], decoding: dict) -> list[dict]:
    """Run multi-turn inference, generating assistant responses for each turn.
    
    For multi-turn attacks (A4) or scenarios (S2), we need to generate
    intermediate responses before sending the next user message.
    """
    conversation = []
    generated_responses = []
    user_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
    final_user_index = user_indices[-1] if user_indices else None
    intermediate_max_tokens = decoding.get("intermediate_max_tokens")
    
    for idx, msg in enumerate(messages):
        if msg["role"] == "system":
            conversation.append(msg)
        elif msg["role"] == "user":
            conversation.append(msg)
            # Generate assistant response
            turn_decoding = decoding.copy()
            turn_decoding.pop("intermediate_max_tokens", None)
            if (
                intermediate_max_tokens is not None
                and idx != final_user_index
                and turn_decoding.get("max_tokens", intermediate_max_tokens) > intermediate_max_tokens
            ):
                turn_decoding["max_tokens"] = intermediate_max_tokens
            result = backend.generate(conversation, **turn_decoding)
            assistant_msg = {"role": "assistant", "content": result["response"]}
            conversation.append(assistant_msg)
            generated_responses.append(result)
        elif msg["role"] == "assistant":
            # Check if this is a placeholder
            if "[PLACEHOLDER" in msg.get("content", ""):
                # Skip - we already generated the response above
                pass
            else:
                conversation.append(msg)
    
    # The last generated response is the one we evaluate
    final_response = generated_responses[-1] if generated_responses else {"response": "", "error": "No responses generated"}
    return {
        "response": final_response["response"],
        "full_conversation": conversation,
        "all_responses": generated_responses,
        "error": final_response.get("error"),
    }


# ============================================================
# Main Experiment Loop
# ============================================================
def run_experiment(
    model_key: str,
    data: list[dict],
    attacks: list[str],
    scenarios: list[str],
    defenses: list[str],
    phase_name: str,
    decoding: dict = None,
    output_path: Optional[Path] = None,
    resume: bool = True,
    backend_override: Optional[str] = None,
    retry_failed: bool = False,
) -> list[dict]:
    """Run a full experiment configuration.
    
    Args:
        model_key: Key from MODELS dict
        data: List of seed prompt records
        attacks: List of attack IDs to apply
        scenarios: List of scenario IDs
        defenses: List of defense IDs
        phase_name: Name for output organization
        decoding: Decoding parameters override
    
    Returns:
        List of experiment records
    """
    if decoding is None:
        decoding = DEFAULT_DECODING.copy()
    
    backend = get_backend(model_key, backend_override=backend_override)
    results = []
    existing_keys = set()
    if output_path and resume and output_path.exists():
        loaded = load_jsonl(output_path)
        dropped_failed = 0
        for r in loaded:
            failed = bool(r.get("metadata", {}).get("error")) or not str(r.get("response", "")).strip()
            if retry_failed and failed:
                dropped_failed += 1
                continue
            results.append(r)
            existing_keys.add((
                r.get("sample_id"),
                r.get("model_id"),
                r.get("attack"),
                r.get("scenario"),
                r.get("defense"),
            ))
        # Retry mode must remove failed rows before regenerated records are
        # appended.  Otherwise the JSONL retains both the failed and successful
        # version of the same experimental cell, creating duplicate keys that
        # can leak into judging and summary statistics.
        if retry_failed and dropped_failed:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(
                f"Retry cleanup: removed {dropped_failed} failed/empty records before regeneration",
                flush=True,
            )
        print(f"Resuming from {output_path}: {len(existing_keys)} records already complete", flush=True)
    total = len(data) * len(attacks) * len(scenarios) * len(defenses)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {phase_name} | Model: {model_key}", flush=True)
    print(f"Samples: {len(data)} | Attacks: {len(attacks)} | Scenarios: {len(scenarios)} | Defenses: {len(defenses)}", flush=True)
    print(f"Total inference calls: {total}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    count = 0
    for sample in data:
        behavior = sample["behavior"]
        sample_id = sample["id"]
        sample_type = sample["type"]
        
        for attack_id in attacks:
            # Skip attacks on benign prompts (just use direct)
            if sample_type == "benign" and attack_id != "A0_direct":
                continue
            
            # Apply attack transformation
            attack_prompt = apply_attack(attack_id, behavior)
            
            for scenario_id in scenarios:
                # Build messages with scenario
                if isinstance(attack_prompt, list):
                    # Multi-turn attack returns list of user messages
                    if scenario_id == "S2_multiturn":
                        messages = apply_scenario(
                            scenario_id, behavior, attack_prompt[0],
                            sub_questions=attack_prompt,
                        )
                    else:
                        # For non-multiturn scenarios, use the final (most harmful) sub-question
                        messages = apply_scenario(scenario_id, behavior, attack_prompt[-1])
                else:
                    messages = apply_scenario(scenario_id, behavior, attack_prompt)
                
                for defense_id in defenses:
                    count += 1
                    record_key = (sample_id, model_key, attack_id, scenario_id, defense_id)
                    if record_key in existing_keys:
                        if count % 10 == 0 or count == total:
                            print(f"  [{count}/{total}] skipped existing", flush=True)
                        continue
                    
                    # Apply defense
                    defense_result = apply_defense(defense_id, messages)
                    
                    if defense_result.get("blocked"):
                        # Defense blocked the input
                        response_text = f"[BLOCKED BY {defense_id}]: {defense_result.get('block_reason', '')}"
                        error = None
                    elif "messages_variants" in defense_result:
                        # Perturbation defense: run multiple variants, majority vote
                        responses = []
                        variant_errors = []
                        for variant_msgs in defense_result["messages_variants"]:
                            r = backend.generate(variant_msgs, **decoding)
                            responses.append(r["response"])
                            if r.get("error"):
                                variant_errors.append(r["error"])
                        response_text, aggregation = aggregate_perturbation_responses(responses)
                        defense_result["aggregation"] = aggregation
                        defense_result["response_variants"] = responses
                        error = "; ".join(dict.fromkeys(variant_errors)) if variant_errors else None
                    else:
                        # Normal inference
                        final_messages = defense_result["messages"]
                        
                        # Check if multi-turn inference needed
                        user_count = sum(1 for m in final_messages if m["role"] == "user")
                        has_placeholder = any(
                            "[PLACEHOLDER" in m.get("content", "")
                            for m in final_messages if m["role"] == "assistant"
                        )
                        
                        if user_count > 1 and has_placeholder:
                            mt_result = run_multiturn(backend, final_messages, decoding)
                            response_text = mt_result["response"]
                            error = mt_result.get("error")
                        else:
                            result = backend.generate(final_messages, **decoding)
                            response_text = result["response"]
                            error = result.get("error")

                        if defense_id in {"D4_rule_output", "D4_guard_output"}:
                            output_guard = apply_defense(
                                defense_id,
                                final_messages,
                                response=response_text,
                            )
                            if output_guard.get("blocked"):
                                response_text = f"[BLOCKED BY {defense_id}]: {output_guard.get('block_reason', '')}"
                                defense_result["output_guard"] = output_guard
                    
                    # Create record
                    record = create_experiment_record(
                        sample_id=sample_id,
                        model_id=model_key,
                        attack=attack_id,
                        scenario=scenario_id,
                        defense=defense_id,
                        prompt=messages,
                        response=response_text,
                        metadata={
                            "behavior": behavior,
                            "judge_behavior": sample.get("judge_behavior", behavior),
                            "original_behavior": sample.get("original_behavior"),
                            "generator_model": sample.get("generator_model"),
                            "generator_family": sample.get("generator_family"),
                            "generator_strategy": sample.get("generator_strategy"),
                            "sample_type": sample_type,
                            "category": sample.get("category"),
                            "phase": phase_name,
                            "decoding": decoding,
                            "error": error,
                            "defense_result": {
                                k: v for k, v in defense_result.items()
                                if k not in ("messages", "messages_variants")
                            },
                        },
                    )
                    results.append(record)
                    if output_path:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f.flush()
                    
                    # Progress
                    if count % 10 == 0 or count == total:
                        print(f"  [{count}/{total}] {sample_id} | {attack_id} | {scenario_id} | {defense_id}", flush=True)
    
    return results


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run jailbreak benchmark experiments")
    parser.add_argument("--phase", choices=["tiny", "phase1_baseline", "phase2_scenarios", "phase3_defenses", "phase4_generated_attacks"],
                        default="tiny")
    parser.add_argument("--model", type=str, default="qwen3-8b",
                        help="Model key or 'all' for all models")
    parser.add_argument("--attacks", nargs="*", default=None,
                        help="Override attack list")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Override scenario list")
    parser.add_argument("--defenses", nargs="*", default=None,
                        help="Override defense list")
    parser.add_argument("--data-file", type=Path, default=None,
                        help="Override data file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_new_tokens / num_predict for faster runs")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override decoding temperature")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path; enables incremental append/resume")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not skip records already present in --output")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-run existing records with an error or empty response")
    parser.add_argument("--backend-override", choices=["mlx"], default=None,
                        help="Force a backend for this run, e.g. mlx for Apple Silicon Gemma4 runs")
    args = parser.parse_args()
    
    # Load data
    default_data_path = (
        HARMFUL_BEHAVIORS_V2
        if args.phase in PHASES
        else DATA_PROCESSED / "seed_prompts.jsonl"
    )
    data_path = args.data_file or default_data_path
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run 'python prepare_data.py --source manual' first.")
        return
    
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")
    
    # Configure phase
    if args.phase == "tiny":
        # Tiny milestone: minimal test
        attacks = ["A0_direct"]
        scenarios = ["S1_direct"]
        defenses = ["D0_none"]
        if args.max_samples is None:
            args.max_samples = 10
    elif args.phase in PHASES:
        phase_cfg = PHASES[args.phase]
        attacks = args.attacks or phase_cfg.attacks
        scenarios = args.scenarios or phase_cfg.scenarios
        defenses = args.defenses or phase_cfg.defenses
        if args.max_samples is None:
            args.max_samples = phase_cfg.n_behaviors
    else:
        attacks = args.attacks or ["A0_direct"]
        scenarios = args.scenarios or ["S1_direct"]
        defenses = args.defenses or ["D0_none"]
    
    # Limit samples
    if args.max_samples:
        data = (
            stratified_sample(data, args.max_samples)
            if args.phase in PHASES
            else data[:args.max_samples]
        )
    
    # Determine models to run
    if args.model == "all":
        model_keys = list(MODELS.keys())
    elif args.model == "formal":
        model_keys = FORMAL_MODELS
    elif args.model == "lite":
        model_keys = list(MODELS_LITE.keys())
    else:
        model_keys = [args.model]
    
    # Run experiments
    all_results = []
    all_result_keys = set()
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"Warning: Model '{model_key}' not in config, skipping.")
            continue
        
        run_id = generate_run_id(model_key, args.phase)
        output_path = args.output or RUNS_DIR / f"{run_id}.jsonl"
        decoding = DEFAULT_DECODING.copy()
        model_cfg = MODELS[model_key]
        decoding["max_tokens"] = model_cfg.max_tokens or decoding["max_tokens"]
        decoding["temperature"] = model_cfg.temperature
        decoding["top_p"] = model_cfg.top_p
        decoding["seed"] = model_cfg.seed
        if args.max_tokens is not None:
            decoding["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            decoding["temperature"] = args.temperature
        results = run_experiment(
            model_key=model_key,
            data=data,
            attacks=attacks,
            scenarios=scenarios,
            defenses=defenses,
            phase_name=args.phase,
            decoding=decoding,
            output_path=output_path,
            resume=not args.no_resume,
            backend_override=args.backend_override,
            retry_failed=args.retry_failed,
        )
        
        # Incremental mode appends throughout the run; rewrite once at the end
        # to normalize the file if no interruption happened.
        save_jsonl(results, output_path)
        print(f"\nSaved {len(results)} results -> {output_path}")
        for record in results:
            key = (
                record.get("sample_id"),
                record.get("model_id"),
                record.get("attack"),
                record.get("scenario"),
                record.get("defense"),
            )
            if key not in all_result_keys:
                all_result_keys.add(key)
                all_results.append(record)
    
    print(f"\n{'='*60}")
    print(f"DONE. Total records: {len(all_results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
