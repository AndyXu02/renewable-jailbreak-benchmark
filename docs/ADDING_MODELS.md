# Testing another model

## Ollama

Any installed Ollama chat model can be passed directly with `--model`. No code
or registry edit is required.

```bash
ollama list
rjbench-run --backend ollama --model MODEL_NAME ...
```

Use the exact identifier displayed by `ollama list`. If the model exposes a
reasoning mode, record the `--think` choice.

## OpenAI-compatible servers

Use `--backend openai`, `--api-base`, and `--api-key-env`. This works with many
hosted providers and local servers such as vLLM when they expose the standard
chat-completions shape.

The endpoint must return:

```json
{"choices": [{"message": {"content": "..."}}]}
```

## Adding a new backend

Implement a class with:

```python
def generate(self, messages, temperature, max_tokens, seed):
    return {"response": "...", "error": null}
```

Then add it to `create_backend()` in `src/rjbench/backends.py`. Add a test that
does not require network access.

## Comparability checklist

- Freeze the behavior and attack files before the run.
- Use a separate judge model.
- Keep the prompt-delivery scenario and defense identical.
- Report output caps and reasoning settings.
- Treat scores as judge-dependent estimates.
- Do not mix in-sample PAIR optimization with transfer results.

