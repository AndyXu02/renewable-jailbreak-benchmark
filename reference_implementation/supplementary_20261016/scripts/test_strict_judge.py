"""Regression tests: final-channel-only, complete and model-identified judging."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from multijudge_run import call, parse_score, validate_reply


class StrictJudgeTests(unittest.TestCase):
    def test_final_score_format(self):
        self.assertEqual(parse_score("Score: **4**\nJustification: incomplete assistance"), 4)
        self.assertEqual(parse_score("4\nJustification: partial assistance"), 4)
        self.assertEqual(parse_score("3"), 3)
        self.assertIsNone(parse_score("3 possible interpretations"))
        self.assertIsNone(parse_score("Score: 1\nScore: 5"))
        self.assertIsNone(parse_score("<think>Score: 4"))
        self.assertEqual(parse_score("<think>Score: 5</think>\nScore: 1"), 1)

    def test_truncation_and_wrong_version(self):
        self.assertEqual(validate_reply("Score: 4", "length", "m3", ["m3"])[0], None)
        self.assertEqual(validate_reply("Score: 4", "stop", "m3.1", ["m3"])[0], None)
        self.assertEqual(validate_reply("Score: 4", "stop", "m3", ["m3"]), (4, None))

    @patch("multijudge_run.requests.post")
    def test_reasoning_is_not_an_answer(self, post):
        response = Mock()
        response.json.return_value = {
            "model": "glm-5.2", "choices": [{"finish_reason": "length", "message": {
                "content": "", "reasoning_content": "Score: 5"}}],
        }
        post.return_value = response
        args = SimpleNamespace(model="glm-5.2", max_tokens=4096, extra_body_json=None,
                               max_retries=0, timeout=30, allowed_served_models=["glm-5.2"])
        result = call(args, "https://example.invalid/v1", "not-a-key",
                      {"kind": "harmful", "behavior": "synthetic test objective", "response": "refusal"})
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"], "truncated_or_nonfinal_finish")
        self.assertEqual(result["raw"], "")
        self.assertEqual(result["request_settings"]["max_tokens"], 4096)

    @patch("multijudge_run.requests.post")
    def test_preserves_full_answer(self, post):
        raw = "Score: 1\nJustification: " + "x" * 1200
        response = Mock()
        response.json.return_value = {"model": "glm-5.2", "choices": [{
            "finish_reason": "stop", "message": {"content": raw}}]}
        post.return_value = response
        args = SimpleNamespace(model="glm-5.2", max_tokens=4096, extra_body_json=None,
                               max_retries=0, timeout=30, allowed_served_models=["glm-5.2"])
        result = call(args, "https://example.invalid/v1", "not-a-key",
                      {"kind": "benign", "behavior": "test", "response": "test"})
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["raw"], raw)
        self.assertEqual(result["validation_version"], 2)


if __name__ == "__main__":
    unittest.main()
