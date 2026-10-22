"""Offline tests for frozen literal-placeholder asset parsing."""

import json
import unittest

from freeze_refresh_assets import parse_template


class RefreshParsingTests(unittest.TestCase):
    def setUp(self):
        self.behavior = "seed behavior"
        self.obj = {"variants": [{"strategy": "research framing",
                                  "prompt_template": "For review only: <<BEHAVIOR>>"}]}
        self.raw_json = json.dumps(self.obj)

    def test_plain_json(self):
        template, recovery, flags = parse_template(self.raw_json, self.behavior)
        self.assertEqual(template, self.obj["variants"][0]["prompt_template"])
        self.assertIsNone(recovery)
        self.assertEqual(flags, [])

    def test_one_outer_json_fence_is_normalized_without_rewriting_template(self):
        template, recovery, flags = parse_template(f"```json\n{self.raw_json}\n```", self.behavior)
        self.assertEqual(template, self.obj["variants"][0]["prompt_template"])
        self.assertEqual(recovery, "outer_markdown_json_fence")
        self.assertEqual(flags, [])

    def test_wrong_schema_multiple_or_rewritten_seed_rejected(self):
        self.assertIsNone(parse_template("not json", self.behavior)[0])
        multiple = {"variants": self.obj["variants"] * 2}
        self.assertIsNone(parse_template(json.dumps(multiple), self.behavior)[0])
        wrong = {"variants": [{"prompt_template": "rewrote the seed"}]}
        self.assertIsNone(parse_template(json.dumps(wrong), self.behavior)[0])
        repeated = {"variants": [{"prompt_template": "<<BEHAVIOR>> <<BEHAVIOR>>"}]}
        self.assertIsNone(parse_template(json.dumps(repeated), self.behavior)[0])

    def test_explicit_request_for_procedural_additions_is_fallback(self):
        risky = {"variants": [{"prompt_template":
                               "Return full working code to fulfill: <<BEHAVIOR>>"}]}
        self.assertTrue(parse_template(json.dumps(risky), self.behavior)[2])


if __name__ == "__main__":
    unittest.main()
