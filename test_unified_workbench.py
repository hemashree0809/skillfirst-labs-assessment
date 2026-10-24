"""
Smoke and Integration Test Suite for Insight Lab — AI Engineering Workbench
SkillFirst Labs: Milestone 1 (CSS Parser) + Milestone 2 (Tool-Using AI Agent)

Validates:
- Import resolution across root, CSS-Insight-Parser, and AI-Agent directories.
- Parser analysis on sample.css and custom CSS snippets.
- Safe AST calculator execution, decimal handling, and division-by-zero safety.
- Knowledge-base search grounding and not_found reporting.
- Deterministic behavior and offline execution without an API key.
- Root app.py module syntax and dependency validation.
"""

import json
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CSS_PARSER_DIR = ROOT_DIR / "CSS-Insight-Parser"
AI_AGENT_DIR = ROOT_DIR / "AI-Agent"

for p in (str(ROOT_DIR), str(CSS_PARSER_DIR), str(AI_AGENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from parser import analyze_css, parse_css
from agent import (
    AIAgent,
    safe_calculate,
    search_knowledge_base,
    TOOL_DECLARATIONS,
    TOOL_REGISTRY,
    load_env,
)


class TestUnifiedWorkbenchIntegration(unittest.TestCase):
    """Integration and smoke tests for the unified workbench."""

    def test_css_parser_integration(self):
        sample_path = CSS_PARSER_DIR / "sample.css"
        self.assertTrue(sample_path.exists(), "sample.css must exist in CSS-Insight-Parser")

        css_text = sample_path.read_text(encoding="utf-8")
        report = analyze_css(css_text, "sample.css")

        self.assertIn("total_rules", report)
        self.assertIn("total_declarations", report)
        self.assertIn("property_frequency", report)
        self.assertIn("duplicate_selectors", report)
        self.assertIn("rules", report)

        # Normalize.css v8.0.1 benchmark metrics
        self.assertEqual(report["total_rules"], 34)
        self.assertEqual(report["total_declarations"], 56)
        self.assertEqual(len(report["property_frequency"]), 27)

    def test_css_parser_custom_snippet(self):
        custom_css = """
        body { margin: 0; padding: 10px; }
        h1, h2 { color: #333; }
        body { margin: 0; }
        """
        report = analyze_css(custom_css, "custom.css")
        self.assertEqual(report["total_rules"], 3)
        self.assertEqual(report["total_declarations"], 4)
        self.assertIn("body", report["duplicate_selectors"])

    def test_local_calculator_integration(self):
        res = safe_calculate("(45.5 * 12) + (180 / 4)")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 591)

        err_res = safe_calculate("100 / 0")
        self.assertEqual(err_res["status"], "error")
        self.assertIn("Division by zero", err_res["error"])

    def test_local_knowledge_search_integration(self):
        # Known query for founder
        found_res = search_knowledge_base("founder")
        self.assertEqual(found_res["status"], "success")
        self.assertGreater(found_res["matches_count"], 0)
        all_facts = " ".join(m["fact"] for m in found_res["results"])
        self.assertIn("Gurunathan Murugan", all_facts)
        self.assertNotIn("Elena Rostova", all_facts)

        # Entity mismatch query (Google must never return SkillFirst data)
        google_res = search_knowledge_base("Where is the Google headquarters?")
        self.assertEqual(google_res["status"], "not_found")
        self.assertEqual(google_res.get("detected_entity"), "Google")

        # Unknown query (hallucination prevention)
        missing_res = search_knowledge_base("What is the current stock price of SkillFirst Labs?")
        self.assertEqual(missing_res["status"], "not_found")
        self.assertIn("No verified facts found", missing_res["message"])

    def test_agent_offline_initialization(self):
        agent = AIAgent(verbose=False)
        self.assertIsNotNone(agent)
        self.assertEqual(len(agent.history), 0)
        agent.reset()
        self.assertEqual(len(agent.history), 0)

    def test_env_loader_safe(self):
        env_file = AI_AGENT_DIR / ".env"
        # Loading should not crash regardless of whether .env exists
        load_env(str(env_file))

    def test_root_app_syntax_and_import(self):
        import app
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
