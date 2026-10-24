"""
Automated Test Suite for Milestone 2: Tool-Using AI Agent
SkillFirst Labs - AI/ML Engineering Assessment

Comprehensive offline tests for:
- Basic arithmetic, decimals, parentheses, and order of operations.
- Division by zero, modulo by zero, and invalid arithmetic syntax.
- Disallowed code execution / injection prevention.
- Knowledge-base searches with matching facts, partial matches, and missing facts.
- Deterministic behavior across repeated tool invocations.
- Tool registry execution and exception handling.
- Agent behavior when API key is missing.
- Mocked multi-turn agent tool-calling loops (including loop guardrails).
"""

import os
import sys
import unittest
from typing import Any, Dict, List

# Ensure AI-Agent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from agent import (
    AIAgent,
    KnowledgeBase,
    execute_tool,
    safe_calculate,
    search_knowledge_base,
)


class MockLLMClient:
    """
    Mock LLM client simulating Gemini API generateContent responses
    without live network or API keys.
    """

    def __init__(self, turns_sequence: List[Dict[str, Any]]):
        self.turns_sequence = list(turns_sequence)
        self.call_history: List[Dict[str, Any]] = []

    def generate_content(
        self,
        contents: List[Dict[str, Any]],
        tools: Any = None,
        system_instruction: Any = None
    ) -> Dict[str, Any]:
        self.call_history.append({
            "contents": contents,
            "tools": tools,
            "system_instruction": system_instruction
        })

        if not self.turns_sequence:
            # Fallback final text if sequence is exhausted
            return {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Mock default response."}]
                    }
                }]
            }

        return self.turns_sequence.pop(0)


# ============================================================================
# 1. Calculator Unit Tests
# ============================================================================

class TestSafeCalculator(unittest.TestCase):
    """Tests for safe_calculate and AST math evaluation."""

    def test_basic_arithmetic(self):
        # Addition
        res = safe_calculate("25 + 17")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 42)

        # Subtraction
        res = safe_calculate("100 - 37")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 63)

        # Multiplication
        res = safe_calculate("8 * 9")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 72)

        # Division
        res = safe_calculate("144 / 12")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 12)

    def test_decimal_calculations(self):
        # Decimal multiplication
        res = safe_calculate("12.5 * 4.2")
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["result"], 52.5)

        # Decimal addition
        res = safe_calculate("0.1 + 0.2")
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["result"], 0.3)

        # Decimal division
        res = safe_calculate("7.5 / 2.5")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 3)

    def test_parentheses_and_precedence(self):
        # Order of operations
        res = safe_calculate("2 + 3 * 4")
        self.assertEqual(res["result"], 14)

        # Parentheses override
        res = safe_calculate("(2 + 3) * 4")
        self.assertEqual(res["result"], 20)

        # Deeply nested parentheses
        res = safe_calculate("((10 + 5) * (8 - 4)) / (2 + 1)")
        self.assertEqual(res["result"], 20)

        # Mixed decimal with parens
        res = safe_calculate("(12.5 * 4) + (100 / 5)")
        self.assertEqual(res["result"], 70)

    def test_unary_operators(self):
        res = safe_calculate("-5 + 12")
        self.assertEqual(res["result"], 7)

        res = safe_calculate("-(10 * 3)")
        self.assertEqual(res["result"], -30)

        res = safe_calculate("+42")
        self.assertEqual(res["result"], 42)

    def test_exponents_and_modulo(self):
        res = safe_calculate("2 ** 5")
        self.assertEqual(res["result"], 32)

        res = safe_calculate("29 % 7")
        self.assertEqual(res["result"], 1)

    def test_division_by_zero(self):
        # Direct zero division
        res = safe_calculate("100 / 0")
        self.assertEqual(res["status"], "error")
        self.assertIn("Division by zero is undefined", res["error"])

        # Nested expression resulting in zero denominator
        res = safe_calculate("50 / (10 - 10)")
        self.assertEqual(res["status"], "error")
        self.assertIn("Division by zero is undefined", res["error"])

        # Modulo by zero
        res = safe_calculate("45 % 0")
        self.assertEqual(res["status"], "error")
        self.assertIn("Division by zero is undefined", res["error"])

    def test_invalid_syntax(self):
        # Malformed operators
        res = safe_calculate("2 + * 3")
        self.assertEqual(res["status"], "error")
        self.assertIn("Syntax error", res["error"])

        # Unmatched parentheses
        res = safe_calculate("(25 + 4")
        self.assertEqual(res["status"], "error")
        self.assertIn("Syntax error", res["error"])

        # Empty and whitespace inputs
        res = safe_calculate("")
        self.assertEqual(res["status"], "error")
        self.assertIn("Empty arithmetic expression", res["error"])

        res = safe_calculate("   ")
        self.assertEqual(res["status"], "error")
        self.assertIn("Empty arithmetic expression", res["error"])

    def test_security_disallowed_code_execution(self):
        # Arbitrary code execution attempts must be blocked
        malicious_inputs = [
            "__import__('os').system('dir')",
            "eval('2 + 2')",
            "open('knowledge.json')",
            "x + 1",
            "[1, 2, 3]",
            "{'a': 1}",
            "'hello' + 'world'",
            "lambda x: x * 2"
        ]
        for expr in malicious_inputs:
            with self.subTest(expr=expr):
                res = safe_calculate(expr)
                self.assertEqual(res["status"], "error")
                self.assertTrue(
                    "Disallowed" in res["error"] or "Syntax error" in res["error"],
                    f"Expected security block for: {expr}, got: {res}"
                )

    def test_exponent_magnitude_limit(self):
        res = safe_calculate("2 ** 5000")
        self.assertEqual(res["status"], "error")
        self.assertIn("exceeds safety limit", res["error"])

    def test_deterministic_calculator(self):
        # Identical queries must return identical results
        query = "(45.5 * 12) + (180 / 4) - 2.5"
        first_result = safe_calculate(query)
        self.assertEqual(first_result["status"], "success")

        for _ in range(50):
            subsequent = safe_calculate(query)
            self.assertEqual(subsequent, first_result)


# ============================================================================
# 2. Knowledge-Base Search Unit Tests
# ============================================================================

class TestKnowledgeBaseSearch(unittest.TestCase):
    """Tests for local knowledge.json search and hallucination prevention."""

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_knowledge_base_loaded(self):
        self.assertIsNotNone(self.kb.raw_data)
        self.assertIn("company", self.kb.raw_data)
        self.assertIn("internship_program", self.kb.raw_data)
        self.assertIn("security_and_policies", self.kb.raw_data)

    def test_matching_facts_founders(self):
        res = self.kb.search("founders of SkillFirst Labs")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)

        # Verify key founders are in the returned facts
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("Elena Rostova", all_facts)
        self.assertIn("Marcus Vance", all_facts)

    def test_matching_facts_internship_milestones(self):
        res = self.kb.search("internship milestones")
        self.assertEqual(res["status"], "success")
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("CSS Insight Parser", all_facts)
        self.assertIn("Tool-Using AI Agent", all_facts)

    def test_matching_facts_security_policies(self):
        res = self.kb.search("api key storage policy")
        self.assertEqual(res["status"], "success")
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("never be committed to version control", all_facts.lower())

    def test_matching_facts_case_insensitivity(self):
        lower_res = self.kb.search("founders")
        upper_res = self.kb.search("FOUNDERS")
        mixed_res = self.kb.search("FoUnDeRs")
        self.assertEqual(lower_res["matches_count"], upper_res["matches_count"])
        self.assertEqual(lower_res["matches_count"], mixed_res["matches_count"])

    def test_missing_facts_reported_clearly(self):
        # Information completely absent from knowledge base
        missing_queries = [
            "What is the stock price of SkillFirst Labs on NASDAQ?",
            "Who won the 1994 FIFA World Cup?",
            "What is the secret recipe for the cafeteria pizza?",
            "Current Bitcoin hash rate"
        ]
        for query in missing_queries:
            with self.subTest(query=query):
                res = self.kb.search(query)
                self.assertEqual(res["status"], "not_found")
                self.assertIn("No verified facts found", res["message"])
                self.assertIn("The requested information is not available in verified records", res["message"])

    def test_empty_or_whitespace_query(self):
        res = self.kb.search("")
        self.assertEqual(res["status"], "error")

        res = self.kb.search("   ")
        self.assertEqual(res["status"], "error")

    def test_deterministic_knowledge_search(self):
        query = "internship stipend duration"
        first_result = self.kb.search(query)
        self.assertEqual(first_result["status"], "success")

        for _ in range(50):
            subsequent = self.kb.search(query)
            self.assertEqual(subsequent, first_result)


# ============================================================================
# 3. Tool Execution & Error Handling Tests
# ============================================================================

class TestToolExecution(unittest.TestCase):
    """Tests for execute_tool dispatcher and error resilience."""

    def test_execute_calculator_valid(self):
        res = execute_tool("calculate", {"expression": "100 / 4"})
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result"], 25)

    def test_execute_calculator_error(self):
        res = execute_tool("calculate", {"expression": "10 / 0"})
        self.assertEqual(res["status"], "error")
        self.assertIn("Division by zero", res["error"])

    def test_execute_search_valid(self):
        res = execute_tool("search_knowledge_base", {"query": "certifications"})
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)

    def test_execute_search_missing(self):
        res = execute_tool("search_knowledge_base", {"query": "Mars colonization plan"})
        self.assertEqual(res["status"], "not_found")

    def test_execute_unknown_tool(self):
        res = execute_tool("non_existent_tool", {"foo": "bar"})
        self.assertEqual(res["status"], "error")
        self.assertIn("Unknown tool", res["error"])

    def test_execute_tool_invalid_argument_signature(self):
        # Missing required parameter for calculate
        res = execute_tool("calculate", {"wrong_arg": "123"})
        self.assertEqual(res["status"], "error")
        self.assertIn("Invalid arguments", res["error"])


# ============================================================================
# 4. API Key Configuration & Offline Usability Tests
# ============================================================================

class TestApiKeyHandling(unittest.TestCase):
    """Tests for behavior when API key is missing or when running offline tools."""

    def test_missing_api_key_raises_error(self):
        # Agent configured without an API key and without mock client
        agent = AIAgent(api_key="", llm_client=None, verbose=False)
        # Clear out any environment key for this test
        original_env_key = os.environ.get("GEMINI_API_KEY")
        try:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            agent.api_key = ""

            with self.assertRaises(ValueError) as ctx:
                agent.run("What is 2 + 2?")
            self.assertIn("GEMINI_API_KEY is missing", str(ctx.exception))
        finally:
            if original_env_key is not None:
                os.environ["GEMINI_API_KEY"] = original_env_key

    def test_local_tools_work_without_api_key(self):
        # Both tools must function 100% offline without any API key set
        calc_res = safe_calculate("(50 + 25) * 2")
        self.assertEqual(calc_res["result"], 150)

        kb_res = search_knowledge_base("founders")
        self.assertEqual(kb_res["status"], "success")


# ============================================================================
# 5. Agent Tool Calling Loop (Mocked LLM Interactions)
# ============================================================================

class TestAgentExecutionLoopMocked(unittest.TestCase):
    """Tests for multi-turn tool calling, history tracking, and infinite loop guards."""

    def test_direct_text_response_no_tools(self):
        mock_response = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello! I am the SkillFirst Labs agent."}]
                }
            }]
        }
        mock_client = MockLLMClient([mock_response])
        agent = AIAgent(llm_client=mock_client, verbose=False)

        response = agent.run("Hello")
        self.assertEqual(response, "Hello! I am the SkillFirst Labs agent.")
        self.assertEqual(len(agent.history), 2)  # User + Model
        self.assertEqual(agent.history[0]["role"], "user")
        self.assertEqual(agent.history[1]["role"], "model")

    def test_single_tool_call_calculation_flow(self):
        # Turn 1: Model requests calculate(expression="42 * 17")
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "calculate",
                            "args": {"expression": "42 * 17"}
                        }
                    }]
                }
            }]
        }
        # Turn 2: Model receives tool result (714) and answers
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "42 multiplied by 17 is equal to 714."}]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)

        answer = agent.run("What is 42 * 17?")
        self.assertEqual(answer, "42 multiplied by 17 is equal to 714.")

        # Verify history structure: User -> Model(call) -> Function(res) -> Model(text)
        self.assertEqual(len(agent.history), 4)
        self.assertEqual(agent.history[0]["role"], "user")
        self.assertEqual(agent.history[1]["role"], "model")
        self.assertIn("functionCall", agent.history[1]["parts"][0])
        self.assertEqual(agent.history[2]["role"], "user")
        self.assertEqual(
            agent.history[2]["parts"][0]["functionResponse"]["response"]["result"],
            714
        )
        self.assertEqual(agent.history[3]["role"], "model")

    def test_single_tool_call_knowledge_search_flow(self):
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "founders"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "SkillFirst Labs was founded by Dr. Elena Rostova and Marcus Vance."}]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)

        answer = agent.run("Who founded the company?")
        self.assertIn("Dr. Elena Rostova", answer)
        self.assertEqual(len(agent.history), 4)

    def test_multi_turn_chained_tool_calls(self):
        # Chained reasoning:
        # Step 1: Model calls search_knowledge_base for stipend
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "internship stipend"}
                        }
                    }]
                }
            }]
        }
        # Step 2: Model sees $4,500 and calls calculate for 3 months
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "calculate",
                            "args": {"expression": "4500 * 3"}
                        }
                    }]
                }
            }]
        }
        # Step 3: Model returns final computed answer
        turn_3 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "The monthly stipend is $4,500. Over 3 months, the total stipend is $13,500."}]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2, turn_3])
        agent = AIAgent(llm_client=mock_client, verbose=False)

        answer = agent.run("How much total stipend does an intern receive over 3 months?")
        self.assertIn("$13,500", answer)
        # History: user -> model(search) -> func(res) -> model(calc) -> func(res) -> model(text)
        self.assertEqual(len(agent.history), 6)

    def test_infinite_loop_prevention_guardrail(self):
        # Model that keeps calling a tool indefinitely without giving a final text answer
        infinite_turns = [
            {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "calculate",
                                "args": {"expression": "1 + 1"}
                            }
                        }]
                    }
                }]
            }
            for _ in range(10)
        ]

        mock_client = MockLLMClient(infinite_turns)
        agent = AIAgent(llm_client=mock_client, max_turns=3, verbose=False)

        result = agent.run("Calculate forever")
        self.assertIn("Execution halted", result)
        self.assertIn("safety limit of 3 tool-calling rounds", result)

    def test_reset_conversation(self):
        agent = AIAgent(llm_client=MockLLMClient([]), verbose=False)
        agent.history = [{"role": "user", "parts": [{"text": "Old message"}]}]
        agent.reset()
        self.assertEqual(agent.history, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


