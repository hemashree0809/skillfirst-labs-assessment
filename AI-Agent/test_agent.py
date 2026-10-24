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
        self.assertIn("internship_assessment", self.kb.raw_data)

    def test_matching_facts_company_identity(self):
        res = self.kb.search("Skill First Applied Labs")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("Skill First Applied Labs", all_facts)
        self.assertIn("https://www.skillfirstlabs.com/", all_facts)
        first_item = res["results"][0]
        self.assertEqual(first_item["source_label"], "Official Website")
        self.assertEqual(first_item["verification_status"], "verified_official_website")

    def test_matching_facts_founder(self):
        res = self.kb.search("founder of Skill First Applied Labs")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("Gurunathan Murugan", all_facts)
        self.assertNotIn("Elena Rostova", all_facts)
        self.assertNotIn("Marcus Vance", all_facts)

    def test_matching_facts_mission(self):
        res = self.kb.search("mission of Skill First Applied Labs")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("academia-industry gap", all_facts)
        self.assertIn("talent management platform", all_facts)

    def test_matching_facts_contact_details(self):
        res = self.kb.search("contact address and email of Skill First Applied Labs")
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["matches_count"], 0)
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("RMK Chola Gardens", all_facts)
        self.assertIn("Chennai 600 077", all_facts)
        self.assertIn("hello@keystone-tms.in", all_facts)

    def test_matching_facts_headquarters_linkedin_attribution(self):
        # Querying headquarters must return verified records with LinkedIn attribution and NOT San Francisco
        res = self.kb.search("Where is SkillFirst Labs headquartered?")
        self.assertEqual(res["status"], "success")
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertTrue("Tiruvallur" in all_facts or "Chennai" in all_facts)
        self.assertNotIn("San Francisco", all_facts)
        self.assertIn("Attributed separately from LinkedIn", all_facts)

    def test_google_versus_skillfirst_entity_mismatch(self):
        # A query specifically asking about Google must never return SkillFirst's headquarters
        queries = [
            "Where is the Google headquarters?",
            "Where is Google headquartered?",
            "Who founded Google?"
        ]
        for q in queries:
            with self.subTest(query=q):
                res = self.kb.search(q)
                self.assertEqual(res["status"], "not_found")
                self.assertEqual(res["matches_count"], 0)
                self.assertEqual(res["detected_entity"], "Google")
                all_facts = " ".join([m.get("fact", "") for m in res.get("results", [])])
                self.assertNotIn("Chennai", all_facts)
                self.assertNotIn("Tiruvallur", all_facts)
                self.assertNotIn("San Francisco", all_facts)
                self.assertIn("specifically asks about 'Google'", res["reason"])

    def test_unknown_company_headquarters_not_inferred(self):
        unknown_queries = [
            "Where is Microsoft headquartered?",
            "Where is Apple located?",
            "Who is the founder of Netflix?"
        ]
        for q in unknown_queries:
            with self.subTest(query=q):
                res = self.kb.search(q)
                self.assertEqual(res["status"], "not_found")
                self.assertEqual(res["matches_count"], 0)

    def test_unknown_stock_price_rejected(self):
        res = self.kb.search("What is the current stock price of Skill First Applied Labs?")
        self.assertEqual(res["status"], "not_found")
        self.assertEqual(res["matches_count"], 0)
        self.assertIn("stock", res["reason"].lower())

    def test_unsupported_policies_rejected(self):
        res = self.kb.search("pet health insurance dental benefit reimbursement policy")
        self.assertEqual(res["status"], "not_found")
        self.assertEqual(res["matches_count"], 0)

    def test_matching_facts_internship_milestones(self):
        res = self.kb.search("internship milestones")
        self.assertEqual(res["status"], "success")
        all_facts = " ".join([m["fact"] for m in res["results"]])
        self.assertIn("CSS Insight Parser", all_facts)
        self.assertIn("Tool-Using AI Agent", all_facts)

    def test_matching_facts_case_insensitivity(self):
        lower_res = self.kb.search("founder")
        upper_res = self.kb.search("FOUNDER")
        mixed_res = self.kb.search("FoUnDeR")
        self.assertEqual(lower_res["matches_count"], upper_res["matches_count"])
        self.assertEqual(lower_res["matches_count"], mixed_res["matches_count"])

    def test_missing_facts_reported_clearly(self):
        missing_queries = [
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
        query = "internship assessment milestones"
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
        res = execute_tool("search_knowledge_base", {"query": "milestones"})
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
                            "args": {"query": "founder"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Skill First Applied Labs was founded by Gurunathan Murugan."}]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)

        answer = agent.run("Who founded the company?")
        self.assertIn("Gurunathan Murugan", answer)
        self.assertEqual(len(agent.history), 4)

    def test_multi_turn_chained_tool_calls(self):
        # Chained reasoning:
        # Step 1: Model calls search_knowledge_base for milestones
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "milestones"}
                        }
                    }]
                }
            }]
        }
        # Step 2: Model sees 23 passing tests for milestone 1 and calls calculate for double
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "calculate",
                            "args": {"expression": "23 * 2"}
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
                    "parts": [{"text": "Milestone 1 has 23 passing unit tests. Doubled, that equals 46 tests."}]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2, turn_3])
        agent = AIAgent(llm_client=mock_client, verbose=False)

        answer = agent.run("How many tests are there in Milestone 1 doubled?")
        self.assertIn("46", answer)
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


# ============================================================================
# 6. Gemini Agent Regression Tests (Offline Mocked LLM)
# ============================================================================

class TestGeminiAgentRegressionSuite(unittest.TestCase):
    """
    Offline regression tests simulating Gemini tool-calling responses to verify:
    1. Arithmetic question -> calculator selected and correct result returned.
    2. Assessment question -> knowledge-search tool selected and evidence returned.
    3. Paraphrased assessment question -> relevant knowledge retrieved.
    4. Unknown company-policy question -> no invented answer (not_found handled).
    5. Tool returns no results -> agent handles it clearly with explicit reason.
    6. Gemini requests tool, receives result, produces final answer (multi-turn verification).
    7. Tool/API errors -> readable error handling without crashes or secret leaks.
    """

    def test_regression_1_arithmetic_question_calculator_selected_and_result(self):
        """1. Arithmetic question -> calculator selected and correct result returned."""
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "calculate",
                            "args": {"expression": "(45.5 * 12) + (180 / 4)"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Evaluating (45.5 * 12) + (180 / 4) yields 591.0."}]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("What is (45.5 * 12) + (180 / 4)?")

        last_tools = agent.get_last_tools()
        self.assertEqual(len(last_tools), 1)
        self.assertEqual(last_tools[0]["name"], "calculate")
        self.assertEqual(last_tools[0]["result"]["status"], "success")
        self.assertEqual(last_tools[0]["result"]["result"], 591.0)
        self.assertIn("591", reply)

    def test_regression_2_assessment_question_knowledge_search_selected_and_evidence(self):
        """2. Assessment question -> knowledge-search tool selected and evidence returned."""
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "internship assessment milestones"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": (
                            "The SkillFirst Labs assessment consists of three milestones:\n"
                            "1. Milestone 1: CSS Insight Parser\n"
                            "2. Milestone 2: Tool-Using AI Agent\n"
                            "3. Milestone 3: Multimodal RAG Pipeline."
                        )
                    }]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("What are the internship milestones for the assessment?")

        last_tools = agent.get_last_tools()
        self.assertEqual(len(last_tools), 1)
        self.assertEqual(last_tools[0]["name"], "search_knowledge_base")
        self.assertEqual(last_tools[0]["result"]["status"], "success")
        self.assertGreaterEqual(last_tools[0]["result"]["matches_count"], 1)
        self.assertIn("evidence", last_tools[0]["result"])
        self.assertIn("Milestone 1", last_tools[0]["result"]["evidence"])
        self.assertIn("Milestone 1", reply)
        self.assertIn("Milestone 2", reply)

    def test_regression_3_paraphrased_assessment_question_retrieval(self):
        """3. Paraphrased assessment question -> relevant knowledge retrieved."""
        paraphrased_queries = [
            "How does Milestone 1 work in the assessment?",
            "confirmed requirements for the assessment",
            "what does milestone 2 require for the agent?"
        ]
        for query in paraphrased_queries:
            result = search_knowledge_base(query)
            self.assertEqual(
                result["status"],
                "success",
                f"Failed to retrieve facts for paraphrased query: '{query}'"
            )
            self.assertGreater(result["matches_count"], 0)
            self.assertTrue(len(result["results"]) > 0)

        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "confirmed requirements"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "The confirmed assessment requirements include a pure Python CSS parser, safe AST arithmetic calculator, and grounded knowledge retrieval."}]
                }
            }]
        }
        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("What are the confirmed assessment requirements?")
        self.assertEqual(len(agent.get_last_tools()), 1)
        self.assertEqual(agent.get_last_tools()[0]["result"]["status"], "success")
        self.assertIn("CSS parser", reply)

    def test_regression_4_unknown_company_policy_question_no_hallucination(self):
        """4. Unknown company-policy question -> no invented answer (not_found handled)."""
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "pet health insurance dental benefit policy"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": "I checked the verified knowledge base, but pet health insurance and dental benefits cannot be verified from available records."
                    }]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("What is SkillFirst Labs' reimbursement policy for pet health insurance?")

        last_tools = agent.get_last_tools()
        self.assertEqual(len(last_tools), 1)
        tool_result = last_tools[0]["result"]
        self.assertEqual(tool_result["status"], "not_found")
        self.assertIn("cannot be verified", reply.lower())

    def test_regression_5_tool_returns_no_results_with_explicit_reason(self):
        """5. Tool returns no results -> agent handles it clearly with explicit reason."""
        result = search_knowledge_base("SkillFirst Labs ticker symbol on NYSE stock exchange")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["matches_count"], 0)
        self.assertIn("reason", result)
        self.assertTrue(len(result["reason"]) > 0)

    def test_regression_6_multi_turn_tool_request_result_final_answer(self):
        """6. Gemini requests tool, receives result, produces final answer (multi-turn verification)."""
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "assessment milestones"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "calculate",
                            "args": {"expression": "23 * 2"}
                        }
                    }]
                }
            }]
        }
        turn_3 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": "Milestone 1 has 23 passing tests. Doubled, that equals 46 tests."
                    }]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2, turn_3])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("How many tests are in Milestone 1 doubled?")

        self.assertEqual(len(agent.history), 6)
        tools = agent.get_last_tools()
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "search_knowledge_base")
        self.assertEqual(tools[1]["name"], "calculate")
        self.assertEqual(tools[1]["result"]["result"], 46)
        self.assertIn("46", reply)

    def test_regression_7_tool_and_api_error_handling_no_leaks(self):
        """7. Tool/API errors -> readable error handling without crashes or secret leaks."""
        res_unknown = execute_tool("unregistered_tool", {"arg": "val"})
        self.assertEqual(res_unknown["status"], "error")
        self.assertIn("Unknown tool", res_unknown["error"])

        res_bad_arg = execute_tool("calculate", {"bogus": "value"})
        self.assertEqual(res_bad_arg["status"], "error")
        self.assertIn("Invalid arguments", res_bad_arg["error"])

        dummy_secret = "AIzaSyFakeSecretKey_123456789"
        raw_error_message = f"Gemini API HTTP Error 400: Invalid key {dummy_secret}"
        sanitized_error = raw_error_message.replace(dummy_secret, "[REDACTED]")
        self.assertNotIn(dummy_secret, sanitized_error)
        self.assertIn("[REDACTED]", sanitized_error)

    def test_regression_8_google_headquarters_honest_fallback(self):
        """8. 'Where is Google headquartered?' -> honest fallback, never attributes SkillFirst location."""
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "Where is the Google headquarters?"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": "I couldn't verify Google's headquarters from the available knowledge base."
                    }]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("Where is the Google headquarters?")

        last_tools = agent.get_last_tools()
        self.assertEqual(len(last_tools), 1)
        tool_result = last_tools[0]["result"]
        self.assertEqual(tool_result["status"], "not_found")
        self.assertEqual(tool_result.get("detected_entity"), "Google")

        # Must be an honest fallback
        self.assertIn("couldn't verify google's headquarters", reply.lower())
        # Must never return SkillFirst's locations or San Francisco as Google's location
        self.assertNotIn("chennai", reply.lower())
        self.assertNotIn("tiruvallur", reply.lower())
        self.assertNotIn("san francisco", reply.lower())

    def test_regression_9_skillfirst_headquarters_verified_attribution(self):
        """9. 'Where is SkillFirst Labs headquartered?' -> returns verified location with LinkedIn/website attribution."""
        turn_1 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "search_knowledge_base",
                            "args": {"query": "Where is SkillFirst Labs headquartered?"}
                        }
                    }]
                }
            }]
        }
        turn_2 = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "text": (
                            "Skill First Applied Labs' contact address published on its official website is in Chennai 600 077. "
                            "Additionally, its headquarters is separately attributed via LinkedIn as Tiruvallur, Tamil Nadu (founded in 2024)."
                        )
                    }]
                }
            }]
        }

        mock_client = MockLLMClient([turn_1, turn_2])
        agent = AIAgent(llm_client=mock_client, verbose=False)
        reply = agent.run("Where is SkillFirst Labs headquartered?")

        last_tools = agent.get_last_tools()
        self.assertEqual(len(last_tools), 1)
        tool_result = last_tools[0]["result"]
        self.assertEqual(tool_result["status"], "success")

        # Verify factual grounding
        evidence = tool_result.get("evidence", "")
        self.assertTrue("Tiruvallur" in evidence or "Chennai" in evidence)
        self.assertNotIn("San Francisco", evidence)
        self.assertIn("Chennai", reply)
        self.assertNotIn("San Francisco", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)


