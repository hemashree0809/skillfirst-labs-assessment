"""
Milestone 2: Tool-Using AI Agent
SkillFirst Labs - AI/ML Engineering Assessment

This module implements an autonomous, tool-calling AI agent equipped with:
1. Safe AST-based calculator (no unrestricted eval)
2. Verified knowledge-base retrieval (grounded in knowledge.json)
3. Multi-turn tool execution loop with infinite loop guardrails
4. Deterministic tool execution and hallucination prevention
5. Native Gemini tool-calling interface (REST and SDK compatible)
"""

import ast
import json
import operator
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ============================================================================
# Environment Configuration Helper
# ============================================================================

def load_env(filepath: str = ".env") -> None:
    """
    Loads key-value pairs from a .env file into os.environ.
    Uses python-dotenv if installed, otherwise falls back to standard file parsing.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(filepath)
        return
    except ImportError:
        pass

    # Standard library fallback
    if not os.path.isabs(filepath):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = os.path.join(base_dir, filepath)
    else:
        resolved_path = filepath

    if os.path.isfile(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val


# ============================================================================
# Tool 1: Safe AST-Based Arithmetic Calculator
# ============================================================================

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    """
    Recursively evaluates an AST node strictly restricted to numeric math operations.
    Raises ValueError or ZeroDivisionError on disallowed syntax or math violations.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}. Only numbers are permitted.")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")

        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        # Division by zero checks
        if op_type in (ast.Div, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Division by zero is undefined.")

        # Exponent magnitude limit to prevent CPU/memory exhaustion
        if op_type == ast.Pow:
            if abs(right) > 1000:
                raise ValueError(f"Exponent magnitude {right} exceeds safety limit of 1000.")

        return SAFE_OPERATORS[op_type](left, right)

    # Any other AST node (Call, Name, Attribute, Import, List, Dict, etc.) is rejected
    raise ValueError(f"Disallowed syntax element: {type(node).__name__}. Arbitrary code execution is blocked.")


def safe_calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluates an arithmetic expression without unrestricted eval().
    Supports:
      - Addition (+), Subtraction (-), Multiplication (*), Division (/)
      - Decimals, Parentheses (), Modulo (%), Exponents (**)
      - Unary signs (+, -)
    Returns:
      {"status": "success", "result": <number>} or
      {"status": "error", "error": <message>}
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"status": "error", "error": "Empty arithmetic expression."}

    cleaned = expression.strip()

    try:
        # Parse expression into AST in eval mode
        parsed = ast.parse(cleaned, mode="eval")
        result = _eval_ast_node(parsed)

        # Clean formatting: convert whole floats (e.g. 15.0) to int for readability
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 8)

        return {"status": "success", "expression": cleaned, "result": result}

    except ZeroDivisionError as zde:
        return {"status": "error", "expression": cleaned, "error": str(zde)}
    except SyntaxError:
        return {"status": "error", "expression": cleaned, "error": f"Syntax error in expression: '{cleaned}'"}
    except ValueError as ve:
        return {"status": "error", "expression": cleaned, "error": str(ve)}
    except Exception as exc:
        return {"status": "error", "expression": cleaned, "error": f"Evaluation error: {str(exc)}"}


# ============================================================================
# Tool 2: Grounded Knowledge-Base Search
# ============================================================================

def _build_facts_index(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs discrete, semantically grouped fact records from knowledge.json.
    Ensures that entities (e.g. founders, milestones, products) are preserved
    as complete, coherent facts rather than fragmented single-key pieces.
    """
    facts = []

    # 1. Company Overview & Details
    if "company" in data:
        comp = data["company"]
        overview_text = (
            f"SkillFirst Labs was founded in {comp.get('founded')}. "
            f"Headquarters: {comp.get('headquarters')}. "
            f"Mission: {comp.get('mission')} "
            f"Work model: {comp.get('work_model')}. "
            f"Funding: {comp.get('funding')}."
        )
        facts.append({
            "topic": "company.overview",
            "keywords": ["company", "overview", "founded", "founding", "headquarters", "mission", "location", "funding", "work", "model"],
            "fact": overview_text
        })

        # Founders
        for f in comp.get("founders", []):
            f_text = f"Co-Founder: {f.get('name')} ({f.get('role')}). Background: {f.get('background')}"
            facts.append({
                "topic": "company.founders",
                "keywords": ["founder", "founders", "co-founder", "leadership", "executive", f.get("name", "").lower()],
                "fact": f_text
            })

    # 2. Internship Program
    if "internship_program" in data:
        prog = data["internship_program"]
        prog_text = (
            f"Program: {prog.get('title')} ({prog.get('cohort')}). "
            f"Duration: {prog.get('duration_weeks')} weeks. "
            f"Stipend: {prog.get('stipend')}. "
            f"Mentorship: {prog.get('mentorship')}."
        )
        facts.append({
            "topic": "internship_program.overview",
            "keywords": ["internship", "program", "cohort", "duration", "weeks", "stipend", "pay", "salary", "compensation", "mentorship"],
            "fact": prog_text
        })

        # Milestones
        for m in prog.get("milestones", []):
            m_text = (
                f"Milestone {m.get('milestone')}: {m.get('name')} (Folder: {m.get('folder')}). "
                f"Status: {m.get('status')}. Description: {m.get('description')}"
            )
            facts.append({
                "topic": f"internship_program.milestone_{m.get('milestone')}",
                "keywords": ["milestone", "milestones", str(m.get("milestone")), m.get("name", "").lower(), m.get("folder", "").lower()],
                "fact": m_text
            })

        # Evaluation Criteria
        for crit in prog.get("evaluation_criteria", []):
            facts.append({
                "topic": "internship_program.evaluation_criteria",
                "keywords": ["evaluation", "criteria", "rubric", "grading", "assessment", "rules"],
                "fact": f"Evaluation Criterion: {crit}"
            })

    # 3. Products
    for p in data.get("products", []):
        p_text = f"Product: {p.get('name')} (v{p.get('version')}, {p.get('category')}). Description: {p.get('description')}"
        facts.append({
            "topic": f"products.{p.get('id', '')}",
            "keywords": ["product", "products", p.get("id", "").lower(), p.get("name", "").lower(), p.get("category", "").lower()],
            "fact": p_text
        })

    # 4. Security & Policies
    if "security_and_policies" in data:
        sec = data["security_and_policies"]
        for k, v in sec.items():
            if isinstance(v, list):
                v_str = ", ".join(str(item) for item in v)
            else:
                v_str = str(v)
            facts.append({
                "topic": f"security_and_policies.{k}",
                "keywords": ["security", "policy", "policies", k.replace("_", " "), k.lower()],
                "fact": f"{k.replace('_', ' ').capitalize()}: {v_str}"
            })

    # 5. Contact & Support
    if "contact_and_support" in data:
        cs = data["contact_and_support"]
        for k, v in cs.items():
            facts.append({
                "topic": f"contact_and_support.{k}",
                "keywords": ["contact", "support", "email", "office", "hours", k.replace("_", " ")],
                "fact": f"{k.replace('_', ' ').capitalize()}: {v}"
            })

    # 6. FAQ
    for idx, item in enumerate(data.get("faq", [])):
        q = item.get("question", "")
        a = item.get("answer", "")
        facts.append({
            "topic": f"faq[{idx}]",
            "keywords": ["faq", "question", "help"] + q.lower().split(),
            "fact": f"Q: {q} A: {a}"
        })

    return facts


class KnowledgeBase:
    """Manages loading and grounded keyword querying of knowledge.json."""

    def __init__(self, json_path: Optional[str] = None):
        if json_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, "knowledge.json")
        self.json_path = json_path
        self._raw_data: Optional[Dict[str, Any]] = None
        self._facts: List[Dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        """Loads and indexes facts from knowledge.json."""
        if not os.path.isfile(self.json_path):
            self._raw_data = {}
            self._facts = []
            return

        with open(self.json_path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)

        self._facts = _build_facts_index(self._raw_data)

    @property
    def raw_data(self) -> Dict[str, Any]:
        return self._raw_data or {}

    def search(self, query: str) -> Dict[str, Any]:
        """
        Searches the knowledge base for facts matching the query keywords.
        Returns matching information if found, or a clear not-found notification.
        Never invents or hallucinates facts.
        """
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error": "Query string cannot be empty."
            }

        cleaned_query = query.strip()
        query_lower = cleaned_query.lower()

        # Extract words from query
        raw_words = [re.sub(r"[^\w\-]", "", w) for w in query_lower.split()]
        raw_words = [w for w in raw_words if w]

        stop_words = {
            "what", "who", "where", "when", "why", "how", "is", "are", "was",
            "were", "the", "a", "an", "of", "in", "for", "to", "at", "and",
            "or", "on", "with", "does", "do", "did", "tell", "me", "about",
            "can", "you", "please", "give", "find", "show", "which", "there"
        }
        domain_words = {"skillfirst", "labs", "company", "information", "details"}

        # Substantive query terms excluding conversational stopwords and domain brand words
        substantive_tokens = [w for w in raw_words if w not in stop_words and w not in domain_words and len(w) > 1]

        # If user only asked domain words (e.g. "SkillFirst Labs" or "company"), fall back to them
        if not substantive_tokens:
            substantive_tokens = [w for w in raw_words if w not in stop_words and len(w) > 1]
        if not substantive_tokens:
            substantive_tokens = [query_lower]

        scored_matches: List[Tuple[float, Dict[str, Any]]] = []

        for item in self._facts:
            fact_text = item["fact"]
            fact_lower = fact_text.lower()
            topic_lower = item["topic"].lower()
            keywords_lower = [str(k).lower() for k in item.get("keywords", [])]

            matched_substantive = 0
            score = 0.0

            # Exact phrase match
            if query_lower in fact_lower or query_lower in topic_lower:
                score += 25.0
                matched_substantive = len(substantive_tokens)

            # Check individual substantive tokens with word boundary matching
            for token in substantive_tokens:
                token_matched = False

                for kw in keywords_lower:
                    if token in kw or kw in token:
                        score += 10.0
                        token_matched = True
                        break

                if re.search(rf"\b{re.escape(token)}\b", topic_lower):
                    score += 8.0
                    token_matched = True

                if re.search(rf"\b{re.escape(token)}\b", fact_lower):
                    score += 5.0
                    token_matched = True

                if token_matched:
                    matched_substantive += 1

            # Require sufficient substantive token match coverage
            if matched_substantive > 0:
                coverage = matched_substantive / len(substantive_tokens)
                if len(substantive_tokens) > 2 and coverage < 0.4:
                    continue
                scored_matches.append((score * coverage, item))

        scored_matches.sort(key=lambda x: x[0], reverse=True)

        if not scored_matches:
            return {
                "status": "not_found",
                "query": cleaned_query,
                "message": (
                    f"No verified facts found in the knowledge base matching '{cleaned_query}'. "
                    "The requested information is not available in verified records."
                )
            }

        # Deduplicate and return top matches
        unique_results = []
        seen = set()
        for _, item in scored_matches[:6]:
            if item["fact"] not in seen:
                seen.add(item["fact"])
                unique_results.append({
                    "topic": item["topic"],
                    "fact": item["fact"]
                })

        return {
            "status": "success",
            "query": cleaned_query,
            "matches_count": len(unique_results),
            "results": unique_results
        }


# Global knowledge base instance
_KB_INSTANCE: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _KB_INSTANCE
    if _KB_INSTANCE is None:
        _KB_INSTANCE = KnowledgeBase()
    return _KB_INSTANCE


def search_knowledge_base(query: str) -> Dict[str, Any]:
    """Exposed agent tool function for searching verified knowledge."""
    kb = get_knowledge_base()
    return kb.search(query)


# ============================================================================
# Tool Declarations & Dispatch
# ============================================================================

TOOL_DECLARATIONS = [
    {
        "name": "calculate",
        "description": "Safely evaluates an arithmetic expression containing numbers, decimals, parentheses, and operators (+, -, *, /).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "expression": {
                    "type": "STRING",
                    "description": "The mathematical expression to evaluate, e.g. '(12.5 * 4) + (100 / 5)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Searches verified company facts, products, internship details, security policies, and FAQs from the local knowledge base.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Keywords or questions to look up in the verified knowledge base."
                }
            },
            "required": ["query"]
        }
    }
]

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "calculate": lambda expression: safe_calculate(expression),
    "search_knowledge_base": lambda query: search_knowledge_base(query),
}


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a tool by name with provided arguments.
    Safely captures all exceptions and returns structured output.
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "status": "error",
            "error": f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
        }

    func = TOOL_REGISTRY[tool_name]
    try:
        return func(**args)
    except TypeError as te:
        return {
            "status": "error",
            "error": f"Invalid arguments for tool '{tool_name}': {str(te)}"
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Execution error in tool '{tool_name}': {str(exc)}"
        }


# ============================================================================
# Gemini API Client (Direct REST client with standard library)
# ============================================================================

SYSTEM_INSTRUCTION = """You are the official SkillFirst Labs AI Agent.
You have access to two tools:
1. 'calculate': Use this tool for ANY AND ALL mathematical and arithmetic calculations (e.g. addition, subtraction, multiplication, division, percentages). Never compute calculations manually.
2. 'search_knowledge_base': Use this tool for questions about SkillFirst Labs, company background, founders, products, technologies, internship programs, milestones, security policies, and FAQs.

HALLUCINATION PREVENTION RULES:
- Never invent company facts, statistics, credentials, or policies.
- If the knowledge base does not contain the requested information or returns 'not_found', you MUST explicitly report that the information cannot be verified from available records. Never pretend unverified assumptions are facts.
- Present tool results accurately and faithfully.
"""


class GeminiRESTClient:
    """
    Lightweight, zero-external-dependency Gemini REST client.
    Supports official Google Gemini v1beta generateContent API with tool calling.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def generate_content(
        self,
        contents: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Sends a request to the Gemini generateContent API and returns parsed JSON response."""
        url = f"{self.base_url}?key={self.api_key}"

        payload: Dict[str, Any] = {
            "contents": contents
        }

        if tools:
            payload["tools"] = [{"function_declarations": tools}]

        if system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction}]
            }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP Error {http_err.code}: {error_body}") from http_err
        except urllib.error.URLError as url_err:
            raise RuntimeError(f"Gemini API Network Error: {str(url_err.reason)}") from url_err


# ============================================================================
# AI Agent Implementation
# ============================================================================

class AIAgent:
    """
    Autonomous tool-calling AI agent.
    Coordinates user conversation, model invocations, tool execution, and guardrails.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: int = 5,
        llm_client: Optional[Any] = None,
        verbose: bool = True
    ):
        load_env()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.max_turns = max_turns
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []

        # Use injected client (e.g. for testing) or REST client
        if llm_client is not None:
            self.client = llm_client
        else:
            self.client = None

    def _get_active_client(self) -> Any:
        """Returns the active LLM client, verifying API key presence."""
        if self.client is not None:
            return self.client

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set the GEMINI_API_KEY environment variable "
                "or define it in an .env file. See .env.example for details."
            )

        self.client = GeminiRESTClient(api_key=self.api_key, model=self.model)
        return self.client

    def reset(self) -> None:
        """Clears conversation history."""
        self.history = []

    def run(self, user_prompt: str) -> str:
        """
        Runs the agentic execution loop for a user query:
        1. Appends prompt to history.
        2. Calls LLM with tool declarations.
        3. Parses model response:
           - If text output: returns final answer.
           - If tool call: executes Python tool, passes result back to LLM.
        4. Enforces max_turns guardrail to prevent infinite loops.
        """
        if not user_prompt or not user_prompt.strip():
            return "Please provide a non-empty question or instruction."

        client = self._get_active_client()

        # Append user turn
        self.history.append({
            "role": "user",
            "parts": [{"text": user_prompt.strip()}]
        })

        turn_count = 0

        while turn_count < self.max_turns:
            turn_count += 1

            response_data = client.generate_content(
                contents=self.history,
                tools=TOOL_DECLARATIONS,
                system_instruction=SYSTEM_INSTRUCTION
            )

            candidates = response_data.get("candidates", [])
            if not candidates:
                return "Model returned no response candidates."

            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # Extract text parts and functionCall parts
            text_outputs = []
            function_calls = []

            for part in parts:
                if "text" in part:
                    text_outputs.append(part["text"])
                if "functionCall" in part:
                    function_calls.append(part["functionCall"])

            # If no tool calls, model returned its final answer
            if not function_calls:
                final_answer = "\n".join(text_outputs).strip()
                # Record model's response in history
                self.history.append({
                    "role": "model",
                    "parts": parts
                })
                return final_answer

            # If model requested function calls, append model turn to history
            self.history.append({
                "role": "model",
                "parts": parts
            })

            # Execute each requested tool call and build function response parts
            response_parts = []
            for call in function_calls:
                tool_name = call.get("name", "")
                args = call.get("args", {})

                if self.verbose:
                    print(f"  [Tool Invocation] {tool_name}({args})")

                tool_result = execute_tool(tool_name, args)

                if self.verbose:
                    print(f"  [Tool Result] {tool_result}")

                response_parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": tool_result
                    }
                })

            # Send function responses back in the next conversation turn
            # In Gemini REST API v1beta, function responses can be sent under role 'function' or 'user'
            self.history.append({
                "role": "user",
                "parts": response_parts
            })

        # Loop limit reached guardrail
        limit_msg = (
            f"Execution halted: Reached safety limit of {self.max_turns} tool-calling rounds. "
            "Halting to prevent infinite loops."
        )
        if self.verbose:
            print(f"  [Safety Guard] {limit_msg}")
        return limit_msg


# ============================================================================
# Command-Line Interface (CLI)
# ============================================================================

def print_banner() -> None:
    print("=" * 65)
    print("  SkillFirst Labs AI Agent — Milestone 2")
    print("  Safe AST Calculator & Grounded Knowledge Retrieval")
    print("=" * 65)


def cli_main() -> None:
    """Entry point for CLI interaction and local tool testing."""
    args = sys.argv[1:]

    # Direct tool testing flags (no API key required)
    if args and args[0] in ("--calc", "-c"):
        expr = " ".join(args[1:]) if len(args) > 1 else "12.5 * 4 + (10 - 2) / 4"
        res = safe_calculate(expr)
        print(f"Expression: {expr}")
        print(f"Result: {json.dumps(res, indent=2)}")
        return

    if args and args[0] in ("--search", "-s"):
        query = " ".join(args[1:]) if len(args) > 1 else "founders"
        res = search_knowledge_base(query)
        print(f"Query: {query}")
        print(f"Result: {json.dumps(res, indent=2)}")
        return

    print_banner()

    agent = AIAgent(verbose=True)

    # Single-query execution
    if args:
        user_query = " ".join(args)
        print(f"\nUser: {user_query}")
        try:
            answer = agent.run(user_query)
            print(f"\nAgent:\n{answer}\n")
        except ValueError as ve:
            print(f"\nConfiguration Error: {ve}")
        except Exception as exc:
            print(f"\nError: {exc}")
        return

    # Interactive REPL
    print("Interactive session started.")
    print("Commands:")
    print("  'exit' or 'quit' -> Exit the application")
    print("  'reset'          -> Clear conversation history")
    print("  'tools'          -> List available tools\n")

    while True:
        try:
            user_input = input("User> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("Conversation history reset.")
            continue

        if user_input.lower() == "tools":
            print("\nAvailable Tools:")
            for tool in TOOL_DECLARATIONS:
                print(f" - {tool['name']}: {tool['description']}")
            print()
            continue

        try:
            response = agent.run(user_input)
            print(f"\nAgent:\n{response}\n")
        except ValueError as ve:
            print(f"\nSetup Notice: {ve}\n")
        except Exception as exc:
            print(f"\nExecution Error: {exc}\n")


if __name__ == "__main__":
    cli_main()
