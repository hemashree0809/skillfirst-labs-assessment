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
    Checks both the specified path and the agent's directory to ensure secrets are found.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = []
    if os.path.isabs(filepath):
        candidate_paths.append(filepath)
    else:
        candidate_paths.append(os.path.join(base_dir, filepath))
        candidate_paths.append(os.path.abspath(filepath))
        candidate_paths.append(os.path.join(os.path.dirname(base_dir), filepath))

    resolved_path = None
    for cp in candidate_paths:
        if os.path.isfile(cp):
            resolved_path = cp
            break

    if not resolved_path:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(resolved_path)
        if "GEMINI_API_KEY" in os.environ:
            return
    except ImportError:
        pass

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

KNOWN_HOST_ENTITIES = {
    "skill first applied labs", "skill first", "skillfirst labs", "skillfirst",
    "applied labs", "the company", "this company", "our company", "company", "organization"
}

KNOWN_SUB_ENTITIES = {
    "gurunathan murugan", "gurunathan", "murugan",
    "css insight parser", "insight parser", "css parser",
    "tool-using ai agent", "ai agent",
    "milestone 1", "milestone 2"
}


def extract_query_subject(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts (subject_entity, attribute_target) from a query.
    For example:
    - 'Where is the Google headquarters?' -> ('Google', 'headquarters')
    - 'Where is Google headquartered?' -> ('Google', 'headquartered')
    - 'Where is SkillFirst Labs headquartered?' -> ('SkillFirst Labs', 'headquartered')
    - 'Who is the founder of Skill First Applied Labs?' -> ('Skill First Applied Labs', 'founder')
    - 'Who founded Google?' -> ('Google', 'founder')
    """
    q_clean = query.strip().rstrip("?").strip()

    # 1. Possessive: e.g. "Google's headquarters", "SkillFirst's founder"
    m = re.search(r"\b([A-Za-z0-9\.\-\s]+?)'s\s+(headquarters|headquarter|hq|office|location|founders?|ceo|cto|stipend|stock\s+price|stock|mission|funding|address|email)\b", q_clean, re.I)
    if m:
        return m.group(1).strip(), m.group(2).lower()

    # 2. Attribute of Entity: e.g. "headquarters of Google", "founder of Skill First Applied Labs"
    m = re.search(r"\b(headquarters|headquarter|hq|location|founders?|founder|ceo|cto|stock\s+price|stipend|mission|funding|address|email|phone)\s+of\s+(?:the\s+)?([A-Za-z0-9\.\-\s]+)", q_clean, re.I)
    if m:
        return m.group(2).strip(), m.group(1).lower()

    # 3. Who founded Entity
    m = re.search(r"\bwho\s+(?:founded|started|created|leads|run|runs)\s+(?:the\s+)?([A-Za-z0-9\.\-\s]+)", q_clean, re.I)
    if m:
        return m.group(1).strip(), "founder"

    # 4. Where/who/what is (the) Entity headquarters/headquartered/located/founder...
    m = re.search(r"\b(?:where|who|what)\s+(?:is|are|was|were)\s+(?:the\s+)?([A-Za-z0-9\.\-\s]+?)\s+(headquarters|headquartered|hq|office|located|location|founded|founders?|founder|ceo|cto|stipend|stock\s+price|stock|mission|funding|address|email)\b", q_clean, re.I)
    if m:
        ent = m.group(1).strip()
        if ent.lower() not in ("the", "a", "an", "of", "for", "in", "at", "to"):
            return ent, m.group(2).lower()

    return None, None


def _build_facts_index(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs discrete, verified, source-attributed fact records from knowledge.json.
    Every record includes source URL, source label, and verification status.
    """
    facts = []

    company_base_keywords = [
        "skill", "first", "applied", "labs", "skillfirst", "skill first",
        "applied labs", "skill first applied labs", "skillfirst labs", "company", "organization"
    ]

    comp = data.get("company", {})
    if comp:
        # 1. Organization Identity & Official Website
        org_name = comp.get("name", "Skill First Applied Labs")
        website = comp.get("official_website", "https://www.skillfirstlabs.com/")
        facts.append({
            "topic": "company.identity",
            "entity": "Skill First Applied Labs",
            "attribute": "identity",
            "source": comp.get("source", website),
            "source_label": comp.get("source_label", "Official Website"),
            "verification_status": comp.get("verification_status", "verified_official_website"),
            "keywords": list(set(company_base_keywords + [
                "name", "identity", "official website", "website", "domain", "url"
            ])),
            "fact": f"Organization Name: {org_name}. Official Website: {website}."
        })

        # 2. Mission
        mission_text = comp.get("mission")
        if mission_text:
            m_src = comp.get("mission_source", {})
            facts.append({
                "topic": "company.mission",
                "entity": "Skill First Applied Labs",
                "attribute": "mission",
                "source": m_src.get("source", website),
                "source_label": m_src.get("source_label", "Official Website"),
                "verification_status": m_src.get("verification_status", "verified_official_website"),
                "keywords": list(set(company_base_keywords + [
                    "mission", "purpose", "goal", "vision", "academia", "industry",
                    "academia-industry gap", "talent management", "platform", "academic institutions"
                ])),
                "fact": f"Mission of Skill First Applied Labs: {mission_text}"
            })

        # 3. Platform Capabilities
        capabilities = comp.get("platform_capabilities", [])
        if capabilities:
            cap_str = ", ".join(capabilities)
            c_src = comp.get("platform_capabilities_source", {})
            facts.append({
                "topic": "company.platform_capabilities",
                "entity": "Skill First Applied Labs",
                "attribute": "platform_capabilities",
                "source": c_src.get("source", website),
                "source_label": c_src.get("source_label", "Official Website"),
                "verification_status": c_src.get("verification_status", "verified_official_website"),
                "keywords": list(set(company_base_keywords + [
                    "platform", "capabilities", "features", "skills", "tracking", "validation",
                    "insights", "ai-powered", "student", "student skill tracking", "multi-channel"
                ])),
                "fact": f"Platform Capabilities of Skill First Applied Labs (Official Website): {cap_str}."
            })

        # 4. Founder
        founder_info = comp.get("founder", {})
        if founder_info:
            f_name = founder_info.get("name", "Gurunathan Murugan")
            f_role = founder_info.get("role", "Founder")
            facts.append({
                "topic": "company.founder",
                "entity": "Skill First Applied Labs",
                "attribute": "founder",
                "source": founder_info.get("source", website),
                "source_label": founder_info.get("source_label", "Official Website"),
                "verification_status": founder_info.get("verification_status", "verified_official_website"),
                "keywords": list(set(company_base_keywords + [
                    "founder", "founders", "founded", "gurunathan", "murugan", "gurunathan murugan",
                    "leadership", "executive", "who founded", "creator"
                ])),
                "fact": f"Founder of Skill First Applied Labs: {f_name} ({f_role}, Skill First Applied Labs)."
            })

        # 5. Contact Details (Published on Official Website)
        contact_info = comp.get("contact", {})
        if contact_info:
            addr = contact_info.get("address", "")
            email = contact_info.get("email", "")
            facts.append({
                "topic": "company.contact",
                "entity": "Skill First Applied Labs",
                "attribute": "contact",
                "source": contact_info.get("source", website),
                "source_label": contact_info.get("source_label", "Official Website"),
                "verification_status": contact_info.get("verification_status", "verified_official_website"),
                "keywords": list(set(company_base_keywords + [
                    "contact", "address", "email", "office", "location", "chennai", "thiruverkadu",
                    "sundara cholavaram", "rmk chola gardens", "600 077", "hello@keystone-tms.in",
                    "where is", "located", "headquarters", "headquartered"
                ])),
                "fact": f"Contact Address for Skill First Applied Labs (Official Website): {addr}. Contact Email: {email}."
            })

        # 6. LinkedIn-Attributed Info (Headquarters & Founded Year)
        linkedin_info = comp.get("linkedin_attributed_info", {})
        if linkedin_info:
            hq = linkedin_info.get("headquarters", "")
            founded_yr = linkedin_info.get("founded", "")
            note = linkedin_info.get("note", "")
            facts.append({
                "topic": "company.linkedin_attributed_info",
                "entity": "Skill First Applied Labs",
                "attribute": "headquarters",
                "source": linkedin_info.get("source", "LinkedIn Company Profile"),
                "source_label": linkedin_info.get("source_label", "LinkedIn (Separately Attributed)"),
                "verification_status": linkedin_info.get("verification_status", "verified_linkedin_attributed"),
                "keywords": list(set(company_base_keywords + [
                    "headquarters", "headquartered", "hq", "location", "tiruvallur", "tamil nadu",
                    "founded", "year", "2024", "linkedin", "where is", "located"
                ])),
                "fact": f"Headquarters for Skill First Applied Labs (Separately Attributed via LinkedIn): {hq} (Founded: {founded_yr}). Note: {note}"
            })

    # 7. Internship Assessment Milestones (Supported by Repository Assignment Files)
    assessment = data.get("internship_assessment", {})
    if assessment:
        milestones = assessment.get("milestones", [])
        for m in milestones:
            m_num = str(m.get("milestone"))
            m_name = m.get("name", "")
            m_folder = m.get("folder", "")
            m_status = m.get("status", "")
            m_desc = m.get("description", "")
            facts.append({
                "topic": f"internship_assessment.milestone_{m_num}",
                "entity": "Skill First Applied Labs",
                "attribute": "milestones",
                "source": m.get("source", "Repository Assignment Files"),
                "source_label": m.get("source_label", "Repository Assignment Files"),
                "verification_status": m.get("verification_status", "verified_repo_assignment"),
                "keywords": [
                    "milestone", "milestones", m_num, f"milestone {m_num}", f"milestone{m_num}",
                    "assessment", "internship", "task", "project", m_name.lower(), m_folder.lower(),
                    "parser", "agent"
                ],
                "fact": f"Assessment Milestone {m_num}: {m_name} (Folder: {m_folder}). Status: {m_status}. Description: {m_desc}"
            })

        # Confirmed Requirements
        requirements = assessment.get("confirmed_requirements", [])
        if requirements:
            req_src = assessment.get("confirmed_requirements_source", {})
            req_summary = "; ".join(requirements)
            facts.append({
                "topic": "internship_assessment.confirmed_requirements",
                "entity": "Skill First Applied Labs",
                "attribute": "requirements",
                "source": req_src.get("source", "Repository Assignment Files"),
                "source_label": req_src.get("source_label", "Repository Assignment Files"),
                "verification_status": req_src.get("verification_status", "verified_repo_assignment"),
                "keywords": [
                    "requirements", "confirmed requirements", "criteria", "rubric", "evaluation",
                    "assessment", "rules", "testing", "calculator", "parser", "agent"
                ],
                "fact": f"Confirmed Assessment Requirements (Repository Assignment Files): {req_summary}."
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
        Entity-aware, source-attributed search of the verified knowledge base.
        Validates:
        1. Subject entity match (foreign entities like Google return not_found immediately).
        2. Attribute relevance (evidence must actually answer the question).
        Never invents or hallucinates facts.
        """
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error": "Query string cannot be empty."
            }

        cleaned_query = query.strip()
        query_lower = cleaned_query.lower()

        # Step 1: Entity & Subject Extraction
        subject_entity, attribute_target = extract_query_subject(cleaned_query)

        # If a subject entity was detected, verify if it is an external/unknown entity
        if subject_entity:
            ent_norm = subject_entity.lower().strip()
            is_host_entity = any(h in ent_norm or ent_norm in h for h in KNOWN_HOST_ENTITIES)
            is_sub_entity = any(s in ent_norm or ent_norm in s for s in KNOWN_SUB_ENTITIES)

            if not is_host_entity and not is_sub_entity:
                # Query explicitly asks about a foreign entity (e.g. Google, Microsoft, Apple)
                return {
                    "status": "not_found",
                    "query": cleaned_query,
                    "matches_count": 0,
                    "results": [],
                    "detected_entity": subject_entity,
                    "entity_match": False,
                    "message": (
                        f"No verified facts found in the knowledge base for '{subject_entity}'. "
                        "The requested information is not available in verified records."
                    ),
                    "reason": (
                        f"Query specifically asks about '{subject_entity}', which is an external or unverified entity "
                        "not present in the Skill First Applied Labs knowledge base."
                    )
                }

        # Step 2: Tokenize and filter query
        raw_words = [re.sub(r"[^\w\-]", "", w) for w in query_lower.split()]
        raw_words = [w for w in raw_words if w]

        stop_words = {
            "what", "who", "where", "when", "why", "how", "is", "are", "was",
            "were", "the", "a", "an", "of", "in", "for", "to", "at", "and",
            "or", "on", "with", "does", "do", "did", "tell", "me", "about",
            "can", "you", "please", "give", "find", "show", "which", "there",
            "work", "works", "explain", "describe", "many", "much",
            "current", "present", "latest", "official", "verified", "details", "information"
        }
        domain_words = {"skill", "first", "applied", "labs", "company", "information", "details"}
        attribute_stop_words = {
            "headquarters", "headquarter", "headquartered", "hq", "location", "located",
            "founder", "founders", "founded", "ceo", "cto", "address", "email", "phone"
        }

        # Check for unknown entity tokens not present in any fact or keyword
        all_known_tokens = set()
        for item in self._facts:
            all_known_tokens.update(item["topic"].lower().replace(".", " ").split())
            all_known_tokens.update(item["fact"].lower().split())
            for kw in item.get("keywords", []):
                all_known_tokens.update(str(kw).lower().split())

        unknown_tokens = [
            w for w in raw_words
            if w not in stop_words and w not in domain_words and w not in attribute_stop_words
            and not any(w in known for known in all_known_tokens)
        ]

        if unknown_tokens:
            # Contains unrecorded entity or concept (e.g. "google", "netflix", "stock", "pet", "dental")
            first_unknown = unknown_tokens[0]
            return {
                "status": "not_found",
                "query": cleaned_query,
                "matches_count": 0,
                "results": [],
                "detected_entity": first_unknown.capitalize(),
                "entity_match": False,
                "message": (
                    f"No verified facts found in the knowledge base matching '{cleaned_query}'. "
                    "The requested information is not available in verified records."
                ),
                "reason": (
                    f"Query contains unverified term or entity '{first_unknown}', "
                    "which does not exist in verified Skill First Applied Labs records."
                )
            }

        substantive_tokens = [
            w for w in raw_words
            if w not in stop_words and (len(w) > 1 or w.isdigit())
        ]
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

            # Exact phrase match in fact or topic
            if query_lower in fact_lower or query_lower in topic_lower:
                score += 30.0
                matched_substantive = len(substantive_tokens)

            # Check individual substantive tokens with word boundary matching
            for token in substantive_tokens:
                token_matched = False

                for kw in keywords_lower:
                    if token == kw or (len(token) > 3 and token in kw):
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

            if matched_substantive > 0:
                coverage = matched_substantive / len(substantive_tokens)
                if len(substantive_tokens) > 2 and coverage < 0.35:
                    continue
                scored_matches.append((score * coverage, item))

        scored_matches.sort(key=lambda x: x[0], reverse=True)

        if not scored_matches:
            return {
                "status": "not_found",
                "query": cleaned_query,
                "matches_count": 0,
                "results": [],
                "message": (
                    f"No verified facts found in the knowledge base matching '{cleaned_query}'. "
                    "The requested information is not available in verified records."
                ),
                "reason": f"No entries in knowledge.json matched substantive search terms: {substantive_tokens}."
            }

        # Deduplicate and return top matches with source attribution
        unique_results = []
        seen = set()
        for _, item in scored_matches[:6]:
            if item["fact"] not in seen:
                seen.add(item["fact"])
                unique_results.append({
                    "topic": item["topic"],
                    "fact": item["fact"],
                    "source": item.get("source", ""),
                    "source_label": item.get("source_label", "Verified Record"),
                    "verification_status": item.get("verification_status", "verified")
                })

        evidence_lines = [
            f"- [{item['topic']}] {item['fact']} (Source: {item.get('source_label')} - {item.get('source')})"
            for item in unique_results
        ]
        evidence_summary = "\n".join(evidence_lines)

        return {
            "status": "success",
            "query": cleaned_query,
            "matches_count": len(unique_results),
            "results": unique_results,
            "evidence": evidence_summary
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
        "description": (
            "Searches verified, source-attributed facts about Skill First Applied Labs (https://www.skillfirstlabs.com/), "
            "including organization mission (bridging the academia-industry gap via talent management platform), "
            "platform capabilities (real-time student skill tracking, multi-channel skill validation, AI-powered insights), "
            "founder Gurunathan Murugan, official Chennai contact address, official contact email (hello@keystone-tms.in), "
            "separately attributed LinkedIn details (Tiruvallur, Tamil Nadu headquarters, 2024 founding year), "
            "and repository assessment milestones (Milestone 1 CSS Insight Parser, Milestone 2 Tool-Using AI Agent)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Keywords, entity, or question to look up in the verified knowledge base."
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
    Resilient against parameter naming variations (query, q, question, expression, expr).
    Safely captures all exceptions and returns structured output.
    """
    try:
        if tool_name == "calculate":
            expr_keys = ("expression", "expr", "input", "formula")
            matched_key = next((k for k in expr_keys if k in args), None)
            if matched_key is None:
                return {
                    "status": "error",
                    "error": "Invalid arguments for 'calculate'. Missing required 'expression' parameter."
                }
            expr = args.get(matched_key, "")
            return safe_calculate(str(expr))
        elif tool_name == "search_knowledge_base":
            query_keys = ("query", "q", "question", "keywords", "search_query", "topic")
            matched_key = next((k for k in query_keys if k in args), None)
            if matched_key is None:
                return {
                    "status": "error",
                    "error": "Invalid arguments for 'search_knowledge_base'. Missing required 'query' parameter."
                }
            query = args.get(matched_key, "")
            return search_knowledge_base(str(query))
        else:
            return {
                "status": "error",
                "error": f"Unknown tool '{tool_name}'. Available tools: ['calculate', 'search_knowledge_base']"
            }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Execution error in tool '{tool_name}': {str(exc)}"
        }


# ============================================================================
# Gemini API Client (Direct REST client with standard library)
# ============================================================================

SYSTEM_INSTRUCTION = """You are the official AI Agent for Skill First Applied Labs (https://www.skillfirstlabs.com/).
You have access to two tools:
1. 'calculate': Use this tool for ANY AND ALL mathematical and arithmetic calculations (e.g. addition, subtraction, multiplication, division, percentages). Never compute calculations manually.
2. 'search_knowledge_base': Use this tool for ANY questions about Skill First Applied Labs, official website facts (mission, platform capabilities, founder Gurunathan Murugan, Chennai contact address, hello@keystone-tms.in), separately attributed LinkedIn details (Tiruvallur, Tamil Nadu headquarters, 2024 founding year), and confirmed repository assessment milestones (Milestone 1 CSS Insight Parser, Milestone 2 Tool-Using AI Agent). Always consult this tool to retrieve verified evidence before answering.

HALLUCINATION PREVENTION & ENTITY INTEGRITY RULES:
- Never invent company facts, stock prices, credentials, unsupported policies, or assessment milestones.
- CRITICAL ENTITY INTEGRITY: The knowledge base is strictly about Skill First Applied Labs. Never use facts about Skill First Applied Labs to answer questions about a different company or entity (e.g. Google, Microsoft, Apple, Amazon, Meta, Netflix).
- If the question asks about an entity not present in verified records (e.g. Google), or if 'search_knowledge_base' returns 'not_found', you MUST provide an honest fallback stating that the information cannot be verified from the available knowledge base (e.g., "I couldn't verify Google's headquarters from the available knowledge base.").
- Never substitute, infer, or fabricate an answer for an unknown entity using another organization's data.
- Ground all responses strictly in the verified evidence returned by 'search_knowledge_base'. Attribute sources accurately (e.g. distinguishing official website facts from separately attributed LinkedIn information).
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
        self.last_run_tools: List[Dict[str, Any]] = []

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
        """Clears conversation history and tool run records."""
        self.history = []
        self.last_run_tools = []

    def get_last_tools(self) -> List[Dict[str, Any]]:
        """Returns the list of tools executed in the most recent run() call."""
        return list(self.last_run_tools)

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
        self.last_run_tools = []
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
                # Post-processing entity guardrail: if tool execution returned not_found for an external entity,
                # ensure the response does not falsely assert host company locations or facts.
                for t in self.last_run_tools:
                    res = t.get("result", {})
                    if res.get("status") == "not_found" and res.get("detected_entity"):
                        ent = res["detected_entity"]
                        if ent.lower() not in ("skillfirst", "skill first", "skill first applied labs", "company"):
                            if any(loc in final_answer.lower() for loc in ["chennai", "tiruvallur", "san francisco", "thiruverkadu"]):
                                final_answer = f"I couldn't verify {ent}'s headquarters from the available knowledge base."

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

                self.last_run_tools.append({
                    "name": tool_name,
                    "args": args,
                    "result": tool_result
                })

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
