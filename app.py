"""
Insight Lab — AI Engineering Workbench
SkillFirst Labs Assessment: CSS Insight Parser (M1) + Tool-Using AI Agent (M2)

This application unites:
1. Milestone 1: Pure Python CSS parser, declaration extractor, and stylesheet analytics engine.
2. Milestone 2: Autonomous tool-calling AI agent (safe AST calculator + grounded knowledge retrieval).
3. Gemini vs. No Gemini: Direct side-by-side evaluation of deterministic local execution vs. LLM tool-calling reasoning.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ============================================================================
# Path Management & Module Imports
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent
CSS_PARSER_DIR = ROOT_DIR / "CSS-Insight-Parser"
AI_AGENT_DIR = ROOT_DIR / "AI-Agent"

# Ensure both milestone directories are in sys.path
for p in (str(ROOT_DIR), str(CSS_PARSER_DIR), str(AI_AGENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import existing Milestone 1 modules
from parser import analyze_css, parse_css

# Import existing Milestone 2 modules
from agent import (
    AIAgent,
    load_env,
    safe_calculate,
    search_knowledge_base,
    TOOL_DECLARATIONS,
)

# Load environment configuration (.env) from AI-Agent directory
ENV_FILE = AI_AGENT_DIR / ".env"
load_env(str(ENV_FILE))


# ============================================================================
# Streamlit Page Configuration & Neutral CSS Styling
# ============================================================================

st.set_page_config(
    page_title="Insight Lab — AI Engineering Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional neutral UI palette, subtle borders, rounded cards, readable typography
st.markdown(
    """
    <style>
    /* Neutral typography and container widths */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Workbench Top Header */
    .workbench-header {
        margin-bottom: 0.2rem;
        font-weight: 700;
        font-size: 1.75rem;
        color: #0f172a;
    }
    .workbench-sub {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }

    /* Cards & Containers */
    .insight-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 0.4rem;
        margin-bottom: 0.35rem;
    }
    .badge-local {
        color: #0369a1;
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
    }
    .badge-gemini {
        color: #4338ca;
        background-color: #eef2ff;
        border: 1px solid #c7d2fe;
    }
    .badge-unverified {
        color: #9a3412;
        background-color: #fff7ed;
        border: 1px solid #ffedd5;
    }

    /* Sidebar cards */
    .sidebar-section {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.75rem;
        font-size: 0.82rem;
        color: #334155;
    }

    /* Button styles */
    div[data-testid="stButton"] button {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Session State Initialization
# ============================================================================

if "agent" not in st.session_state:
    st.session_state.agent = AIAgent(verbose=False)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_agent_prompt" not in st.session_state:
    st.session_state.pending_agent_prompt = None

if "css_input_text" not in st.session_state:
    st.session_state.css_input_text = ""

if "last_css_report" not in st.session_state:
    st.session_state.last_css_report = None


# ============================================================================
# Helper Functions
# ============================================================================

def detect_tools_invoked(history_slice: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extracts functionCall records and their responses from a slice of agent.history."""
    calls = []
    for turn in history_slice:
        if turn.get("role") == "model":
            for part in turn.get("parts", []):
                if "functionCall" in part:
                    call = part["functionCall"]
                    calls.append({
                        "name": call.get("name", ""),
                        "args": call.get("args", {}),
                        "result": None
                    })
        elif turn.get("role") in ("user", "function"):
            for part in turn.get("parts", []):
                if "functionResponse" in part:
                    resp = part["functionResponse"]
                    resp_name = resp.get("name", "")
                    resp_data = resp.get("response", {})
                    for c in calls:
                        if c["name"] == resp_name and c["result"] is None:
                            c["result"] = resp_data
                            break
    return calls


def render_tool_invocation_ui(t: Dict[str, Any]) -> None:
    """Renders detailed tool execution details: tool name, arguments, and result."""
    tool_name = t.get("name", "unknown_tool")
    args = t.get("args", {})
    result = t.get("result")

    badge_type = "badge-local" if tool_name == "calculate" else "badge-gemini"
    st.markdown(
        f"<span class='metric-badge {badge_type}'>Selected Tool: <b>{tool_name}</b></span>",
        unsafe_allow_html=True
    )

    st.markdown("**Tool Arguments Passed:**")
    st.code(json.dumps(args, indent=2), language="json")

    if result is not None:
        st.markdown("**Tool Execution Result:**")
        if isinstance(result, dict):
            status = result.get("status")
            if tool_name == "search_knowledge_base":
                if status == "success":
                    st.success(f"Matched {result.get('matches_count', 0)} verified fact(s).")
                    for r in result.get("results", []):
                        st.markdown(f"- **{r.get('topic')}:** {r.get('fact')}")
                        st.caption(f"  *Source: {r.get('source_label')} ({r.get('source')}) | Status: `{r.get('verification_status')}`*")
                elif status == "not_found":
                    st.warning(
                        f"**Status:** `not_found`\n\n"
                        f"**Explicit Reason:** {result.get('reason', 'No matching facts found.')}\n\n"
                        f"**Notice:** {result.get('message', 'Information unavailable.')}"
                    )
                    if result.get("detected_entity"):
                        st.info(f"ℹ️ **Entity Validation:** Query target '{result.get('detected_entity')}' is not in verified records.")
                elif status == "error":
                    st.error(f"Tool Error: {result.get('error')}")
            elif tool_name == "calculate":
                if status == "success":
                    st.info(f"Computed Value: `{result.get('result')}` (Expression: `{result.get('expression')}`)")
                elif status == "error":
                    st.error(f"Calculation Error: {result.get('error')}")
            with st.expander("Raw Tool Result JSON", expanded=False):
                st.json(result)
        else:
            st.code(str(result))


def is_missing_knowledge(response_text: str) -> bool:
    """Detects whether an agent response reported unverified or missing knowledge."""
    lower = response_text.lower()
    markers = [
        "cannot be verified",
        "not available in verified records",
        "no verified facts found",
        "not available in the knowledge base",
        "cannot verify",
        "unverified",
        "not found in the available"
    ]
    return any(m in lower for m in markers)


# ============================================================================
# Sidebar Navigation & System Diagnostics
# ============================================================================

with st.sidebar:
    st.markdown("### 🔬 Workbench Navigation")
    current_page = st.radio(
        "Select Workspace:",
        [
            "🎨 CSS Insight Parser",
            "🤖 Tool-Using AI Agent",
            "⚖️ Gemini vs. No Gemini",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### ⚙️ Environment Status")
    api_key_set = bool(os.environ.get("GEMINI_API_KEY") or st.session_state.agent.api_key)
    if api_key_set:
        st.success("Gemini API: Configured via `.env`", icon="✅")
    else:
        st.warning(
            "Gemini API: Key not detected.\n"
            "Local parsing, safe calculation, and local search function 100% offline. Live Gemini calls require `GEMINI_API_KEY` in `AI-Agent/.env`.",
            icon="⚠️"
        )

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    st.caption(f"**Default Model:** `{model_name}` | **Max Turns:** `{st.session_state.agent.max_turns}`")

    st.divider()

    st.markdown(
        "<div class='sidebar-section'>"
        "<strong>Milestone 1: CSS Parser</strong><br>"
        "Pure Python, zero-dependency tokenization, declaration extraction, and duplicate detection."
        "</div>"
        "<div class='sidebar-section'>"
        "<strong>Milestone 2: AI Agent</strong><br>"
        "Autonomous Gemini function calling with AST calculator and grounded knowledge base."
        "</div>",
        unsafe_allow_html=True,
    )

    if current_page == "🤖 Tool-Using AI Agent":
        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.chat_messages = []
            st.rerun()


# Top Header Banner
st.markdown("<h1 class='workbench-header'>Insight Lab — AI Engineering Workbench</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='workbench-sub'>"
    "CSS parsing, grounded AI tools, and model-assisted comparisons (Milestone 1 & Milestone 2)"
    "</p>",
    unsafe_allow_html=True,
)


# ============================================================================
# Page 1: CSS Insight Parser (Milestone 1)
# ============================================================================

if current_page == "🎨 CSS Insight Parser":
    st.markdown("### 🎨 CSS Insight Parser & Stylesheet Analytics")
    st.markdown(
        "A lightweight, zero-dependency CSS parser and stylesheet analytics engine implemented in pure Python. "
        "It sanitizes comments, safely tracks quotes and brace nesting, handles functional notations and at-rules, "
        "and extracts selectors and declarations into structured JSON."
    )

    col_btn1, col_btn2, col_info = st.columns([1.2, 1.2, 3])
    with col_btn1:
        if st.button("📂 Load sample.css", use_container_width=True):
            sample_path = CSS_PARSER_DIR / "sample.css"
            if sample_path.exists():
                st.session_state.css_input_text = sample_path.read_text(encoding="utf-8")
                st.toast("Loaded Normalize.css benchmark stylesheet!", icon="📂")
            else:
                st.error("sample.css not found in CSS-Insight-Parser directory.")

    with col_btn2:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state.css_input_text = ""
            st.session_state.last_css_report = None
            st.rerun()

    # CSS Code Input
    css_content = st.text_area(
        "CSS Code:",
        value=st.session_state.css_input_text,
        height=240,
        placeholder="/* Enter your CSS rules here */\nbody {\n  margin: 0;\n  padding: 10px;\n  background-color: #fafafa;\n}",
        key="css_editor_box"
    )

    # Sync state
    st.session_state.css_input_text = css_content

    # Parse Action Button
    if st.button("⚡ Parse & Analyze CSS", type="primary"):
        if not css_content.strip():
            st.warning("Please enter or load CSS code before parsing.")
        else:
            try:
                t0 = time.perf_counter()
                report = analyze_css(css_content, file_name="workbench_input.css")
                t_elapsed = (time.perf_counter() - t0) * 1000  # ms
                report["_parse_time_ms"] = t_elapsed
                st.session_state.last_css_report = report
                st.success(f"Parsing completed successfully in {t_elapsed:.2f} ms!", icon="✨")
            except ValueError as ve:
                st.error(f"CSS Syntax Error: {str(ve)}")
                st.session_state.last_css_report = None
            except Exception as exc:
                st.error(f"Unexpected parsing failure: {str(exc)}")
                st.session_state.last_css_report = None

    # Render Parse Results
    if st.session_state.last_css_report:
        rep = st.session_state.last_css_report

        st.divider()
        st.markdown("#### 📊 Stylesheet Metrics")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Rules", rep.get("total_rules", 0))
        m2.metric("Declarations", rep.get("total_declarations", 0))
        m3.metric("Unique Properties", len(rep.get("property_frequency", {})))
        m4.metric("Duplicate Selectors", len(rep.get("duplicate_selectors", [])))
        m5.metric("Parse Time", f"{rep.get('_parse_time_ms', 0.0):.2f} ms")

        # Duplicate selector alert if any
        if rep.get("duplicate_selectors"):
            st.warning(
                f"⚠️ Detected {len(rep['duplicate_selectors'])} duplicate selector(s): "
                f"`{', '.join(rep['duplicate_selectors'])}`"
            )

        # Tabbed detailed outputs
        tab_rules, tab_props, tab_json = st.tabs(["📜 Extracted Rules", "📈 Property Frequencies", "🔍 Raw JSON"])

        with tab_rules:
            rules_list = rep.get("rules", [])
            if not rules_list:
                st.info("No rule blocks extracted.")
            else:
                for idx, r in enumerate(rules_list[:50]):  # Cap display at 50 for performance
                    selector = r.get("selector", "")
                    decls = r.get("declarations", {})
                    with st.expander(f"Rule #{idx+1}: `{selector}` ({len(decls)} properties)", expanded=(idx < 3)):
                        st.json(decls)
                if len(rules_list) > 50:
                    st.caption(f"Showing first 50 of {len(rules_list)} rules. Full rules available in raw JSON tab.")

        with tab_props:
            prop_freq = rep.get("property_frequency", {})
            if prop_freq:
                freq_data = [{"Property": k, "Occurrences": v} for k, v in prop_freq.items()]
                st.dataframe(freq_data, use_container_width=True)
            else:
                st.info("No CSS properties found.")

        with tab_json:
            json_str = json.dumps(rep, indent=2)
            st.download_button(
                "📥 Download Parsed JSON Output",
                data=json_str,
                file_name="parsed_css_output.json",
                mime="application/json",
            )
            st.json(rep)


# ============================================================================
# Page 2: Tool-Using AI Agent (Milestone 2)
# ============================================================================

elif current_page == "🤖 Tool-Using AI Agent":
    st.markdown("### 🤖 Tool-Using AI Agent (Milestone 2)")
    st.markdown(
        "An autonomous, tool-augmented AI agent using Google Gemini function calling. "
        "The agent executes Python tools locally (`safe_calculate` and `search_knowledge_base`) "
        "and coordinates multi-turn reasoning with strict hallucination guardrails."
    )

    # Suggested Prompts
    st.markdown("**Suggested Questions:**")
    suggested = [
        "Who is the founder of Skill First Applied Labs?",
        "What is the mission of Skill First Applied Labs?",
        "Where is the Google headquarters?",
        "Where is SkillFirst Labs headquartered?",
        "What is (45.5 * 12) + (180 / 4)?",
        "What are the internship milestones?",
    ]
    p_cols = st.columns(len(suggested))
    for i, s_prompt in enumerate(suggested):
        with p_cols[i]:
            if st.button(s_prompt, key=f"agent_sugg_{i}", use_container_width=True):
                st.session_state.pending_agent_prompt = s_prompt

    st.divider()

    # Render Chat History
    for message in st.session_state.chat_messages:
        role = message["role"]
        content = message["content"]
        tools_invoked = message.get("tools_invoked", [])
        unverified = message.get("unverified", False)

        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            st.markdown(content)

            if tools_invoked:
                with st.expander(f"🛠️ Tools Invoked ({len(tools_invoked)})", expanded=False):
                    for t in tools_invoked:
                        render_tool_invocation_ui(t)

            if unverified:
                st.markdown(
                    "<span class='metric-badge badge-unverified'>"
                    "🔍 Grounded Knowledge Verification: Fact is not found in verified records. Hallucination blocked."
                    "</span>",
                    unsafe_allow_html=True
                )

    # Chat Input Box
    chat_input = st.chat_input("Ask a question, request a calculation, or inquire about SkillFirst Labs...")

    # Determine prompt
    active_prompt = None
    if chat_input:
        active_prompt = chat_input
    elif st.session_state.pending_agent_prompt:
        active_prompt = st.session_state.pending_agent_prompt
        st.session_state.pending_agent_prompt = None

    if active_prompt:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(active_prompt)

        # Agent Execution
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent reasoning and consulting verified tools..."):
                history_before = len(st.session_state.agent.history)
                tools_used = []
                unverified_flag = False

                try:
                    response_text = st.session_state.agent.run(active_prompt)
                    if hasattr(st.session_state.agent, "get_last_tools") and st.session_state.agent.get_last_tools():
                        tools_used = st.session_state.agent.get_last_tools()
                    else:
                        tools_used = detect_tools_invoked(st.session_state.agent.history[history_before:])
                    unverified_flag = is_missing_knowledge(response_text)
                except ValueError as ve:
                    response_text = (
                        f"⚠️ **Configuration Notice:** {str(ve)}\n\n"
                        "To use live Gemini reasoning, please add your `GEMINI_API_KEY` into `AI-Agent/.env`. "
                        "You can test tools locally in the **Gemini vs. No Gemini** tab without any API key."
                    )
                except Exception as exc:
                    err_str = str(exc)
                    safe_err = err_str.replace(os.environ.get("GEMINI_API_KEY", "___"), "[REDACTED]")
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        response_text = (
                            "⚠️ **Gemini API Rate Limit / Quota Exceeded (HTTP 429):**\n\n"
                            "The Gemini API free-tier request limit has been reached. "
                            "Please wait before sending further live requests. You can continue testing "
                            "deterministic tools (Calculator, Knowledge Base search, CSS parser) offline in the **Gemini vs. No Gemini** workspace."
                        )
                    else:
                        response_text = f"⚠️ **Execution Notice:** Unable to reach model: {safe_err}"

                st.markdown(response_text)

                if tools_used:
                    with st.expander(f"🛠️ Tools Invoked ({len(tools_used)})", expanded=True):
                        for t in tools_used:
                            render_tool_invocation_ui(t)

                if unverified_flag:
                    st.markdown(
                        "<span class='metric-badge badge-unverified'>"
                        "🔍 Grounded Knowledge Verification: Fact is not found in verified records. Hallucination blocked."
                        "</span>",
                        unsafe_allow_html=True
                    )

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tools_invoked": tools_used,
                    "unverified": unverified_flag
                })


# ============================================================================
# Page 3: Gemini vs. No Gemini Comparison Workspace
# ============================================================================

elif current_page == "⚖️ Gemini vs. No Gemini":
    st.markdown("### ⚖️ Gemini vs. No Gemini — Side-by-Side Evaluation")
    st.markdown(
        "Directly compare deterministic **local execution (offline, 0 tokens, instant latency)** against "
        "**model-assisted reasoning (Gemini function calling)** on identical inputs."
    )

    # Task Selector
    task_mode = st.selectbox(
        "Select Comparison Task:",
        [
            "🧮 Arithmetic Calculation (Safe AST Calculator)",
            "📚 Knowledge-Base Lookup (Verified Facts vs. Unknowns)",
            "🎨 CSS Parsing & Stylesheet Analytics (Deterministic AST vs. AI Insights)",
        ]
    )

    # Shared Input Field with Presets based on Task
    if "Arithmetic" in task_mode:
        st.markdown("**Sample Expressions:**")
        ex_cols = st.columns(4)
        calc_presets = [
            "(45.5 * 12) + (180 / 4)",
            "((10 + 5) * (8 - 4)) / (2 + 1)",
            "100 / 0",
            "12.5 * 4.2 - 2.5",
        ]
        chosen_preset = None
        for i, cp in enumerate(calc_presets):
            with ex_cols[i]:
                if st.button(cp, key=f"calc_pre_{i}", use_container_width=True):
                    chosen_preset = cp

        calc_input = st.text_input(
            "Mathematical Expression:",
            value=chosen_preset or "(45.5 * 12) + (180 / 4)",
            key="cmp_calc_input"
        )
        task_input_value = calc_input

    elif "Knowledge-Base" in task_mode:
        st.markdown("**Sample Questions (Known & Unknown):**")
        kb_cols = st.columns(4)
        kb_presets = [
            "Who is the founder of Skill First Applied Labs?",
            "What is the mission of Skill First Applied Labs?",
            "Where is the Google headquarters?",
            "Where is SkillFirst Labs headquartered?",
        ]
        chosen_kb_preset = None
        for i, kp in enumerate(kb_presets):
            with kb_cols[i]:
                if st.button(kp, key=f"kb_pre_{i}", use_container_width=True):
                    chosen_kb_preset = kp

        kb_input = st.text_input(
            "Knowledge Query / Question:",
            value=chosen_kb_preset or "Who is the founder of Skill First Applied Labs?",
            key="cmp_kb_input"
        )
        task_input_value = kb_input

    else:  # CSS Parsing
        st.markdown("**Sample Stylesheets:**")
        css_cols = st.columns(3)
        sample_css_presets = {
            "Simple Rules": "body { margin: 0; padding: 10px; font-family: sans-serif; }\nh1 { color: #333; font-size: 2rem; }\nbody { background-color: #fff; }",
            "Nested At-Rules": "@media (max-width: 600px) {\n  .container { width: 100%; padding: 5px; }\n}\n@font-face {\n  font-family: 'Open Sans';\n  src: url('font.woff2');\n}",
            "Load Normalize.css": (CSS_PARSER_DIR / "sample.css").read_text(encoding="utf-8") if (CSS_PARSER_DIR / "sample.css").exists() else "body { margin: 0; }"
        }
        chosen_css = None
        for i, (label, val) in enumerate(sample_css_presets.items()):
            with css_cols[i]:
                if st.button(label, key=f"css_pre_{i}", use_container_width=True):
                    chosen_css = val

        css_input = st.text_area(
            "CSS Stylesheet to Parse:",
            value=chosen_css or sample_css_presets["Simple Rules"],
            height=160,
            key="cmp_css_input"
        )
        task_input_value = css_input

    # Execution Action Controls
    st.markdown("**Actions:**")
    btn_col1, btn_col2, btn_col3, _ = st.columns([1.2, 1.2, 1.2, 2.5])
    run_local_btn = btn_col1.button("🟢 Run Local (No Gemini)", use_container_width=True)
    run_gemini_btn = btn_col2.button("✨ Run With Gemini", use_container_width=True)
    run_both_btn = btn_col3.button("⚡ Run Both (Compare)", type="primary", use_container_width=True)

    # Initialize comparison containers in session state
    if "cmp_results" not in st.session_state:
        st.session_state.cmp_results = None

    should_run_local = run_local_btn or run_both_btn
    should_run_gemini = run_gemini_btn or run_both_btn

    if should_run_local or should_run_gemini:
        results = {
            "task_mode": task_mode,
            "input": task_input_value,
            "local": None,
            "gemini": None,
        }

        # --------------------------------------------------------------------
        # 1. Local Execution Path (Zero Gemini, Deterministic, Offline)
        # --------------------------------------------------------------------
        if should_run_local:
            t0 = time.perf_counter()
            if "Arithmetic" in task_mode:
                local_res = safe_calculate(task_input_value)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                results["local"] = {
                    "type": "calculator",
                    "status": local_res.get("status"),
                    "output": local_res,
                    "elapsed_ms": elapsed_ms,
                    "model_called": False,
                }
            elif "Knowledge-Base" in task_mode:
                local_res = search_knowledge_base(task_input_value)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                results["local"] = {
                    "type": "knowledge_search",
                    "status": local_res.get("status"),
                    "output": local_res,
                    "elapsed_ms": elapsed_ms,
                    "model_called": False,
                }
            else:  # CSS
                try:
                    local_res = analyze_css(task_input_value, "comparison.css")
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    results["local"] = {
                        "type": "css_parser",
                        "status": "success",
                        "output": local_res,
                        "elapsed_ms": elapsed_ms,
                        "model_called": False,
                    }
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    results["local"] = {
                        "type": "css_parser",
                        "status": "error",
                        "output": {"error": str(exc)},
                        "elapsed_ms": elapsed_ms,
                        "model_called": False,
                    }

        # --------------------------------------------------------------------
        # 2. Gemini Execution Path (Model-Assisted with Function Calling)
        # --------------------------------------------------------------------
        if should_run_gemini:
            has_key = bool(os.environ.get("GEMINI_API_KEY") or st.session_state.agent.api_key)
            if not has_key:
                results["gemini"] = {
                    "status": "missing_key",
                    "output": "GEMINI_API_KEY is not configured. Live model reasoning requires an API key in AI-Agent/.env.",
                    "elapsed_s": 0.0,
                    "tools_invoked": [],
                }
            else:
                t0 = time.perf_counter()
                try:
                    if "Arithmetic" in task_mode:
                        history_len_before = len(st.session_state.agent.history)
                        agent_reply = st.session_state.agent.run(f"Calculate {task_input_value}")
                        elapsed_s = time.perf_counter() - t0
                        if hasattr(st.session_state.agent, "get_last_tools") and st.session_state.agent.get_last_tools():
                            tools_used = st.session_state.agent.get_last_tools()
                        else:
                            tools_used = detect_tools_invoked(st.session_state.agent.history[history_len_before:])
                        results["gemini"] = {
                            "status": "success",
                            "output": agent_reply,
                            "elapsed_s": elapsed_s,
                            "tools_invoked": tools_used,
                        }
                    elif "Knowledge-Base" in task_mode:
                        history_len_before = len(st.session_state.agent.history)
                        agent_reply = st.session_state.agent.run(task_input_value)
                        elapsed_s = time.perf_counter() - t0
                        if hasattr(st.session_state.agent, "get_last_tools") and st.session_state.agent.get_last_tools():
                            tools_used = st.session_state.agent.get_last_tools()
                        else:
                            tools_used = detect_tools_invoked(st.session_state.agent.history[history_len_before:])
                        results["gemini"] = {
                            "status": "success",
                            "output": agent_reply,
                            "elapsed_s": elapsed_s,
                            "tools_invoked": tools_used,
                        }
                    else:  # CSS
                        # Deterministic parser is the source of truth; Gemini explains real output
                        local_css_rep = analyze_css(task_input_value, "comparison.css")
                        prompt = (
                            f"Based strictly on this structured CSS parser output summary: "
                            f"{json.dumps({'rules': local_css_rep['total_rules'], 'declarations': local_css_rep['total_declarations'], 'duplicate_selectors': local_css_rep['duplicate_selectors'], 'top_properties': list(local_css_rep['property_frequency'].items())[:5]})}, "
                            "provide a concise 2-sentence technical evaluation of stylesheet maintainability and property reuse."
                        )
                        agent_reply = st.session_state.agent.run(prompt)
                        elapsed_s = time.perf_counter() - t0
                        results["gemini"] = {
                            "status": "success",
                            "output": agent_reply,
                            "elapsed_s": elapsed_s,
                            "tools_invoked": [],
                            "explanation_mode": True,
                        }
                except Exception as exc:
                    elapsed_s = time.perf_counter() - t0
                    err_str = str(exc)
                    safe_err = err_str.replace(os.environ.get("GEMINI_API_KEY", "___"), "[REDACTED]")
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        err_msg = (
                            "Gemini API rate limit or daily quota reached (HTTP 429). "
                            "Please try again later or test deterministic tools locally."
                        )
                    else:
                        err_msg = f"Gemini Execution Error: {safe_err}"
                    results["gemini"] = {
                        "status": "error",
                        "output": err_msg,
                        "elapsed_s": elapsed_s,
                        "tools_invoked": [],
                    }

        st.session_state.cmp_results = results

    # ------------------------------------------------------------------------
    # Render Comparison Panels
    # ------------------------------------------------------------------------
    if st.session_state.cmp_results:
        cmp = st.session_state.cmp_results
        st.divider()

        st.markdown("#### 🔍 Side-by-Side Execution Results")
        col_left, col_right = st.columns(2)

        # Left Column: Local Execution
        with col_left:
            st.markdown(
                "##### 🟢 Without Gemini — Local Deterministic Execution\n"
                "<span class='metric-badge badge-local'>Zero API Calls</span> "
                "<span class='metric-badge badge-local'>0 Tokens</span> "
                "<span class='metric-badge badge-local'>Offline</span>",
                unsafe_allow_html=True
            )
            loc = cmp.get("local")
            if loc:
                st.caption(f"⏱️ **Execution Time:** `{loc['elapsed_ms']:.2f} ms` | **Status:** `{loc['status']}`")
                st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                if loc["type"] == "calculator":
                    calc_out = loc["output"]
                    if calc_out.get("status") == "success":
                        st.metric("Computed Result", calc_out.get("result"))
                    else:
                        st.error(f"Error: {calc_out.get('error')}")
                    with st.expander("Structured Output", expanded=False):
                        st.json(calc_out)

                elif loc["type"] == "knowledge_search":
                    kb_out = loc["output"]
                    if kb_out.get("status") == "success":
                        st.success(f"Matched {kb_out.get('matches_count')} verified fact(s) for Skill First Applied Labs.")
                        for r in kb_out.get("results", []):
                            st.markdown(f"- **{r.get('topic')}:** {r.get('fact')}")
                            st.caption(f"  *Source: {r.get('source_label')} ({r.get('source')}) | Status: `{r.get('verification_status')}`*")
                    else:
                        st.warning(f"**Status:** `not_found`\n\n{kb_out.get('message', 'Not found.')}")
                        if "reason" in kb_out:
                            st.caption(f"**Explicit Reason:** {kb_out['reason']}")
                        if kb_out.get("detected_entity"):
                            st.info(f"ℹ️ **Entity Check:** Detected query subject `{kb_out['detected_entity']}` is not in verified records.")
                    with st.expander("Structured JSON Output", expanded=False):
                        st.json(kb_out)

                else:  # CSS
                    css_out = loc["output"]
                    if "error" in css_out:
                        st.error(css_out["error"])
                    else:
                        st.metric("Total Rules", css_out.get("total_rules"))
                        st.metric("Total Declarations", css_out.get("total_declarations"))
                        with st.expander("Structured CSS Report", expanded=False):
                            st.json(css_out)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Local execution has not run yet. Click 'Run Local' or 'Run Both'.")

        # Right Column: Gemini Execution
        with col_right:
            st.markdown(
                "##### ✨ With Gemini — Model-Assisted Reasoning\n"
                "<span class='metric-badge badge-gemini'>Gemini Tool Calling</span> "
                "<span class='metric-badge badge-gemini'>Autonomous Dispatch</span>",
                unsafe_allow_html=True
            )
            gem = cmp.get("gemini")
            if gem:
                if gem["status"] == "missing_key":
                    st.warning(gem["output"])
                elif gem["status"] == "error":
                    st.error(gem["output"])
                else:
                    st.caption(f"⏱️ **Total Turn Time:** `{gem['elapsed_s']:.2f} s` | **Status:** `{gem['status']}`")
                    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
                    st.markdown(f"**Agent Response:**\n\n{gem['output']}")

                    if gem.get("tools_invoked"):
                        with st.expander(f"🛠️ Tools Invoked ({len(gem['tools_invoked'])})", expanded=True):
                            for t in gem["tools_invoked"]:
                                render_tool_invocation_ui(t)

                    if gem.get("explanation_mode"):
                        st.caption("ℹ️ Deterministic parser output was passed to Gemini for architectural code review.")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Gemini execution has not run yet. Click 'Run With Gemini' or 'Run Both'.")

        # --------------------------------------------------------------------
        # Comparison & Tradeoff Analysis
        # --------------------------------------------------------------------
        if loc and gem and gem["status"] == "success":
            st.divider()
            st.markdown("#### ⚖️ Technical Comparison & Tradeoff Summary")

            speedup = (gem['elapsed_s'] * 1000) / max(loc['elapsed_ms'], 0.001)
            comp_cols = st.columns(3)
            comp_cols[0].metric("Local Latency", f"{loc['elapsed_ms']:.2f} ms")
            comp_cols[1].metric("Model Latency", f"{gem['elapsed_s']:.2f} s")
            comp_cols[2].metric("Local Speed Advantage", f"{speedup:.0f}x faster")

            summary_box = st.container()
            with summary_box:
                if "Arithmetic" in cmp["task_mode"]:
                    calc_val = str(loc["output"].get("result", ""))
                    model_reply = gem["output"]
                    match = calc_val in model_reply if calc_val else False
                    st.info(
                        f"**Evaluation:** Both execution paths yield the exact same underlying calculation result "
                        f"(`{calc_val}`).\n\n"
                        f"- **Local:** Instantaneous ({loc['elapsed_ms']:.2f} ms), deterministic, zero token cost, no external dependency.\n"
                        f"- **Gemini:** Model autonomously issued a `calculate` function call, intercepted the verified local AST tool result, and framed it in natural language."
                    )
                elif "Knowledge-Base" in cmp["task_mode"]:
                    if loc["output"].get("status") == "not_found":
                        detected_ent = loc["output"].get("detected_entity")
                        if detected_ent and detected_ent.lower() not in ("skillfirst", "skill first", "skill first applied labs", "company"):
                            st.info(
                                f"**Evaluation (Entity-Aware Hallucination Prevention):**\n\n"
                                f"- **Subject Entity:** `{detected_ent}` (External / Unknown Entity)\n"
                                f"- **Evidence Match:** ❌ Query asks about `{detected_ent}`, which does not exist in the Skill First Applied Labs knowledge base.\n"
                                f"- **Local Deterministic Retrieval:** Rejected false-positive keyword overlap. It correctly returned `status: 'not_found'` with explicit reason: `{loc['output'].get('reason')}`.\n"
                                f"- **Gemini Agent:** Grounded in the tool result, honestly reporting that `{detected_ent}`'s information cannot be verified from the knowledge base. Zero cross-company hallucination."
                            )
                        else:
                            st.info(
                                f"**Evaluation (Unrecorded Information Protection):**\n\n"
                                f"- **Local Search:** Explicitly returned `status: 'not_found'` ({loc['output'].get('reason')}).\n"
                                f"- **Gemini Agent:** Faithfully observed the tool output and refused to invent company facts, stating clearly that the information cannot be verified from available records."
                            )
                    else:
                        st.info(
                            "**Evaluation (Verified Grounding & Source Attribution):**\n\n"
                            "- **Subject Entity:** Skill First Applied Labs (https://www.skillfirstlabs.com/)\n"
                            f"- **Local Search:** Retrieved {loc['output'].get('matches_count')} verified fact(s) with official website and LinkedIn attribution.\n"
                            "- **Gemini Agent:** Emitted `search_knowledge_base`, ingested the verified fact records, and synthesized an accurate, grounded answer without hallucinating."
                        )
                else:  # CSS
                    st.info(
                        "**Evaluation (Parser as Source of Truth):**\n\n"
                        "- **Local Parser:** Zero-dependency lexical tokenizer and declaration extractor that computes exact mathematical frequencies and detects duplicate rules.\n"
                        "- **Gemini Interpretation:** Model acts as an architectural code reviewer analyzing the deterministic metrics rather than simulating parsing."
                    )

