# Milestone 2: Tool-Using AI Agent
**SkillFirst Labs — AI/ML Engineering Internship Assessment**

A production-grade, tool-augmented AI agent implemented in Python. The agent leverages the official Google Gemini tool-calling (function calling) protocol to dynamically reason, perform safe arithmetic calculations without arbitrary code execution, and retrieve verified company information from a grounded local knowledge base.

---

## Table of Contents
1. [Architecture & Overview](#architecture--overview)
2. [Tools & Safeguards](#tools--safeguards)
   - [Tool 1: Safe AST Calculator](#tool-1-safe-ast-calculator)
   - [Tool 2: Grounded Knowledge-Base Search](#tool-2-grounded-knowledge-base-search)
3. [Agent Execution Loop](#agent-execution-loop)
4. [Hallucination Prevention & Repeatability](#hallucination-prevention--repeatability)
5. [Beginner-Friendly Setup Guide](#beginner-friendly-setup-guide)
   - [Step 1: Open the Project in VS Code](#step-1-open-the-project-in-vs-code)
   - [Step 2: Create & Activate Virtual Environment](#step-2-create--activate-virtual-environment)
   - [Step 3: Install Dependencies](#step-3-install-dependencies)
   - [Step 4: Configure Gemini API Key](#step-4-configure-gemini-api-key)
   - [Step 5: Run Automated Tests](#step-5-run-automated-tests)
   - [Step 6: Run the Agent](#step-6-run-the-agent)
6. [Offline Verification (Without API Key)](#offline-verification-without-api-key)
7. [Sample Queries & Expected Behaviors](#sample-queries--expected-behaviors)
8. [Automated Test Suite](#automated-test-suite)
9. [Project File Structure](#project-file-structure)

---

## Architecture & Overview

The agent operates on an autonomous reasoning and tool-dispatch loop:

```text
 User Question
      │
      ▼
┌──────────────┐
│  AI Agent    │◄─────────────────────────────┐
│ (agent.py)   │                              │
└──────┬───────┘                              │
       │ Send message + Tool Schemas          │
       ▼                                      │
┌──────────────┐                              │
│  Gemini LLM  │                              │
└──────┬───────┘                              │
       │                                      │
       ├─► [Text Answer] ─────────────► Final Output to User
       │                                      │
       └─► [Function Call Request]            │
                 │                            │
                 ▼                            │
      ┌─────────────────────┐                 │
      │   Tool Dispatcher   │                 │
      └──────┬──────────────┘                 │
             │                                │
      ┌──────┴────────────────┐               │
      ▼                       ▼               │
┌──────────────┐      ┌──────────────┐        │
│  Safe AST    │      │  Knowledge   │        │
│  Calculator  │      │  Base Search │        │
└──────┬───────┘      └──────┬───────┘        │
       │                     │                │
       └──────────┬──────────┘                │
                  ▼                           │
       Return Structured Result ──────────────┘
       (Next Conversation Turn)
```

1. **User Prompt**: The user inputs a natural language question.
2. **Model Reasoning & Tool Invocation**: Gemini inspects the query against the tool declarations and emits a structured `functionCall` request (`calculate` or `search_knowledge_base`).
3. **Local Tool Execution**: The Python program intercepts the function call, executes the verified local Python function with provided parameters, and captures the result.
4. **Context Injection**: The tool's output is passed back to Gemini in the subsequent conversation turn (`functionResponse`).
5. **Final Answer or Iteration**: The model either synthesizes the verified answer or requests further tool executions (up to a configured round limit).

---

## Tools & Safeguards

### Tool 1: Safe AST Calculator
- **Function**: `safe_calculate(expression: str) -> Dict[str, Any]`
- **Capabilities**:
  - Supports addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), modulo (`%`), exponents (`**`), unary operators (`+`, `-`), parentheses (`()`), and floating-point decimals.
  - Automatically formats integer-equivalent floats (e.g. `15.0` -> `15`) and limits decimal float precision to 8 decimal places to eliminate IEEE 754 precision artifacts.
- **Safety & Security (No unrestricted `eval()`)**:
  - Uses Python's standard library `ast.parse(expression, mode='eval')` to parse expressions into an Abstract Syntax Tree.
  - Strictly traverses the AST and permits only numeric constants (`ast.Constant`), unary operations (`ast.UnaryOp`), and safe binary operations (`ast.BinOp`).
  - **Blocks all malicious payloads**: Any presence of `ast.Call`, `ast.Name`, `ast.Attribute`, `ast.Import`, list comprehensions, or string operations triggers a security error and prevents execution.
  - **Division by Zero Protection**: Explicitly checks for zero denominators in division and modulo, returning `"Division by zero is undefined."` rather than crashing.
  - **DoS & Overflow Guardrails**: Restricts exponent magnitudes (`abs(exponent) <= 1000`) to prevent memory exhaustion or denial-of-service.

### Tool 2: Grounded Knowledge-Base Search
- **Function**: `search_knowledge_base(query: str) -> Dict[str, Any]`
- **Capabilities**:
  - Reads verified facts from `knowledge.json`.
  - Indexes company background, founders, internship milestones, evaluation criteria, products, security policies, and FAQs into discrete semantic fact records.
  - Uses keyword extraction, domain-term isolation, and word-boundary matching.
  - Calculates relevance scores and token coverage ratios to prioritize high-confidence matches.
- **Hallucination Prevention**:
  - Clearly reports when requested information is absent:
    ```json
    {
      "status": "not_found",
      "query": "stock price on NASDAQ",
      "message": "No verified facts found in the knowledge base matching 'stock price on NASDAQ'. The requested information is not available in verified records."
    }
    ```
  - Isolates ubiquitous brand stop-words (`"skillfirst"`, `"labs"`, `"company"`) so that questions with ungrounded terms (e.g. `"What is the stock price of SkillFirst Labs?"`) do not trigger false positive matches.
  - System instructions strictly mandate that the agent must never invent facts, guess missing metrics, or present unverified assumptions as truth.

---

## Agent Execution Loop

The agent's multi-turn loop is implemented in `AIAgent.run()`:

1. **State Tracking**: `self.history` maintains the full conversation trajectory according to the official Gemini multi-turn role structure (`user`, `model`, `function`).
2. **Turn Limit Guardrail**: To prevent infinite loops caused by model recursion or degenerate prompts, the loop is strictly bounded by `max_turns` (default: 5). If the threshold is reached, execution gracefully halts with an informative safety message.
3. **Exception Safety**: All tool executions are wrapped in guarded try-except blocks. If an invalid argument or computation error occurs, the error message is packaged and returned to the model as a valid tool response, enabling the model to self-correct.

---

## Hallucination Prevention & Repeatability

### 1. Missing Knowledge Handling
When a user asks about an unknown topic (e.g., stock price, external events, unrecorded company policies), `search_knowledge_base` returns `status: "not_found"`. The agent's system prompt strictly instructs it to state clearly that the requested information cannot be verified from available records, rather than improvising.

### 2. Deterministic Tool Execution
For identical inputs, local tools (`safe_calculate` and `search_knowledge_base`) return identical results every time:
- The calculator evaluates pure mathematical operations without side effects.
- The knowledge-base search scans `knowledge.json` using deterministic tokenization and relevance scoring.
- Automated tests verify this by running 50 repeated iterations of identical queries and asserting identical outputs.

### 3. LLM Non-Determinism vs. Deterministic Tools
While tool outputs are 100% deterministic, LLM responses may vary slightly in phrasing across runs due to probabilistic token generation (temperature). Grounding the agent with deterministic tools ensures that all numerical figures and company facts remain exact and verified, even if the surrounding natural-language phrasing fluctuates.

### 4. Tool-Call Round Limits (Infinite Loop Prevention)
Without recursion guards, an agent could continuously re-invoke tools in an infinite loop. The `max_turns` parameter sets a hard ceiling on tool execution rounds per user prompt. Once reached, the loop safely terminates.

### 5. API Key Security & Isolation
- API keys must never be hardcoded in source code or committed to version control.
- All secrets are loaded from `.env` or system environment variables.
- `.gitignore` explicitly excludes `.env`, `*.env`, and virtual environment directories.

---

## Beginner-Friendly Setup Guide

### Step 1: Open the Project in VS Code
1. Open Visual Studio Code.
2. Select **File > Open Folder...** and navigate to:
   ```text
   D:\projects\AIML-Task\AI-Agent
   ```
3. Open an integrated terminal: **Terminal > New Terminal** (or press `` Ctrl + ` ``).

---

### Step 2: Create & Activate Virtual Environment

In Windows PowerShell:

```powershell
# 1. Create a virtual environment named 'venv'
python -m venv venv

# 2. Allow script execution in current PowerShell session (if restricted)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

*(You will see `(venv)` appear at the beginning of your PowerShell prompt).*

> **Tip for Command Prompt (cmd.exe):** Run `venv\Scripts\activate.bat` instead.  
> **Tip for macOS / Linux:** Run `source venv/bin/activate`.

---

### Step 3: Install Dependencies

With the virtual environment activated:

```powershell
pip install -r requirements.txt
```

This installs:
- `google-genai`: The official Google GenAI SDK.
- `python-dotenv`: Environment configuration parser.

---

### Step 4: Configure Gemini API Key

1. Copy the `.env.example` template to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open `.env` in VS Code and replace `your_gemini_api_key_here` with your genuine Gemini API key:
   ```ini
   GEMINI_API_KEY=AIzaSy...YourRealKeyHere
   GEMINI_MODEL=gemini-2.5-flash
   MAX_TOOL_TURNS=5
   ```
   > You can generate a free Gemini API key at [Google AI Studio](https://aistudio.google.com/).

Alternatively, you can set it as a session environment variable in PowerShell:
```powershell
$env:GEMINI_API_KEY="AIzaSy...YourRealKeyHere"
```

---

### Step 5: Run Automated Tests

The automated test suite runs completely offline with zero dependency on a live API key:

```powershell
python -m unittest test_agent.py -v
```

Expected result:
```text
Ran 32 tests in 0.040s

OK
```

---

### Step 6: Run the Agent

#### Interactive Mode (REPL)
Launch the interactive terminal session:
```powershell
python agent.py
```
Commands inside the interactive shell:
- `tools` — View available tools and descriptions.
- `reset` — Reset conversation history.
- `exit` or `quit` — Exit the agent.

#### Single-Query CLI Mode
You can also ask questions directly from the command line:
```powershell
python agent.py "What is (125.5 * 8) + (450 / 5)?"
```

---

## Offline Verification (Without API Key)

The agent provides built-in CLI flags to test and inspect the underlying tools locally, without requiring any API key or internet access:

### Test the Safe Calculator:
```powershell
python agent.py --calc "12.5 * 4 + (10 - 2) / 4"
```
**Output:**
```json
Expression: 12.5 * 4 + (10 - 2) / 4
Result: {
  "status": "success",
  "expression": "12.5 * 4 + (10 - 2) / 4",
  "result": 52
}
```

### Test Division by Zero:
```powershell
python agent.py --calc "100 / 0"
```
**Output:**
```json
Result: {
  "status": "error",
  "expression": "100 / 0",
  "error": "Division by zero is undefined."
}
```

### Test Knowledge-Base Search:
```powershell
python agent.py --search "founder"
```
**Output:**
```json
Query: founder
Result: {
  "status": "success",
  "query": "founder",
  "matches_count": 1,
  "results": [
    {
      "topic": "company.founder",
      "fact": "Founder of Skill First Applied Labs: Gurunathan Murugan (Founder, Skill First Applied Labs).",
      "source": "https://www.skillfirstlabs.com/",
      "source_label": "Official Website",
      "verification_status": "verified_official_website"
    }
  ]
}
```

### Test Missing Information Retrieval & Entity Isolation:
```powershell
python agent.py --search "Where is the Google headquarters?"
```
**Output:**
```json
Query: Where is the Google headquarters?
Result: {
  "status": "not_found",
  "query": "Where is the Google headquarters?",
  "matches_count": 0,
  "results": [],
  "detected_entity": "Google",
  "entity_match": false,
  "message": "No verified facts found in the knowledge base for 'Google'. The requested information is not available in verified records.",
  "reason": "Query specifically asks about 'Google', which is an external or unverified entity not present in the Skill First Applied Labs knowledge base."
}
```

---

## Sample Queries & Expected Behaviors

| Question / Query | Tool Invoked | Expected Agent Behavior |
| :--- | :--- | :--- |
| `"What is (45.5 * 12) + (180 / 4)?"` | `calculate(expression="(45.5 * 12) + (180 / 4)")` | Computes `546 + 45 = 591` and returns the exact number. |
| `"Who is the founder of Skill First Applied Labs?"` | `search_knowledge_base(query="founder")` | Retrieves Gurunathan Murugan (Source: Official Website: `https://www.skillfirstlabs.com/`). |
| `"What is the mission of Skill First Applied Labs?"` | `search_knowledge_base(query="mission")` | Retrieves verified mission (bridge academia-industry gap via talent management platform). |
| `"Where is the Google headquarters?"` | `search_knowledge_base(query="Google headquarters")` | Entity mismatch detected. Returns `not_found`. Agent reports Google is not in verified records and never claims host company's location. |
| `"Where is SkillFirst Labs headquartered?"` | `search_knowledge_base(query="headquarters")` | Retrieves verified Chennai contact address (Official Website) and Tiruvallur headquarters (LinkedIn separately attributed). |
| `"What is the current stock price of SkillFirst Labs?"` | `search_knowledge_base(query="stock price")` | Tool returns `not_found`. Agent reports that stock price information cannot be verified from available records. |
| `"Evaluate 100 / (5 - 5)"` | `calculate(expression="100 / (5 - 5)")` | Tool catches division by zero; agent reports that division by zero is undefined. |

---

## Automated Test Suite

The test suite (`test_agent.py`) contains **48 comprehensive automated unit and regression tests** divided into 6 test classes:

1. **`TestSafeCalculator` (11 tests)**:
   - `test_basic_arithmetic`: Verifies addition, subtraction, multiplication, and division.
   - `test_decimal_calculations`: Validates floating-point precision and formatting.
   - `test_parentheses_and_precedence`: Nested parentheses and operator precedence.
   - `test_unary_operators`: Unary negation (`-`) and explicit positive signs (`+`).
   - `test_exponents_and_modulo`: Safe powers (`**`) and modulo (`%`).
   - `test_division_by_zero`: Direct and nested division by zero handling.
   - `test_invalid_syntax`: Malformed operators, unclosed brackets, empty inputs.
   - `test_security_disallowed_code_execution`: Rejection of `__import__`, `eval()`, variables, strings, and data structures.
   - `test_exponent_magnitude_limit`: Protection against CPU exhaustion via huge exponents.
   - `test_deterministic_calculator`: Verifies 50 repeated executions return identical results.

2. **`TestKnowledgeBaseSearch` (12 tests)**:
   - `test_knowledge_base_loaded`: Validates JSON schema integrity.
   - `test_matching_facts_company_identity`: Verifies Skill First Applied Labs official website and status.
   - `test_matching_facts_founder`: Verifies Gurunathan Murugan as founder; Elena Rostova/Marcus Vance absent.
   - `test_matching_facts_mission`: Verifies bridging academia-industry gap through talent management platform.
   - `test_matching_facts_contact_details`: Verifies Chennai address and hello@keystone-tms.in email.
   - `test_matching_facts_headquarters_linkedin_attribution`: Verifies separate LinkedIn attribution (Tiruvallur, Tamil Nadu) and absence of San Francisco.
   - `test_google_versus_skillfirst_entity_mismatch`: Queries about Google return `not_found` and never claim host company location.
   - `test_unknown_company_headquarters_not_inferred`: Queries about Microsoft, Apple, Netflix return `not_found`.
   - `test_unknown_stock_price_rejected`: Unknown stock price returns `not_found`.
   - `test_unsupported_policies_rejected`: Unsupported policies return `not_found`.
   - `test_matching_facts_internship_milestones`: Assessment Milestones 1 and 2 retrieval.
   - `test_matching_facts_case_insensitivity`: Case-invariant search behavior.

3. **`TestToolExecution` (6 tests)**:
   - `test_execute_calculator_valid` & `test_execute_calculator_error`: Tool dispatcher execution.
   - `test_execute_search_valid` & `test_execute_search_missing`: Search dispatcher execution.
   - `test_execute_unknown_tool`: Rejection of unregistered tool names.
   - `test_execute_tool_invalid_argument_signature`: Parameter validation and error capture.

4. **`TestApiKeyHandling` (2 tests)**:
   - `test_missing_api_key_raises_error`: Validates error when API key is missing for live calls.
   - `test_local_tools_work_without_api_key`: Validates 100% offline functionality of local tools.

5. **`TestAgentExecutionLoopMocked` (6 tests)**:
   - `test_direct_text_response_no_tools`: Single-turn direct answer.
   - `test_single_tool_call_calculation_flow`: Mocked model arithmetic calling loop.
   - `test_single_tool_call_knowledge_search_flow`: Mocked model founder lookup loop.
   - `test_multi_turn_chained_tool_calls`: Chained knowledge retrieval + calculation reasoning.
   - `test_infinite_loop_prevention_guardrail`: Halts upon reaching max turns safety limit.
   - `test_reset_conversation`: Verifies clean history reset.

6. **`TestGeminiAgentRegressionSuite` (9 tests)**:
   - `test_regression_1_arithmetic_question_calculator_selected_and_result`: Verifies calculator dispatch.
   - `test_regression_2_assessment_question_knowledge_search_selected_and_evidence`: Verifies milestone retrieval.
   - `test_regression_3_paraphrased_assessment_question_retrieval`: Paraphrased requirements questions.
   - `test_regression_4_unknown_company_policy_question_no_hallucination`: Unverified policy refusal.
   - `test_regression_5_tool_returns_no_results_with_explicit_reason`: Explicit `not_found` reason.
   - `test_regression_6_multi_turn_tool_request_result_final_answer`: Multi-turn tool chaining.
   - `test_regression_7_tool_and_api_error_handling_no_leaks`: Tool and API error resilience without secret leaks.
   - `test_regression_8_google_headquarters_honest_fallback`: Honest fallback for Google query without false attribution.
   - `test_regression_9_skillfirst_headquarters_verified_attribution`: Verified attribution distinguishing website contact from LinkedIn headquarters.

4. **`TestApiKeyHandling` (2 tests)**:
   - `test_missing_api_key_raises_error`: Validates error message when `GEMINI_API_KEY` is missing.
   - `test_tools_work_without_api_key`: Confirms offline local tool functionality.

5. **`TestAgentExecutionLoopMocked` (6 tests)**:
   - `test_direct_text_response_no_tools`: Single-turn text response flow.
   - `test_single_tool_call_calculation_flow`: Arithmetic tool call and answer cycle.
   - `test_single_tool_call_knowledge_search_flow`: Knowledge search tool call and answer cycle.
   - `test_multi_turn_chained_tool_calls`: Chained search + calculation workflow.
   - `test_infinite_loop_prevention_guardrail`: Halts runaway tool calls at configured limit.
   - `test_reset_conversation`: Conversation history clearing.

---

## Project File Structure

```text
D:\projects\AIML-Task\AI-Agent/
│
├── agent.py               # Core agent loop, AST calculator, knowledge retrieval, CLI
├── knowledge.json         # Grounded knowledge base (company, internships, policies)
├── test_agent.py          # Automated offline unit test suite (32 tests)
├── requirements.txt       # Python package dependencies
├── .env.example           # Template for environment configuration
├── .gitignore             # Git exclusion rules for secrets and virtualenvs
└── README.md              # Comprehensive documentation and setup guide
```

