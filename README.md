# Insight Lab — AI Engineering Workbench
**SkillFirst Labs — AI/ML Multi-Milestone Assessment Repository**

A unified engineering project and interactive web workbench combining:
1. **Milestone 1: CSS Insight Parser** — Pure Python lexical analysis, AST extraction, and stylesheet analytics.
2. **Milestone 2: Tool-Using AI Agent** — Autonomous Google Gemini function calling with AST calculator, grounded knowledge retrieval, and hallucination prevention.
3. **Gemini vs. No Gemini Comparison** — Side-by-side empirical evaluation of deterministic offline execution vs. model-assisted reasoning.

---

## Repository Structure

```text
AIML-Task/
│
├── app.py                     # Root Unified Workbench Dashboard (Streamlit)
├── requirements.txt           # Unified dependency specifications
├── test_unified_workbench.py  # End-to-end integration & smoke tests
├── README.md                  # Unified workbench documentation (this file)
│
├── CSS-Insight-Parser/        # Milestone 1: CSS Parser & Analytics
│   ├── parser.py              # Pure Python CSS tokenizer & analytics engine
│   ├── sample.css             # Normalize.css v8.0.1 benchmark stylesheet
│   ├── parsed_output.json     # Structured JSON output from parser
│   ├── test_parser.py         # Automated test suite (23 unit tests)
│   └── README.md              # Milestone 1 documentation
│
└── AI-Agent/                  # Milestone 2: Tool-Using AI Agent
    ├── agent.py               # Autonomous agent loop, AST calculator, knowledge base
    ├── knowledge.json         # Grounded knowledge base (verified facts)
    ├── test_agent.py          # Automated offline test suite (32 unit tests)
    ├── .env.example           # Template for GEMINI_API_KEY configuration
    ├── .gitignore             # Exclusion rules for secrets & virtualenvs
    ├── requirements.txt       # AI agent package dependencies
    └── README.md              # Milestone 2 documentation
```

---

## Quick Start Guide

### 1. Open Project in VS Code
Open `D:\projects\AIML-Task` in Visual Studio Code.

### 2. Activate Virtual Environment in Windows PowerShell

```powershell
# Set execution policy for the current session (if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the existing virtual environment in AI-Agent
.\AI-Agent\.venv\Scripts\Activate.ps1
```

*(Alternatively, create a new virtual environment at root with `python -m venv venv` and activate it).*

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

Installed packages:
- `streamlit>=1.35.0`: Lightweight interactive web UI.
- `google-genai>=0.1.1`: Official Google GenAI SDK.
- `python-dotenv>=1.0.0`: Environment configuration loader.

### 4. (Optional) Configure Gemini API Key
To enable live AI agent responses in Page 2 and Page 3:
1. Copy `AI-Agent\.env.example` to `AI-Agent\.env`:
   ```powershell
   Copy-Item AI-Agent\.env.example AI-Agent\.env
   ```
2. Insert your free API key from [Google AI Studio](https://aistudio.google.com/):
   ```ini
   GEMINI_API_KEY=AIzaSy...YourKeyHere
   GEMINI_MODEL=gemini-2.5-flash
   MAX_TOOL_TURNS=5
   ```
> **Note:** The entire dashboard, local CSS parser, local AST calculator, and local knowledge search work **100% offline without an API key**.

### 5. Launch the Web Application

Run the application from the project root:

```powershell
streamlit run app.py
```

The web interface will automatically open in your browser at `http://localhost:8501`.

---

## Unified Dashboard Features

### Page 1 — CSS Insight Parser (Milestone 1)
- **Editor & Presets**: Enter custom CSS or load `sample.css` (Normalize.css v8.0.1) with one click.
- **Analytical Metrics**: Live cards showing Total Rules, Declarations, Unique Properties, Duplicate Selectors, and parse time in milliseconds.
- **Extracted Rules Inspector**: Expandable view of each CSS selector and its declaration dictionary.
- **Property Frequencies**: Structured table ranking CSS property occurrences.
- **Export**: One-click download of parsed output as `parsed_css_output.json`.
- **Fault-Tolerant**: Captures unclosed comments, braces, or strings gracefully without crashing.

### Page 2 — Tool-Using AI Agent (Milestone 2)
- **Interactive Chat**: Natural language interface communicating with Google Gemini.
- **Autonomous Tool Dispatching**:
  - `calculate`: Safe AST-based mathematical evaluation (no unrestricted `eval()`).
  - `search_knowledge_base`: Grounded keyword retrieval from `knowledge.json`.
- **Tool Inspection**: Expandable cards reveal exact tool names, arguments, and return values.
- **Hallucination Prevention**: Explicit unverified warning when a requested fact is absent from verified records.
- **Suggested Prompts**:
  - `What is (45.5 * 12) + (180 / 4)?`
  - `Who are the founders of SkillFirst Labs?`
  - `What are the internship milestones?`
  - `What is the current stock price of SkillFirst Labs?`
- **Session Control**: "Reset Chat History" button clears conversation context cleanly.

### Page 3 — Gemini vs. No Gemini (Side-by-Side Evaluation)
A dedicated comparison workspace evaluating deterministic local execution against model-assisted reasoning:
- **Task Selector**:
  - `🧮 Arithmetic Calculation`
  - `📚 Knowledge-Base Lookup`
  - `🎨 CSS Parsing & Stylesheet Analytics`
- **Actions**:
  - `🟢 Run Local (No Gemini)`: Runs local functions directly (`safe_calculate`, `search_knowledge_base`, `analyze_css`). Instantaneous, 0 token overhead, 100% offline.
  - `✨ Run With Gemini`: Runs real Gemini agent loop with tool-calling dispatch.
  - `⚡ Run Both (Compare)`: Executes both paths on identical inputs.
- **Side-by-Side Results**:
  - Displays execution time (ms vs. seconds), status, and structured outputs.
  - For CSS, preserves the deterministic parser as the source of truth, with Gemini acting as an architectural code reviewer.
- **Comparison Summary**: Automatically assesses output agreement, latency differential, and hallucination prevention.

---

## Automated Test Verification

The repository includes three independent automated test suites totaling **62 tests** (all passing offline):

### 1. Unified Workbench Integration Tests (7 tests)
```powershell
python -m unittest test_unified_workbench.py -v
```
*Validates cross-module imports, CSS parser integration, safe calculator, knowledge-base search, and app syntax.*

### 2. Milestone 1: CSS Insight Parser Tests (23 tests)
```powershell
cd CSS-Insight-Parser
python -m unittest test_parser.py -v
cd ..
```
*Validates comment removal, string handling, at-rules, nested blocks, and error handling.*

### 3. Milestone 2: AI Agent Tests (32 tests)
```powershell
cd AI-Agent
python -m unittest test_agent.py -v
cd ..
```
*Validates AST calculator, division by zero, code injection rejection, knowledge search, missing keys, and multi-turn loops.*

---

## CLI Execution (Without UI)

Both milestones retain full command-line capability:

```powershell
# Milestone 1: Run CSS parser CLI
cd CSS-Insight-Parser
python parser.py
cd ..

# Milestone 2: Run local tool verification CLI
cd AI-Agent
python agent.py --calc "(45.5 * 12) + (180 / 4)"
python agent.py --search "founders"
cd ..
```

---

## Security & Secrets Policy
- API keys are strictly loaded from `AI-Agent/.env` or system environment variables.
- `.env` is explicitly ignored by Git in `.gitignore` and never committed or tracked.
- The UI sanitizes all exception messages to ensure no credentials or raw API keys are displayed.

### 3. Milestone 2: AI Agent Tests (32 tests)
```powershell
cd AI-Agent
python -m unittest test_agent.py -v
cd ..
```
*Validates AST calculator, division by zero, code injection rejection, knowledge search, missing keys, and multi-turn loops.*

---

## CLI Execution (Without UI)

Both milestones retain full command-line capability:

```powershell
# Milestone 1: Run CSS parser CLI
cd CSS-Insight-Parser
python parser.py
cd ..

# Milestone 2: Run local tool verification CLI
cd AI-Agent
python agent.py --calc "(45.5 * 12) + (180 / 4)"
python agent.py --search "founders"
cd ..
```

---

## Security & Secrets Policy
- API keys are strictly loaded from `AI-Agent/.env` or system environment variables.
- `.env` is explicitly ignored by Git in `.gitignore` and never committed or tracked.
- The UI sanitizes all exception messages to ensure no credentials or raw API keys are displayed.

