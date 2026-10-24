# AI & ML Multi-Milestone Task Repository

This repository hosts a multi-milestone engineering project organized into modular components:

- **Milestone 1: CSS Insight Parser (`CSS-Insight-Parser/`)**
  A lightweight, zero-dependency CSS parser and stylesheet analytics engine implemented in pure Python. It sanitizes comments, safely tracks quotes and brace nesting, handles functional notations and at-rules, extracts selectors and declarations into structured JSON, and generates analytical metrics (rule counts, declaration counts, property frequencies, and duplicate selector detection).

- **Milestone 2: Tool-Using AI Agent (`AI-Agent/`)**
  An intelligent, tool-augmented AI agent capable of dynamic reasoning, knowledge-base querying, and autonomous execution. *(Workspace initialized; ready for Milestone 2 implementation).*

---

## Repository Structure

```text
AIML-Task/
│
├── README.md                 # Repository overview & milestone directory
│
├── CSS-Insight-Parser/       # Milestone 1: CSS Parser & Analytics
│   ├── parser.py             # Core parser and analysis pipeline
│   ├── sample.css            # Benchmark stylesheet (Normalize.css v8.0.1)
│   ├── test_parser.py        # Automated test suite (23 unit tests)
│   ├── parsed_output.json    # Structured JSON output from parser
│   └── README.md             # Detailed documentation for Milestone 1
│
└── AI-Agent/                 # Milestone 2: Tool-Using AI Agent
    └── (Ready for implementation)
```

---

## Milestone 1: Quick Start

To run the CSS Insight Parser and generate analytics:

```bash
cd CSS-Insight-Parser
python parser.py
```

To run the automated test suite:

```bash
cd CSS-Insight-Parser
python -m unittest test_parser.py -v
```
