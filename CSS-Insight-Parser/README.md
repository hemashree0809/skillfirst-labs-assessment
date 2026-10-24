# Milestone 1: CSS Insight Parser

A lightweight, robust, and beginner-friendly Python CSS parser designed to inspect stylesheets, extract structured rule information, and generate key stylesheet analytics without external dependencies.

---

## Approach

The parser processes CSS in a multi-stage pipeline designed for safety, clarity, and precision:

1. **Comment Stripping & String Protection (`remove_comments`)**:
   - Scans character-by-character to distinguish CSS comments (`/* ... */`) from string literals (`'...'` and `"..."`).
   - Handles backslash-escaped quotes (`\"`, `\'`) inside strings.
   - Preserves comments inside quoted strings (e.g., `content: "/* not a comment */"`).
   - Validates that comments and strings are properly closed, raising a clear `ValueError` if any remain unterminated.

2. **Block & Rule Traversal (`parse_block` & `find_matching_brace`)**:
   - Traverses the sanitized CSS using brace-depth tracking, correctly ignoring `{` and `}` characters inside strings.
   - Detects standalone statement at-rules (e.g. `@charset`, `@import`, `@namespace`) terminated by semicolons `;` so they do not contaminate the selector of subsequent rules.
   - Identifies block boundaries and extracts the rule header and body.

3. **At-Rules Handling (Statement, Declaration, and Nested)**:
   - **Statement at-rules**: (e.g. `@import url("style.css");`) are recognized by encountering a semicolon at depth 0 and safely skipped without corrupting subsequent rules.
   - **Declaration at-rules**: (e.g. `@font-face`, `@page`) that do not contain nested blocks are parsed directly as declaration sets.
   - **Nested at-rules**: (e.g. `@media`, `@supports`, `@keyframes`) containing child blocks are traversed recursively, safely extracting nested rules.

4. **Declaration Extraction (`split_declarations` & `_parse_declarations`)**:
   - Splits declarations at `;` while tracking quote and parenthesis depth (`depth == 0`).
   - Ignores semicolons inside parentheses (such as `calc()`, `var()`, and `url("data:image/svg+xml;utf8,...")`).
   - Normalizes property names to lowercase and trims surrounding whitespace.

5. **Analytical Reporting (`analyze_css`)**:
   - Calculates total rule count and total declarations.
   - Tallies property frequencies using Python's `collections.Counter` sorted by most frequent.
   - Detects duplicate selectors using whitespace-normalized matching (e.g. matching `b,\nstrong` with `b, strong`).
   - Outputs a clean JSON report to `parsed_output.json`.

---

## Key Design Decisions

- **Pure Python Standard Library**:
  Built entirely with Python built-ins (`json`, `re`, `collections.Counter`, `pathlib.Path`, and `unittest`). No third-party packages or complex dependencies are required, making it lightweight, transparent, and easy to run anywhere.

- **Declarations Stored as a Key-Value Mapping**:
  Declarations for each rule are stored as a dictionary (`{"property": "value"}`). This enables immediate $O(1)$ property lookups and produces intuitive JSON output. When duplicate properties appear within the same rule (e.g., fallback values like `text-decoration: underline; text-decoration: underline dotted;`), the final declaration takes precedence, mirroring standard CSS cascade behavior.

- **Whitespace-Normalized Duplicate Selector Detection**:
  In CSS, selectors like `h1, h2` and `h1,\nh2` represent the identical target. The parser normalizes internal consecutive whitespace (spaces, tabs, newlines) when checking for duplicates, preventing formatting differences from obscuring duplicate rules.

- **Decoupled Analytics and Parsing**:
  `parse_css()` returns raw structured rule dictionaries, while `analyze_css()` compiles high-level metrics. This separation of concerns allows both parsing mechanics and statistical reporting to be tested independently in unit tests without touching disk files.

---

## Notable Edge Case Explored

### Semicolons Inside Parentheses & Data URIs

In CSS, property values can legitimately contain semicolons inside functional notation, most notably in Data URIs and SVG backgrounds:

```css
a {
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'><circle cx='50' cy='50' r='40';/></svg>");
  color: red;
}
```

A naive `.split(";")` would split on the semicolon inside the SVG Data URI, truncating the background declaration and misinterpreting the SVG remainder as an invalid property.

**Solution in `split_declarations`**:
The parser tracks both quote state and parenthesis nesting depth (`depth`). A semicolon is only recognized as a declaration separator when `depth == 0` and outside any active string quote. This guarantees data URIs, `calc(100% - (20px + 5px))`, and `var(--primary, fallback)` remain intact.

---

## Limitations

- **Progressive Enhancement Fallbacks in a Single Rule**:
  Because declarations within a rule are stored in a standard dictionary, declaring the same property multiple times in one rule (e.g., `font-size: 16px; font-size: 1rem;`) retains only the last value.
- **CSS Nesting Module Level 1 (`&` syntax)**:
  Nested at-rules (`@media`, `@supports`) are supported recursively, but modern native CSS nesting inside element selectors (e.g., `.card { & > .title { color: red; } }`) is not expanded.
- **Non-Standard Preprocessor Syntax**:
  Sass/SCSS constructs (such as `//` single-line comments, mixins, or `@extend`) do not conform to standard CSS syntax and are not supported.

---

## Commands to Run

### Run the Parser
Parses `sample.css` and writes structured results to `parsed_output.json`:

```bash
python parser.py
```

### Run the Automated Tests
Executes the comprehensive 23-test suite covering normal CSS, edge cases, error handling, and benchmarks:

```bash
python -m unittest test_parser.py -v
```
