"""Automated test suite for CSS Insight Parser.

Tests cover:
- Normal CSS parsing (rules, selectors, declarations)
- Comments removal (outside vs inside strings)
- Quoted strings (escaped quotes, braces/semicolons inside quotes)
- Semicolons inside parentheses (calc, var, data URIs)
- At-rules (statement at-rules like @import, declaration at-rules like @font-face,
  and nested at-rules like @media and @supports)
- Analytical reporting (total rules, declarations, property frequencies, duplicate selectors)
- Error handling for unclosed syntax
- Integration test on sample.css
"""

import unittest
from pathlib import Path

from parser import (
    analyze_css,
    find_matching_brace,
    parse_css,
    remove_comments,
    split_declarations,
)


class TestCSSParserNormal(unittest.TestCase):
    """Tests for standard, well-formed CSS rules."""

    def test_single_rule(self):
        css = "h1 { color: red; font-size: 2em; }"
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["selector"], "h1")
        self.assertEqual(
            rules[0]["declarations"],
            {"color": "red", "font-size": "2em"}
        )

    def test_multiple_rules(self):
        css = """
        body { margin: 0; padding: 0; }
        p { line-height: 1.5; color: #333; }
        """
        rules = parse_css(css)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["selector"], "body")
        self.assertEqual(rules[1]["selector"], "p")

    def test_rule_without_trailing_semicolon(self):
        css = "div { display: flex; justify-content: center }"
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(
            rules[0]["declarations"],
            {"display": "flex", "justify-content": "center"}
        )

    def test_grouped_selectors(self):
        css = "h1, h2, h3 { font-weight: bold; }"
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["selector"], "h1, h2, h3")
        self.assertEqual(rules[0]["declarations"], {"font-weight": "bold"})

    def test_empty_rule_block(self):
        css = "button {}"
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["selector"], "button")
        self.assertEqual(rules[0]["declarations"], {})


class TestCSSParserCommentsAndStrings(unittest.TestCase):
    """Tests for comments, quoted strings, and escapes."""

    def test_remove_comments_standard(self):
        css = "/* Main heading */ h1 /* inline */ { color: /* red */ red; }"
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["selector"], "h1")
        self.assertEqual(rules[0]["declarations"], {"color": "red"})

    def test_comment_inside_quotes_preserved(self):
        css = 'a::after { content: "/* not a comment */"; }'
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(
            rules[0]["declarations"]["content"],
            '"/* not a comment */"'
        )

    def test_escaped_quotes_inside_strings(self):
        css = r'span { content: "quote: \"hello\" world"; }'
        rules = parse_css(css)
        self.assertEqual(
            rules[0]["declarations"]["content"],
            r'"quote: \"hello\" world"'
        )

    def test_braces_inside_quoted_string(self):
        css = 'p::before { content: "brace { here }"; color: green; }'
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["declarations"]["content"], '"brace { here }"')
        self.assertEqual(rules[0]["declarations"]["color"], "green")

    def test_semicolon_inside_quoted_string(self):
        css = 'div::after { content: "first; second"; color: red; }'
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["declarations"]["content"], '"first; second"')
        self.assertEqual(rules[0]["declarations"]["color"], "red")


class TestCSSParserParenthesesAndSemicolons(unittest.TestCase):
    """Tests for semicolons inside parentheses (functions, data URIs)."""

    def test_semicolon_in_data_uri(self):
        css = 'a { background: url("data:image/svg+xml;utf8,<svg></svg>"); color: red; }'
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(
            rules[0]["declarations"]["background"],
            'url("data:image/svg+xml;utf8,<svg></svg>")'
        )
        self.assertEqual(rules[0]["declarations"]["color"], "red")

    def test_calc_expression(self):
        css = ".box { width: calc(100% - (20px + 5px)); height: 50px; }"
        rules = parse_css(css)
        self.assertEqual(
            rules[0]["declarations"]["width"],
            "calc(100% - (20px + 5px))"
        )
        self.assertEqual(rules[0]["declarations"]["height"], "50px")

    def test_var_fallback_expression(self):
        css = ":root { --accent: var(--user-color, #0070f3); }"
        rules = parse_css(css)
        self.assertEqual(
            rules[0]["declarations"]["--accent"],
            "var(--user-color, #0070f3)"
        )


class TestCSSParserAtRules(unittest.TestCase):
    """Tests for statement, declaration, and nested at-rules."""

    def test_statement_at_rules_ignored_safely(self):
        css = """
        @charset "utf-8";
        @import url("theme.css");
        @namespace svg "http://www.w3.org/2000/svg";

        body { margin: 0; }
        """
        rules = parse_css(css)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["selector"], "body")
        self.assertEqual(rules[0]["declarations"], {"margin": "0"})

    def test_declaration_at_rules(self):
        css = """
        @font-face {
            font-family: 'Open Sans';
            src: url('opensans.woff2') format('woff2');
        }
        @page {
            margin: 1cm;
        }
        """
        rules = parse_css(css)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["selector"], "@font-face")
        self.assertEqual(rules[0]["declarations"]["font-family"], "'Open Sans'")
        self.assertEqual(rules[1]["selector"], "@page")
        self.assertEqual(rules[1]["declarations"], {"margin": "1cm"})

    def test_nested_at_rules(self):
        css = """
        @media screen and (min-width: 768px) {
            @supports (display: grid) {
                .gallery {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                }
            }
            .sidebar {
                width: 250px;
            }
        }
        """
        rules = parse_css(css)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["selector"], ".gallery")
        self.assertEqual(rules[0]["declarations"]["display"], "grid")
        self.assertEqual(rules[1]["selector"], ".sidebar")
        self.assertEqual(rules[1]["declarations"]["width"], "250px")


class TestCSSParserReporting(unittest.TestCase):
    """Tests for analytics: counts, frequencies, duplicate selectors."""

    def test_analyze_css_metrics(self):
        css = """
        h1 { color: red; font-size: 24px; }
        p { color: blue; font-size: 16px; margin: 0; }
        h2 { color: red; }
        """
        report = analyze_css(css, file_name="inline_test")
        self.assertEqual(report["total_rules"], 3)
        self.assertEqual(report["total_declarations"], 6)
        self.assertEqual(report["property_frequency"]["color"], 3)
        self.assertEqual(report["property_frequency"]["font-size"], 2)
        self.assertEqual(report["property_frequency"]["margin"], 1)
        self.assertEqual(report["duplicate_selectors"], [])

    def test_duplicate_selectors_detected(self):
        css = """
        h1 { color: red; }
        p { margin: 0; }
        h1 { font-size: 24px; }
        h1 { font-weight: bold; }
        """
        report = analyze_css(css)
        self.assertEqual(report["total_rules"], 4)
        self.assertEqual(report["duplicate_selectors"], ["h1"])

    def test_duplicate_selectors_with_whitespace_variations(self):
        css = """
        b,
        strong {
            font-weight: bolder;
        }
        b, strong {
            color: black;
        }
        """
        report = analyze_css(css)
        self.assertEqual(report["duplicate_selectors"], ["b, strong"])


class TestCSSParserErrorHandling(unittest.TestCase):
    """Tests that malformed CSS inputs raise appropriate ValueErrors."""

    def test_unclosed_comment_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_css("/* unclosed comment\nbody { color: red; }")
        self.assertIn("Unclosed CSS comment", str(ctx.exception))

    def test_unclosed_string_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_css('body { font-family: "Open Sans; }')
        self.assertIn("Unclosed quoted string", str(ctx.exception))

    def test_unclosed_brace_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_css("body { color: red; ")
        self.assertIn("Unclosed CSS rule", str(ctx.exception))


class TestSampleCSSIntegration(unittest.TestCase):
    """End-to-end integration test parsing sample.css."""

    def test_sample_css_benchmark(self):
        sample_path = Path(__file__).with_name("sample.css")
        self.assertTrue(sample_path.exists(), "sample.css should exist")

        css = sample_path.read_text(encoding="utf-8")
        report = analyze_css(css, file_name=sample_path.name)

        self.assertEqual(report["total_rules"], 34)
        self.assertEqual(report["total_declarations"], 56)
        self.assertEqual(len(report["property_frequency"]), 27)
        self.assertEqual(report["duplicate_selectors"], [])


if __name__ == "__main__":
    unittest.main()

