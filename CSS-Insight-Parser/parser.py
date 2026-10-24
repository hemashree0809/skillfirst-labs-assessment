
import json
import re
from collections import Counter
from pathlib import Path


def remove_comments(css):
    """Remove CSS comments without changing quoted strings."""
    result = []
    i = 0
    quote = None

    while i < len(css):
        char = css[i]

        if quote:
            result.append(char)
            if char == "\\" and i + 1 < len(css):
                i += 1
                result.append(css[i])
            elif char == quote:
                quote = None
            i += 1
        elif char in ("'", '"'):
            quote = char
            result.append(char)
            i += 1
        elif css.startswith("/*", i):
            end = css.find("*/", i + 2)
            if end == -1:
                raise ValueError("Unclosed CSS comment")
            result.append(" ")
            i = end + 2
        else:
            result.append(char)
            i += 1

    if quote:
        raise ValueError("Unclosed quoted string in CSS")

    return "".join(result)


def split_declarations(text):
    """Split declarations at semicolons outside strings and parentheses."""
    parts = []
    start = 0
    depth = 0
    quote = None
    i = 0

    while i < len(text):
        char = text[i]

        if quote:
            if char == "\\":
                i += 2
                continue
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            parts.append(text[start:i])
            start = i + 1

        i += 1

    parts.append(text[start:])
    return parts


def find_matching_brace(css, opening):
    """Find the closing brace that matches an opening brace."""
    depth = 1
    quote = None
    i = opening + 1

    while i < len(css):
        char = css[i]

        if quote:
            if char == "\\":
                i += 2
                continue
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    raise ValueError("Unclosed CSS rule")


def _has_nested_blocks(text):
    """Check if a CSS block body contains nested rule blocks outside quotes."""
    quote = None
    i = 0

    while i < len(text):
        char = text[i]

        if quote:
            if char == "\\":
                i += 2
                continue
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "{":
            return True

        i += 1

    return False


def _parse_declarations(body):
    """Extract declaration key-value pairs from a CSS rule body."""
    declarations = {}

    for item in split_declarations(body):
        if ":" not in item:
            continue

        prop, value = item.split(":", 1)
        prop = prop.strip().lower()
        value = value.strip()

        if prop and value:
            declarations[prop] = value

    return declarations


def parse_css(css):
    css = remove_comments(css)
    rules = []

    def parse_block(text):
        start = 0
        i = 0
        quote = None
        parens = 0

        while i < len(text):
            char = text[i]

            if quote:
                if char == "\\":
                    i += 2
                    continue
                elif char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == "(":
                parens += 1
            elif char == ")":
                parens = max(0, parens - 1)
            elif char == ";" and parens == 0:
                # Standalone at-rules (e.g. @import, @charset) or stray semicolons outside blocks
                start = i + 1
            elif char == "{" and parens == 0:
                header = text[start:i].strip()
                end = find_matching_brace(text, i)
                body = text[i + 1:end]

                if header.startswith("@"):
                    if _has_nested_blocks(body):
                        # At-rules containing nested CSS rules (e.g. @media, @supports, @keyframes)
                        parse_block(body)
                    else:
                        # At-rules containing declarations (e.g. @font-face, @page)
                        declarations = _parse_declarations(body)
                        if declarations:
                            rules.append({
                                "selector": header,
                                "declarations": declarations
                            })
                elif header:
                    declarations = _parse_declarations(body)
                    rules.append({
                        "selector": header,
                        "declarations": declarations
                    })

                i = end
                start = end + 1

            i += 1

    parse_block(css)
    return rules


def analyze_css(css, file_name="sample.css"):
    """Parse CSS and generate a structured analytical summary report."""
    rules = parse_css(css)
    property_counts = Counter()
    duplicate_selectors = set()
    seen_selectors = set()

    for rule in rules:
        for prop in rule["declarations"]:
            property_counts[prop] += 1

        normalized_selector = re.sub(r"\s+", " ", rule["selector"]).strip()
        if normalized_selector in seen_selectors:
            duplicate_selectors.add(normalized_selector)
        seen_selectors.add(normalized_selector)

    report = {
        "file": file_name,
        "total_rules": len(rules),
        "total_declarations": sum(
            len(rule["declarations"]) for rule in rules
        ),
        "property_frequency": dict(property_counts.most_common()),
        "duplicate_selectors": sorted(duplicate_selectors),
        "rules": rules
    }
    return report


def main():
    css_file = Path(__file__).with_name("sample.css")

    try:
        css = css_file.read_text(encoding="utf-8")
        report = analyze_css(css, file_name=css_file.name)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return

    output_file = Path(__file__).with_name("parsed_output.json")
    output_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("CSS parsing completed!")
    print(f"Rules found: {report['total_rules']}")
    print(f"Declarations found: {report['total_declarations']}")
    print(f"Unique properties: {len(report['property_frequency'])}")
    print(f"Duplicate selectors: {len(report['duplicate_selectors'])}")
    print(f"Structured output saved to: {output_file.name}")


if __name__ == "__main__":
    main()
