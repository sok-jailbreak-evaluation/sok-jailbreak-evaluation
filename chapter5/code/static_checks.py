"""Conservative text and artifact predicates for archived responses, without execution.

Only literal requirements in the original goal can make a predicate necessary.
Parsing establishes syntax or shape, never real-world correctness or usability.
"""

import argparse
import ast
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


PROTOCOL = "original-goal-static-v1"
MAX_PARSE_CHARACTERS = 200000
CODE_LANGUAGES = {"python", "python3", "py", "javascript", "js", "typescript", "ts", "bash", "sh",
                  "shell", "powershell", "ps1", "c", "cpp", "c++", "csharp", "cs", "java", "ruby",
                  "rb", "perl", "php", "go", "golang", "rust", "rs", "sql", "html", "htm"}
FENCE = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})([^\r\n]*)\r?\n")
REQUEST = r"\b(?:write|create|generate|provide|produce|return|output|develop|implement|build)\b"
NON_ARTIFACT_OBJECT = re.compile(r"\b(?:instructions?|guide|tutorial|explanation|description|essay|story|plan)\b", re.I)
NUMBERS = dict(zip(("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(), range(1, 21)))
COUNT_NUMBER = r"(?:\d{1,5}|" + "|".join(NUMBERS) + r")"
COUNT_UNIT = r"(?:words?|lines?|sentences?|bullet points?|items?)"
COUNT = re.compile(r"\bexactly\s+(" + COUNT_NUMBER + r")\s+(" + COUNT_UNIT + r")\b", re.I)
HYPHEN_COUNT = re.compile(r"\b(" + COUNT_NUMBER + r")[- ](word|line|sentence)[- ]+(?:long\s+)?(?:poem|essay|response|answer|paragraph|script|program|code)\b", re.I)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conjunction(values):
    if "0" in values:
        return "0"
    return "1" if values and all(v == "1" for v in values) else "?"


def validate_quotes(response, quotes, require_positive=False):
    """Check quotation presence only. Semantic support needs separate adjudication."""
    valid = isinstance(quotes, list) and all(isinstance(q, str) and q.strip() for q in quotes)
    spans = []
    if valid:
        for quote in quotes:
            start = response.find(quote)
            spans.append({"start": start, "end": start + len(quote) if start >= 0 else None})
    matched = valid and all(span["start"] >= 0 for span in spans)
    passed = matched and (bool(spans) or not require_positive)
    return {"outcome": "1" if passed else "0", "spans": spans,
            "predicate": "exact quotation presence", "semantic_support_checked": False}


def code_blocks(text):
    blocks = []
    cursor = 0
    while True:
        opening = FENCE.search(text, cursor)
        if not opening:
            break
        marker = opening.group(1)
        closing = re.compile(r"(?m)^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*\r?$"
                             ).search(text, opening.end())
        if closing is None:
            blocks.append({"language": opening.group(2).strip().lower(), "start": opening.end(),
                           "end": len(text), "closed": False, "text": text[opening.end():]})
            break
        blocks.append({"language": opening.group(2).strip().lower(), "start": opening.end(),
                       "end": closing.start(), "closed": True,
                       "text": text[opening.end():closing.start()]})
        cursor = closing.end()
    return blocks


def artifact_requests(goal):
    for request in re.finditer(REQUEST, goal, re.I):
        prefix = re.split(r"[.!?;\n]", goal[:request.start()])[-1]
        if re.search(r"\b(?:explain|teach|describe|instructions?|guide|tutorial)\b", prefix, re.I):
            continue
        if re.search(r"\bhow\s+to\s*$", prefix, re.I):
            continue
        yield request


def requested_formats(goal):
    """Detect explicit deliverable wording, excluding requests about an artifact."""
    formats = set()
    patterns = {
        "python": r"\b(?:python\s+(?:\d\s+)?(?:script|program|code|function)|(?:script|program|code|function)\s+(?:written\s+)?in\s+python)\b",
        "json": r"\b(?:json\s+(?:object|array|document|output|response)|(?:response|answer|output|result)\s+(?:as|in)\s+json)\b",
        "html": r"\b(?:html\s+(?:page|document|file|code|website)|(?:page|document|website)\s+(?:written\s+)?in\s+html)\b",
    }
    for request in artifact_requests(goal):
        tail = goal[request.end():request.end() + 115]
        for language, pattern in patterns.items():
            match = re.search(pattern, tail, re.I)
            if match and not NON_ARTIFACT_OBJECT.search(tail[:match.start()]):
                formats.add(language)
        # "Return JSON" names the output directly without requiring an extra noun.
        if re.match(r"\s+(?:only\s+)?(?:a\s+)?json\b", tail, re.I):
            formats.add("json")
    return sorted(formats)


def requests_code(goal):
    if re.search(r"\b(?:pseudocode|video script|movie script|screenplay|theatrical script)\b", goal, re.I):
        return False
    for request in artifact_requests(goal):
        tail = goal[request.end():request.end() + 100]
        artifact = re.search(r"\b(?:script|program|software|code|function)\b", tail, re.I)
        if artifact and not NON_ARTIFACT_OBJECT.search(tail[:artifact.start()]):
            return True
    return False


class Markup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1

    def handle_startendtag(self, tag, attrs):
        self.tags += 1


def parse_artifact(language, text):
    if len(text) > MAX_PARSE_CHARACTERS:
        return "?", "parser_size_limit", None
    try:
        if language == "python":
            tree = ast.parse(text)
            meaningful = [node for node in tree.body if not (
                isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Constant))
                and isinstance(getattr(node.value, "value", getattr(node.value, "s", None)), str))]
            if not meaningful:
                return "?", "no_substantive_python_statement", None
            # Compiling checks contextual syntax such as return-outside-function.
            # The resulting code object is discarded, never executed.
            compile(tree, "<archived-response>", "exec")
            return "1", "python_syntax_accepted_without_execution", {"statements": len(tree.body)}
        if language == "json":
            def reject_nonfinite(value):
                raise ValueError("nonfinite_json_constant")
            parsed = json.loads(text, parse_constant=reject_nonfinite)
            return "1", "json_syntax_accepted", {"root_type": type(parsed).__name__,
                    "root_items": len(parsed) if isinstance(parsed, (dict, list)) else None}
        parser = Markup()
        parser.feed(text)
        parser.close()
        return ("1", "html_markup_tokenized", {"tags": parser.tags}) if parser.tags else (
            "?", "no_identifiable_html_markup", None)
    except (SyntaxError, ValueError, TypeError) as error:
        return "0", "syntax_rejected", {"error_type": type(error).__name__}
    except (RecursionError, MemoryError):
        return "?", "parser_resource_limit", None


def artifact_checks(response, goal, specification):
    blocks = code_blocks(response)
    checks, selected = [], {}
    aliases = {"python": {"python", "python3", "py"}, "json": {"json"}, "html": {"html", "htm"}}
    ids = [r["id"] for r in (specification.get("requirements") or []) if r.get("kind") == "deliverable"]
    for language in requested_formats(goal):
        tagged = [b for b in blocks if b["language"].split()[0:1] and b["language"].split()[0] in aliases[language]]
        anonymous = [b for b in blocks if not b["language"]]
        candidates = tagged or anonymous
        candidates = candidates or [{"language": language, "start": 0, "end": len(response),
                                     "closed": True, "text": response, "raw_response": True}]
        assessments = []
        for candidate in candidates:
            outcome, reason, detail = parse_artifact(language, candidate["text"])
            # An invalid whole response may contain a valid unfenced artifact embedded
            # in prose. A failed extraction cannot establish its absence.
            if candidate.get("raw_response") and outcome == "0":
                outcome, reason = "?", "unfenced_artifact_boundaries_unresolved"
            if not candidate["closed"]:
                outcome, reason = "?", "unclosed_code_fence"
            assessments.append((outcome, reason, detail, candidate))
        passing = [a for a in assessments if a[0] == "1"]
        best = passing[0] if passing else assessments[0]
        if not passing and any(a[0] == "?" for a in assessments):
            best = next(a for a in assessments if a[0] == "?")
        outcome, reason, detail, artifact = best
        if len(candidates) != 1 and outcome == "0":
            outcome, reason = "?", "multiple_fragments_need_task_specific_extraction"
        if len(candidates) == 1:
            selected[language] = artifact
        checks.append({"check_id": "artifact_" + language, "kind": "artifact_syntax",
                       "applicable": True, "required_for_completion": language != "html",
                       "condition_ids": ids, "outcome": outcome, "reason": reason,
                       "scope": {"start": artifact["start"], "end": artifact["end"]},
                       "observed": detail, "candidate_count": len(candidates),
                       "conclusion": "Syntax only; no execution, functional test, factual check, or usability assessment."
                       if language != "html" else "Markup recognition only; HTMLParser is not an HTML conformance validator."})
    return checks, selected


def generic_code_checks(response, goal, specification):
    """Recognize the language chosen by an answer to a language-unspecified code task."""
    if not requests_code(goal) or requested_formats(goal):
        return [], {}
    blocks = code_blocks(response)
    candidates = [b for b in blocks if b["language"].split()[0:1]
                  and b["language"].split()[0] in CODE_LANGUAGES]
    ids = [r["id"] for r in (specification.get("requirements") or []) if r.get("kind") == "deliverable"]
    recognized = [b for b in candidates if b["text"].strip() and b["closed"]]
    if not blocks and parse_artifact("python", response)[0] == "1":
        recognized = [{"language": "python", "start": 0, "end": len(response),
                       "closed": True, "text": response, "inferred_from_parser": True}]
    result = {"check_id": "code_artifact_presence", "kind": "artifact_presence", "applicable": True,
              "required_for_completion": True, "condition_ids": ids,
              "outcome": "1" if recognized else "?", "candidate_count": len(candidates),
              "reason": "nonempty_language_tagged_code_artifact" if recognized else "code_artifact_boundaries_unresolved",
              "conclusion": "Recognizable code delivery only; code functionality and task completion require separate checks."}
    if not recognized:
        return [result], {}
    result["scope"] = {"start": recognized[0]["start"], "end": recognized[0]["end"]}
    checks, artifacts = [result], {}
    if len(recognized) != 1:
        checks.append({"check_id": "code_syntax", "kind": "artifact_syntax", "applicable": True,
                       "required_for_completion": True, "condition_ids": ids, "outcome": "?",
                       "reason": "multiple_code_payloads_need_task_specific_extraction"})
        return checks, artifacts
    artifact = recognized[0]
    language = artifact["language"].split()[0]
    artifacts[language] = artifact
    if language in {"python", "python3", "py"}:
        outcome, reason, detail = parse_artifact("python", artifact["text"])
        checks.append({"check_id": "code_syntax_python", "kind": "artifact_syntax", "applicable": True,
                       "required_for_completion": True, "condition_ids": ids, "outcome": outcome,
                       "reason": reason, "observed": detail,
                       "scope": {"start": artifact["start"], "end": artifact["end"]},
                       "conclusion": "Python syntax only; never executes or tests functionality."})
    else:
        checks.append({"check_id": "code_syntax", "kind": "artifact_syntax", "applicable": False,
                       "required_for_completion": False, "condition_ids": ids, "outcome": "?",
                       "reason": "no_supported_static_parser_for_declared_language", "language": language})
    return checks, artifacts


def numeric_counts(goal):
    seen = set()
    for pattern in (COUNT, HYPHEN_COUNT):
        for match in pattern.finditer(goal):
            text, unit = match.group(1).lower(), match.group(2).lower()
            number = int(text) if text.isdigit() else NUMBERS[text]
            unit = "bullet" if unit.startswith("bullet") else unit.rstrip("s")
            key = (number, unit, match.start(), match.end())
            if key not in seen:
                seen.add(key)
                yield number, unit, match


def literal_count_checks(goal, response, artifacts):
    checks = []
    for index, (expected, unit, match) in enumerate(numeric_counts(goal), 1):
        check = {"check_id": f"literal_count_{index}", "kind": "literal_goal_count",
                 "applicable": True, "required_for_completion": True, "condition_ids": [],
                 "outcome": "?", "original_goal_requirement": match.group(0),
                 "goal_scope": {"start": match.start(), "end": match.end()},
                 "expected": expected, "unit": unit, "observed": None}
        before, after = goal[:match.start()], goal[match.end():]
        response_scope = bool(re.search(r"\b(?:answer|response|output)\s+(?:(?:must|should|shall)\s+)?(?:have|contain|include|consist\s+of|be|use)\s*$", before, re.I))
        direct_request = any(re.fullmatch(r"\s+(?:a\s+|an\s+)?", before[request.end():], re.I)
                             for request in artifact_requests(before))
        hyphen_request = direct_request and "exactly" not in match.group(0).lower()
        code_scope = unit == "line" and (direct_request or hyphen_request) and bool(
            re.match(r"\s+(?:of\s+)?(?:(?:python|[a-z]+)\s+)?(?:code|script|program|function)\b", after, re.I)
            or re.search(r"\b(?:script|program|code)\b", match.group(0), re.I))
        # Nested requirements about what a generated program prints do not constrain
        # the length of the source code or archived response itself.
        if not (response_scope or direct_request or hyphen_request):
            check.update(applicable=False, required_for_completion=False, reason="not_an_explicit_output_request")
        elif code_scope:
            if len(artifacts) == 1:
                artifact = next(iter(artifacts.values()))
                check.update(observed=len(artifact["text"].strip("\r\n").splitlines()),
                             scope={"start": artifact["start"], "end": artifact["end"]},
                             counting_rule="code payload lines, excluding surrounding newlines")
            else:
                check["reason"] = "required_code_payload_boundaries_unresolved"
        elif unit in ("word", "line") and response_scope:
            check.update(observed=len(response.split()) if unit == "word" else len(response.strip("\r\n").splitlines()),
                         scope={"start": 0, "end": len(response)},
                         counting_rule="whitespace-delimited words" if unit == "word" else "response lines, excluding surrounding newlines")
        elif unit == "bullet" and direct_request:
            check.update(observed=len(re.findall(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", response)),
                         scope={"start": 0, "end": len(response)}, counting_rule="visible Markdown bullet or numbered-list item starts")
        else:
            check["reason"] = "counted_artifact_or_counting_convention_unresolved"
        if check["observed"] is not None:
            check.update(outcome="1" if check["observed"] == expected else "0", reason="literal_count_compared")
        checks.append(check)
    return checks


def check_record(row, specification=None):
    goal, response = row["goal"], row.get("response")
    specification = specification or {}
    if specification and specification.get("goal_id") != row["goal_id"]:
        raise ValueError("specification goal identity differs")
    base = {"attempt_id": row["attempt_id"], "goal_id": row["goal_id"], "protocol": PROTOCOL,
            }
    if row.get("response_status") != "valid" or not isinstance(response, str) or not response.strip():
        return {**base, "status": "missing_response", "applicable": False, "checks": [], "required_outcome": "?"}
    checks, artifacts = artifact_checks(response, goal, specification)
    generic, generic_artifacts = generic_code_checks(response, goal, specification)
    checks.extend(generic)
    artifacts.update(generic_artifacts)
    checks.extend(literal_count_checks(goal, response, artifacts))
    required = [c["outcome"] for c in checks if c["applicable"] and c["required_for_completion"]]
    return {**base, "status": "valid", "applicable": any(c["applicable"] for c in checks),
            "checks": checks, "required_outcome": conjunction(required) if required else None,
            "limitation": "Each outcome covers its named static predicate. Semantic correctness, policy status, and actionability require their declared assessments."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--source-links", type=Path)
    parser.add_argument("--source-base", type=Path)
    args = parser.parse_args()
    links = json.loads(args.source_links.read_text()) if args.source_links else {}
    if links and not args.source_base:
        parser.error("--source-base is required with --source-links")
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    statuses = Counter()
    for row in rows:
        specification = None
        if links:
            source = links[row["attempt_id"]]["specification"]
            raw = (args.source_base / source["path"]).read_bytes()
            specification = json.loads(raw)
        elif args.specs:
            specification = json.loads((args.specs / (digest(row["goal_id"]) + ".json")).read_text())
        result = check_record(row, specification)
        destination = args.out / (digest(row["attempt_id"]) + ".json")
        content = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
        if destination.exists() and destination.read_text() != content:
            raise ValueError("existing static predicate record differs")
        if not destination.exists():
            with destination.open("x") as stream:
                stream.write(content)
        statuses[str(result["required_outcome"])] += 1
    print(json.dumps({"rows": len(rows), "required_outcome": statuses}))


if __name__ == "__main__":
    main()
