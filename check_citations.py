"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            try:
                numbers.append(int(part))
            except ValueError:
                pass
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    seen_urls = {}
    source_by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not an object: {entry!r}")
            continue
        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source {entry!r}: n must be an int")
            continue
        if n in source_by_n:
            problems.append(f"source n={n} appears multiple times in sources.json")
        source_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}]: url must start with http:// or https:// (got {url!r})")
        else:
            if url in seen_urls:
                problems.append(f"duplicate url {url} in sources.json (sources [{seen_urls[url]}] and [{n}])")
            else:
                seen_urls[url] = n

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("report is missing heading '## References'")
        body = report_text.rstrip()
        ref_section = ""
    else:
        body = report_text[:matches[-1].start()].rstrip()
        ref_section = report_text[matches[-1].end():]

    # Find cited numbers in body only, ignoring code spans
    cited = set()
    segments = _CODE.split(body)
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # code block or inline code
        for match in _GROUP.finditer(segment):
            for n in _group_numbers(match.group(1)):
                cited.add(n)

    # Validate citations against sources
    for n in sorted(cited):
        if n not in source_by_n:
            problems.append(f"[{n}] cited but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # Inspect ## References section lines starting with [n]
    ref_lines_by_n = {}
    for raw_line in ref_section.splitlines():
        line = raw_line.strip()
        m = _REF_LINE.match(line)
        if m:
            ref_n = int(m.group(1))
            ref_lines_by_n.setdefault(ref_n, []).append((line, m.group(2)))

    # Every source needs exactly one reference line
    for n in sorted(source_by_n.keys()):
        lines = ref_lines_by_n.get(n, [])
        if len(lines) == 0:
            problems.append(f"source [{n}] missing reference line in ## References")
        elif len(lines) > 1:
            problems.append(f"source [{n}] has {len(lines)} reference lines in ## References")

    # No reference lines for numbers that are not sources
    for n in sorted(ref_lines_by_n.keys()):
        if n not in source_by_n:
            problems.append(f"reference line [{n}] does not exist in sources.json")
        else:
            # Check URL for each reference line of this source
            for line, rest in ref_lines_by_n[n]:
                urls = re.findall(r"https?://\S+", line)
                if len(urls) != 1:
                    problems.append(f"reference [{n}] must contain exactly one URL, found {len(urls)}")
                else:
                    found_url = urls[0]
                    expected_url = source_by_n[n].get("url", "")
                    # Strip trailing punctuation if expected_url does not end with it
                    clean_found = found_url.rstrip(".,;)>\"'") if not expected_url.endswith(found_url[-1]) else found_url
                    if found_url != expected_url and clean_found != expected_url:
                        problems.append(f"reference [{n}] URL mismatch: expected {expected_url}, got {found_url}")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
