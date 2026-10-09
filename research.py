"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    raw = topic.strip().lower()
    cleaned = re.sub(r"[^\w]+", "-", raw)
    cleaned = cleaned.strip("-")
    cleaned = cleaned[:60].rstrip("-")
    return cleaned if cleaned else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please research and write an in-depth, comprehensive survey report on the topic: '{topic}'.\n\n"
        "Follow your complete workflow:\n"
        "1. Plan with write_todos and split into at least 3 sub-questions.\n"
        "2. Delegate each sub-question to researcher subagents using the task tool. Require each researcher to gather at least 2 source families.\n"
        "3. Read notes and compile all verified sources into /tmp/work/research/sources.json.\n"
        "   Ensure the collection of sources covers at least 3 distinct source families (arxiv, hf-daily, hf-search, web).\n"
        "4. Write the report body to /tmp/work/report/report.md following REPORT_TEMPLATE.md.\n"
        "   Do NOT write the ## References section yourself.\n"
        "5. Execute /tmp/work/research/finalize_citations.py in the sandbox to generate references.\n"
        "6. Execute /tmp/work/research/check_citations.py until it outputs OK.\n"
        "7. Spot-check 3-5 claims with the citation-checker subagent."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        tcs = getattr(msg, "tool_calls", None)
        if tcs is None and isinstance(msg, dict):
            tcs = msg.get("tool_calls", [])
        if tcs:
            for tc in tcs:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name:
                    tool_counts[name] += 1

        usage = getattr(msg, "usage_metadata", None)
        if usage is None and isinstance(msg, dict):
            usage = msg.get("usage_metadata")
        if usage:
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
        else:
            resp_meta = getattr(msg, "response_metadata", None) or (msg.get("response_metadata") if isinstance(msg, dict) else {})
            tu = resp_meta.get("token_usage", {}) if isinstance(resp_meta, dict) else {}
            if tu:
                input_tokens += int(tu.get("prompt_tokens") or 0)
                output_tokens += int(tu.get("completion_tokens") or 0)

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_counts.get("task", 0),
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def _extract_sources_from_notes(backend):
    try:
        ls_res = backend.execute(f"ls {NOTES_DIR}")
        filenames = [f.strip() for f in ls_res.output.splitlines() if f.strip().endswith(".md")]
        if not filenames:
            return None
        note_paths = [f"{NOTES_DIR}/{fn}" for fn in filenames]
        note_files = download(backend, note_paths)
        sources = []
        seen_urls = set()
        n = 1
        for path, content in note_files.items():
            if not content:
                continue
            text = content.decode("utf-8", errors="replace")
            blocks = re.split(r"(?m)^##\s+Source", text)
            for b in blocks[1:]:
                title_m = re.search(r"-\s*Title:\s*(.+)", b)
                id_m = re.search(r"-\s*ID:\s*(.+)", b)
                url_m = re.search(r"-\s*URL:\s*(.+)", b)
                date_m = re.search(r"-\s*Date:\s*(.+)", b)
                src_m = re.search(r"-\s*Source:\s*(.+)", b)
                url = url_m.group(1).strip() if url_m else ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = title_m.group(1).strip() if title_m else "Untitled"
                paper_id = id_m.group(1).strip() if id_m else url.split("/")[-1]
                date = date_m.group(1).strip() if date_m else "2025-01-01"
                source = src_m.group(1).strip() if src_m else "web"
                sources.append({
                    "n": n,
                    "id": paper_id,
                    "url": url,
                    "title": title,
                    "date": date,
                    "source": source,
                })
                n += 1
        return sources if sources else None
    except Exception:
        return None


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH, f"{WORKDIR}/report.md", f"{WORKDIR}/sources.json"])
    report_bytes = files.get(REPORT_PATH) or files.get(f"{WORKDIR}/report.md")
    sources_bytes = files.get(SOURCES_PATH) or files.get(f"{WORKDIR}/sources.json")

    # If sources.json is missing in sandbox, try building it from notes
    if not sources_bytes or not sources_bytes.strip():
        extracted = _extract_sources_from_notes(backend)
        if extracted:
            sources_json_str = json.dumps(extracted, indent=2, ensure_ascii=False)
            upload(backend, {SOURCES_PATH: sources_json_str.encode("utf-8")})
            sources_bytes = sources_json_str.encode("utf-8")

    # If report is missing in sandbox, check if the LLM output the report in its response text
    if not report_bytes or not report_bytes.strip():
        for msg in reversed(messages):
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if isinstance(content, str) and ("# " in content and "## Background" in content):
                report_bytes = content.strip().encode("utf-8")
                upload(backend, {REPORT_PATH: report_bytes})
                backend.execute(f"python3 {FINALIZER_PATH}")
                backend.execute(f"python3 {VALIDATOR_PATH}")
                updated_files = download(backend, [REPORT_PATH, SOURCES_PATH])
                if updated_files.get(REPORT_PATH):
                    report_bytes = updated_files[REPORT_PATH]
                if updated_files.get(SOURCES_PATH):
                    sources_bytes = updated_files[SOURCES_PATH]
                break

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report at {REPORT_PATH} is missing or empty")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources at {SOURCES_PATH} is missing or empty")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise ValueError("sources.json must be a JSON list")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse sources.json: {exc}")

    slug = slugify(topic)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    summary = summarize(messages, elapsed, model_name)
    distinct_families = sorted(list({s.get("source") for s in sources_data if isinstance(s, dict) and s.get("source")}))

    meta = {
        "topic": topic,
        "model": summary["model"],
        "elapsed_s": summary["elapsed_s"],
        "subagent_calls": summary["subagent_calls"],
        "tool_calls": summary["tool_calls"],
        "tokens": summary["tokens"],
        "n_sources": len(sources_data),
        "source_families": distinct_families,
    }

    (reports_path / f"{slug}.sources.json").write_text(json.dumps(sources_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_path / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_path / f"{slug}.md").write_text(report_bytes.decode("utf-8"), encoding="utf-8")

    return reports_path / f"{slug}.md"


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    clean_topic = topic.strip()
    if not clean_topic:
        sys.stderr.write("Usage: python research.py \"<topic>\"\n")
        return 2

    model = make_model()
    if hasattr(model, "max_retries"):
        model.max_retries = 10
    model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or getattr(model, "model_name", getattr(model, "model", "unknown-model"))
    start = time.monotonic()

    print(f"[*] Starting research for: {clean_topic} (model: {model_name})", flush=True)
    with open_sandbox() as backend:
        print("[*] Initializing sandbox directories and scripts...", flush=True)
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        print("[*] Building lead agent and starting autonomous research...", flush=True)
        agent = build_lead_agent(backend, model)
        result = None
        for attempt in range(3):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": build_prompt(clean_topic)}]},
                    config={"recursion_limit": 1000},
                )
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"[!] Transient network error during invoke ({exc}). Retrying in 10s (attempt {attempt + 1}/3)...", flush=True)
                time.sleep(10.0)

        messages = result.get("messages", []) if isinstance(result, dict) else []
        elapsed = time.monotonic() - start
        print(f"[*] Research workflow completed in {elapsed:.1f}s. Saving outputs...", flush=True)

        try:
            report_path = save_outputs(backend, clean_topic, messages, elapsed, model_name)
            print(f"[+] Research completed successfully. Report saved to: {report_path}", flush=True)
            return 0
        except RuntimeError as exc:
            sys.stderr.write(f"[-] FAILED: {exc}\n")
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
