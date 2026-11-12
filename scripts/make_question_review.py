"""Build a review page for the question set: each label next to its evidence in context.

    python -m scripts.make_question_review          # writes eval/questions/review.html
    python -m scripts.apply_question_review <downloaded review.json> --by <name>

Open the page (served by .claude/launch.json "question-review", or just open the file),
mark each question OK or Problem (+ note), then "Download review.json".
Stdlib only.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from scripts.check_questions import DOCS, QUESTIONS, normalize

OUT = QUESTIONS.parent / "review.html"
CONTEXT_LINES = 2


def context(path: str, quote: str) -> str:
    """The quote's line +-CONTEXT_LINES from the raw file, with the quote highlighted."""
    lines = (DOCS / path).read_text(encoding="utf-8").splitlines()
    probe = normalize(quote)[:40]
    hit = next((i for i, line in enumerate(lines) if probe in normalize(line)), None)
    if hit is None:  # quote spans lines: fall back to the quote alone
        return f"<mark>{html.escape(quote)}</mark>"
    out = []
    for i in range(max(0, hit - CONTEXT_LINES), min(len(lines), hit + CONTEXT_LINES + 1)):
        text = html.escape(lines[i])
        if i == hit:
            esc_q = html.escape(quote)
            text = text.replace(esc_q, f"<mark>{esc_q}</mark>") if esc_q in text else f"<mark>{text}</mark>"
        out.append(f'<span class="ln">{i + 1}</span>{text}')
    return "\n".join(out)


def evidence(q: dict, kind: str, title: str) -> str:
    items = q.get(kind, [])
    if not items:
        return ""
    blocks = "".join(
        f'<div class="src"><div class="path">{html.escape(s["path"])}</div>'
        f'<pre>{context(s["path"], s["quote"])}</pre></div>'
        for s in items
    )
    return f'<h4 class="{kind}">{title}</h4>{blocks}'


def card(q: dict) -> str:
    facts = ", ".join(f"<code>{html.escape(f)}</code>" for f in q.get("key_facts", [])) or "—"
    notes = f'<p class="notes">Note: {html.escape(q["notes"])}</p>' if q.get("notes") else ""
    return f"""
<section class="q" id="{q['id']}" data-id="{q['id']}">
  <header><span class="id">{q['id']}</span><span class="tag {q['type']}">{q['type']}</span>
    <span class="expect">expect: {q['expect']}</span></header>
  <p class="question">{html.escape(q['question'])}</p>
  <p class="answer"><b>Expected answer:</b> {html.escape(q['answer'])}</p>
  <p class="facts"><b>Key facts:</b> {facts}</p>
  {notes}
  {evidence(q, 'sources', 'Evidence (must support the answer)')}
  {evidence(q, 'near_miss', 'Near misses (look relevant, must NOT answer it)')}
  <div class="verdict">
    <label><input type="radio" name="v-{q['id']}" value="ok"> OK</label>
    <label><input type="radio" name="v-{q['id']}" value="problem"> Problem</label>
    <input type="text" class="note" placeholder="what's wrong? (optional for OK)">
  </div>
</section>"""


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Question Review</title>
<style>
:root {{ --bg:#fafaf9; --card:#fff; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4; --mark:#fde68a;
        --ok:#15803d; --bad:#b91c1c; --code:#f5f5f4; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --card:#292524; --fg:#f5f5f4; --muted:#a8a29e;
        --line:#44403c; --mark:#854d0e; --code:#1c1917; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:16px; }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line);
       display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:1; }}
button {{ font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--line); background:var(--card);
         color:var(--fg); cursor:pointer; }}
.q {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:16px 0; }}
.q.done-ok {{ border-left:5px solid var(--ok); }} .q.done-problem {{ border-left:5px solid var(--bad); }}
header {{ display:flex; gap:10px; align-items:center; color:var(--muted); font-size:13px; }}
.id {{ font-weight:700; color:var(--fg); }}
.tag {{ padding:1px 8px; border-radius:99px; border:1px solid var(--line); }}
.question {{ font-size:17px; font-weight:600; margin:8px 0; }}
.notes {{ color:var(--muted); font-style:italic; }}
h4 {{ margin:12px 0 4px; font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
h4.near_miss {{ color:var(--bad); }}
.path {{ font:12px ui-monospace, monospace; color:var(--muted); }}
pre {{ white-space:pre-wrap; word-break:break-word; background:var(--code); padding:8px; border-radius:6px;
      font:12.5px/1.45 ui-monospace, monospace; margin:2px 0 8px; }}
mark {{ background:var(--mark); color:inherit; }}
.ln {{ display:inline-block; min-width:3.2em; color:var(--muted); user-select:none; }}
code {{ background:var(--code); padding:0 4px; border-radius:4px; }}
.verdict {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; border-top:1px solid var(--line);
           padding-top:10px; margin-top:6px; }}
.note {{ flex:1; min-width:200px; font:inherit; padding:5px 8px; border-radius:6px; border:1px solid var(--line);
        background:var(--bg); color:var(--fg); }}
</style></head><body><main>
<h1>Question set review</h1>
<p>{count} questions labeled against corpus commit <code>{commit}</code>. For each one: does the evidence
really support the expected answer (and do near misses really <em>not</em> answer it)? Is the question
fair and the answer complete? Your marks are kept in this browser until you download them.</p>
<div class="bar"><span id="progress"></span>
  <button id="download">Download review.json</button>
  <button id="copy">Copy review to clipboard</button>
  <span id="copied"></span></div>
<details id="rawbox"><summary>Show review JSON (select all, copy, paste into chat)</summary>
  <textarea id="raw" readonly rows="10" style="width:100%;font:12px ui-monospace,monospace"></textarea>
</details>
{cards}
</main>
<script>
const KEY = "question-review-{commit}";
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} render(); }};
function render() {{
  const qs = document.querySelectorAll(".q");
  let done = 0, bad = 0;
  qs.forEach(q => {{
    const s = state[q.dataset.id];
    q.classList.remove("done-ok", "done-problem");
    if (s && s.verdict) {{ done++; q.classList.add("done-" + s.verdict); if (s.verdict === "problem") bad++; }}
  }});
  document.getElementById("progress").textContent = `${{done}} / ${{qs.length}} reviewed, ${{bad}} problem(s)`;
}}
document.querySelectorAll(".q").forEach(q => {{
  const id = q.dataset.id, s = state[id] || {{}};
  const note = q.querySelector(".note");
  note.value = s.note || "";
  q.querySelectorAll("input[type=radio]").forEach(r => {{
    r.checked = r.value === s.verdict;
    r.addEventListener("change", () => {{ state[id] = {{...state[id], verdict: r.value}}; save(); }});
  }});
  note.addEventListener("input", () => {{ state[id] = {{...state[id], note: note.value}}; save(); }});
}});
const payload = () => JSON.stringify({{commit: "{commit}", reviews: state}}, null, 2);
const showRaw = () => {{ document.getElementById("raw").value = payload(); }};
document.getElementById("download").addEventListener("click", () => {{
  showRaw();
  try {{
    const blob = new Blob([payload()], {{type: "application/json"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "review.json";
    document.body.appendChild(a); a.click(); a.remove();
  }} catch (e) {{ document.getElementById("rawbox").open = true; }}
}});
document.getElementById("copy").addEventListener("click", async () => {{
  showRaw();
  const msg = document.getElementById("copied");
  try {{ await navigator.clipboard.writeText(payload()); msg.textContent = "Copied - paste it into the chat."; }}
  catch (e) {{
    const box = document.getElementById("raw"); document.getElementById("rawbox").open = true;
    box.focus(); box.select();
    try {{ document.execCommand("copy"); msg.textContent = "Copied - paste it into the chat."; }}
    catch (e2) {{ msg.textContent = "Select the text below and copy it (Ctrl+C)."; }}
  }}
}});
document.getElementById("rawbox").addEventListener("toggle", showRaw);
render();
</script></body></html>
"""


def main() -> None:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    cards = "".join(card(q) for q in data["questions"])
    OUT.write_text(PAGE.format(count=len(data["questions"]), commit=data["corpus_commit"], cards=cards),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(QUESTIONS.parents[2])}")


if __name__ == "__main__":
    main()
