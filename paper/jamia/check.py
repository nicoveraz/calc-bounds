"""Word counts against JAMIA limits and the list of placeholders still to fill.

Usage: python3 paper/jamia/check.py
"""

import re
from pathlib import Path

text = Path(__file__).with_name("manuscript.md").read_text()
text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)


def section(name: str) -> str:
    m = re.search(rf"^## {name}\n(.*?)(?=^## )", text, flags=re.S | re.M)
    return m.group(1) if m else ""


def words(s: str) -> int:
    s = re.sub(r"!\[.*?\]\(.*?\)", "", s, flags=re.S)  # figure legends
    s = "\n".join(ln for ln in s.splitlines() if not ln.startswith("|"))  # tables
    s = re.sub(r"^\*\*Table \d+\.\*\*.*$", "", s, flags=re.M)  # table captions
    s = re.sub(r"\{\{.*?\}\}", "", s)
    s = re.sub(r"\[@[^\]]*\]", "", s)  # citations
    return len(s.split())


body = [
    "Background and Significance",
    "Objective",
    "Materials and Methods",
    "Results",
    "Discussion",
    "Conclusion",
]
n_abs = words(section("Abstract"))
n_body = sum(words(section(s)) for s in body)
figs = len(re.findall(r"^!\[", text, flags=re.M))
tables = len(re.findall(r"^\*\*Table \d+\.\*\*", text, flags=re.M))
print(f"abstract : {n_abs:5d} words (limit 250)")
print(f"main text: {n_body:5d} words (limit 4,000)")
print(f"tables   : {tables:5d}       (limit 4)")
print(f"figures  : {figs:5d}       (limit 6)")
todo = re.findall(r"\[\[(.*?)\]\]", text, flags=re.S)
print(f"\nplaceholders to fill: {len(todo)}")
for t in todo:
    print("  -", " ".join(t.split())[:110])
