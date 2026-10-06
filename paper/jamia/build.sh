#!/bin/sh
# Build the JAMIA draft as Word (for submission) from manuscript.md.
# {{table:NAME}} is replaced with ../tables/NAME.md, as in paper/build.sh.
set -eu
cd "$(dirname "$0")"
mkdir -p build
python3 - <<'PY'
import re
from pathlib import Path
text = Path("manuscript.md").read_text()
text = re.sub(r"\{\{table:([^}]+)\}\}", lambda m: Path(f"../tables/{m.group(1)}.md").read_text(), text)
Path("build/manuscript.md").write_text(text)
PY
pandoc build/manuscript.md -o manuscript.docx --citeproc --csl=ama.csl --resource-path=.:..:../figures \
  2> build/pandoc.log || { cat build/pandoc.log; exit 1; }
grep -v "fig5_mimic" build/pandoc.log || true
echo "built paper/jamia/manuscript.docx"
python3 check.py
