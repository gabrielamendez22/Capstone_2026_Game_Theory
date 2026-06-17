#!/usr/bin/env bash
# Build the capstone report PDF from Markdown using Pandoc + citeproc (APA).
# Usage:  ./build.sh        (from the report/ directory or anywhere)
set -euo pipefail
cd "$(dirname "$0")"

# Pick whichever LaTeX engine is installed.
if   command -v tectonic >/dev/null 2>&1; then ENGINE=tectonic
elif command -v xelatex  >/dev/null 2>&1; then ENGINE=xelatex
elif command -v lualatex >/dev/null 2>&1; then ENGINE=lualatex
elif command -v pdflatex >/dev/null 2>&1; then ENGINE=pdflatex
else
  echo "No LaTeX engine found. Install one, e.g.:"
  echo "    brew install tectonic           # lightweight, recommended"
  echo "    brew install --cask basictex     # full TeX (provides xelatex)"
  exit 1
fi

# The report is split into per-section files so the team can edit different
# sections without merge conflicts. Pandoc concatenates them in filename order
# (00, 01, 02, …); metadata.yaml supplies the title block and citation settings.
echo "Building report.pdf with pandoc (engine: $ENGINE)…"
pandoc metadata.yaml sections/*.md \
  --citeproc \
  --pdf-engine="$ENGINE" \
  -o report.pdf

echo "✓ Wrote $(pwd)/report.pdf"
