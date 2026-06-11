# sources/ — point-in-time snapshots

These files are **frozen copies** taken from `main` on **2026-06-11**, kept here so
the report branch is self-contained for grounding. They are **not** the source of
truth and **may go stale** as the other branches evolve.

| File | Snapshot of |
|---|---|
| `METHODOLOGY.md` | `main:METHODOLOGY.md` |
| `RESEARCH_JOURNAL.md` | `main:RESEARCH_JOURNAL.md` |
| `CLAUDE.md` | `main:CLAUDE.md` |

## Anti-staleness / anti-hallucination protocol

- For **definitions and decisions**, these snapshots are fine as a working reference.
- For anything about **games, scripts, parameters, or results**, read the file
  **live** from the branch that owns it and cite the branch:

  ```bash
  git show main:METHODOLOGY.md
  git show main:RESEARCH_JOURNAL.md
  git show feature/cheap-talk:experiments/cheap_talk_langchain.py
  git show feature/commons-dilemma:experiments/commons_dilemma_langchain.py
  git show feature/cheap-talk:analysis/cheap_talk_full_analysis.ipynb
  ```

## Re-sync a snapshot

```bash
git show main:METHODOLOGY.md      > sources/METHODOLOGY.md
git show main:RESEARCH_JOURNAL.md > sources/RESEARCH_JOURNAL.md
git show main:CLAUDE.md           > sources/CLAUDE.md
```

Re-sync and note the new date in this file whenever you rely on these for a
factual claim in the report.
