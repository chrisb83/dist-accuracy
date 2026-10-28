---
name: dist-accuracy-orchestrate
description: Run the numerical accuracy review of scipy.stats continuous distributions - inventory, one dist-accuracy-analyst subagent per distribution in batches, dist-accuracy-verifier on serious findings, GitHub issue lookup, and reports/INDEX.md. Invoke as /dist-accuracy-orchestrate <dist> [<dist> ...] or /dist-accuracy-orchestrate all.
argument-hint: "<dist> [<dist> ...] | all  [--batch N] [--redo]"
disable-model-invocation: true
allowed-tools: Bash(python ${CLAUDE_SKILL_DIR}/scripts/inventory.py *), Bash(python ${CLAUDE_SKILL_DIR}/scripts/reports.py *)
---

# Orchestrate the accuracy review

Arguments: $ARGUMENTS

You coordinate; the `dist-accuracy-analyst` subagents do the analysis and the
`dist-accuracy-verifier` subagents check it. Do not analyse distributions
yourself, and do not modify SciPy.

All paths are relative to the project root: `reports/`, `refs/`, `scratch/`,
`.cache/`.

## 0. Arguments

- Distribution names, or `all` for every continuous distribution.
- `--batch N`: analysts running in parallel (default 4).
- `--redo`: re-analyse distributions that already have a valid report
  (default: skip them, so an interrupted run can be resumed).
- If no distribution is given, ask the user which ones to run.

## 1. Preflight

```bash
python -c "import sys, numpy, scipy, mpmath; print(sys.version.split()[0], numpy.__version__, scipy.__version__, mpmath.__version__)"
mkdir -p reports refs scratch
```

Stop and tell the user if an import fails. Record the SciPy version; every
report must state it.

## 2. Inventory

```bash
python ${CLAUDE_SKILL_DIR}/scripts/inventory.py <names...> --json reports/inventory.json
```

(`all`: `python ${CLAUDE_SKILL_DIR}/scripts/inventory.py --json reports/inventory.json`.)
Read the output. For each distribution note: implemented methods, related
distributions, light-review flag.

## 3. Plan

- Skip distributions with a valid report (`python ${CLAUDE_SKILL_DIR}/scripts/reports.py check reports/<dist>.md`
  exits 0) unless `--redo`.
- Order: a distribution that others call into goes **before** them in an
  earlier batch (e.g. `burr` before `fisk`), so the later analyst can cite its
  report under "Shared code" instead of re-deriving it. Otherwise
  alphabetical.
- Show the user the plan (number of distributions, batches, light-review
  ones) in two or three lines, then start without waiting for approval.

## 4. Dispatch analysts, batch by batch

For each distribution in the batch, launch a `dist-accuracy-analyst` subagent
(all in the same message so they run in parallel) with this prompt:

```
Review the numerical accuracy of scipy.stats.<dist> (SciPy <version>).
Follow the dist-accuracy-review skill. Write refs/<dist>_ref.py and
reports/<dist>.md; scratch files only under scratch/<dist>/.

Inventory entry:
<the JSON object for <dist> from reports/inventory.json>

Related reports already written: <paths, or "none">.
<If light_review: "Light review: code reading only, no measurements.">
```

When the batch has finished:

1. `python ${CLAUDE_SKILL_DIR}/scripts/reports.py check reports/<dist>.md` for
   each. If malformed or missing, send the problems back to the same analyst
   once (SendMessage, or a new analyst with the problems listed). If it fails
   again, record it under Notes in INDEX.md and move on.
2. Give the user one line per distribution (worst severity, finding counts).

## 5. Verify serious findings

For each report with at least one **loss** or **severe** finding, launch a
`dist-accuracy-verifier` subagent (in parallel, up to the batch size):

```
Verify the serious findings in reports/<dist>.md for scipy.stats.<dist>
(SciPy <version>), following your instructions.
```

If a verifier reports "not reproduced" or "reference disagreement", send the
verifier's note to a new `dist-accuracy-analyst` for that distribution with
the instruction to re-examine only those findings and correct the report.
Do not edit findings yourself.

## 6. Existing GitHub issues

For each report with findings, search SciPy's issue tracker:

```bash
gh search issues --repo scipy/scipy "<dist>" --limit 30 --json number,title,state,url
```

If `gh` is unavailable or not authenticated, use web search
(`site:github.com/scipy/scipy/issues <dist> accuracy`). Keep issues that are
about the accuracy, overflow, nan/inf or warnings of an affected method.
Replace "not checked" under `## Existing GitHub issues` in the report with a
list (`- #1234 (open): title - relates to F2`) or "none found". This is the
only edit you make to analyst reports.

## 7. Index

```bash
python ${CLAUDE_SKILL_DIR}/scripts/reports.py index
```

Then fill in the hand-written sections of `reports/INDEX.md`:

- **Cross-cutting issues:** problems in shared helpers or `scipy.special`
  functions (from the reports' "Shared code" sections), and failure patterns
  that recur across distributions (e.g. "sf computed as 1 - cdf in N
  distributions"), each with the affected distributions. These are the most
  valuable fixes.
- **Notes:** distributions that could not be completed, regimes that could
  not be measured, malformed reports.

## 8. Final message to the user

Five to ten lines: how many distributions were reviewed, the worst ones
(link to INDEX.md), the cross-cutting issues, verification results, and
anything that needs their decision.
