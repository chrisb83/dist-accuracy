# Numerical accuracy review of scipy.stats continuous distributions

Claude Code agents and skills that find floating-point accuracy problems in
the implemented methods (pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf) of
`scipy.stats` continuous distributions, measure them against mpmath, and
document them in `reports/<dist>.md`. Nothing in SciPy is changed.

## Contents

```
.claude/
  agents/
    dist-accuracy-analyst.md      one distribution -> refs/<dist>_ref.py, reports/<dist>.md
    dist-accuracy-verifier.md     independently re-checks loss/severe findings
  skills/
    dist-accuracy-review/         how to review one distribution (used by both agents)
      SKILL.md
      references/patterns.md          failure patterns and how to spot them
      references/report-template.md   report format and severity scale
      references/burr_ref_example.py  worked example: reference module
      references/burr-example.md      worked example: report (SciPy 1.18.1)
      scripts/accuracy.py             measurement tool
    dist-accuracy-orchestrate/    run by you: /dist-accuracy-orchestrate ...
      SKILL.md
      scripts/inventory.py            implemented/inherited methods, shapes, related dists
      scripts/reports.py              report format check, INDEX.md table
```

Created while running: `reports/` (reports, `inventory.json`, `INDEX.md`),
`refs/` (mpmath reference modules), `scratch/`, `.cache/dist-accuracy/`.

## Setup

1. Copy this folder's `.claude/` directory into an empty project directory
   (or use this folder as the project directory).
2. Activate the environment that has SciPy, NumPy, mpmath and matplotlib, e.g.
   `conda activate ds`. The tools call `python`, so it must resolve to that
   environment. Check with:
   `python -c "import scipy, mpmath; print(scipy.__version__, mpmath.__version__)"`
3. Start Claude Code in the project directory. A recent version is needed
   (skills with `${CLAUDE_SKILL_DIR}`, and the `skills:` field in agent files).

### Permissions

The agents run many `python` commands (the tools and small mpmath checks)
and write files under `refs/`, `reports/` and `scratch/`. Subagents run in the
background, so permission prompts would stall them. Either allow these up
front with `/permissions` (for example `Bash(python *)`, `Bash(mkdir *)`,
`Bash(gh search issues *)` and edits in this directory), or start Claude Code
with `--permission-mode acceptEdits` and allow `Bash(python *)`.

## Pilot run

```
/dist-accuracy-orchestrate fisk gamma
```

What to expect:

- `fisk` delegates every method to `burr` with d = 1, so its findings should
  match the burr worked example (SciPy 1.18.1). That is the calibration test.
- `gamma` implements pdf, logpdf, cdf, sf, ppf, isf; logcdf and logsf are
  inherited and should appear under "Opportunities".
- Running `accuracy.py` takes seconds to a few minutes per distribution
  (references are evaluated at 400 digits; results are cached).

After the pilot, read `reports/fisk.md`, `reports/gamma.md` and
`reports/INDEX.md`, and adjust the skill (`SKILL.md`, `patterns.md`, the
template) before running more distributions.

## Full run

```
/dist-accuracy-orchestrate all --batch 4
```

Interrupted runs resume: distributions with a valid report are skipped
unless you pass `--redo`.

## Using the tools directly

```bash
# inventory of one or more distributions
python .claude/skills/dist-accuracy-orchestrate/scripts/inventory.py gamma

# full measurement with a reference module
python .claude/skills/dist-accuracy-review/scripts/accuracy.py refs/gamma_ref.py --verbose --plot gamma.png

# one point (reproducer)
python .claude/skills/dist-accuracy-review/scripts/accuracy.py refs/gamma_ref.py --at sf 1e3 30

# compare a candidate implementation (for the later fixing step)
python .claude/skills/dist-accuracy-review/scripts/accuracy.py refs/burr_ref.py --candidate burr_new.py:burr

# report checks and index
python .claude/skills/dist-accuracy-orchestrate/scripts/reports.py status reports/*.md
python .claude/skills/dist-accuracy-orchestrate/scripts/reports.py index
```

## How errors are judged

- Error in ulp: |computed - reference| / (|reference| * 2.2e-16).
- Only results whose true value is a normal double are measured.
- A point is within conditioning if the error is at most 8 + cond ulp, where
  cond = |x F'(x) / F(x)| is the condition number with respect to the
  method's argument (x, or q for ppf/isf). Shape parameters are treated as
  exact; their condition number is shown for context only.
- Beyond conditioning: loss (0/inf/nan or no correct digit), severe
  (> 1e6 ulp), moderate (> 1e3 ulp), minor.
