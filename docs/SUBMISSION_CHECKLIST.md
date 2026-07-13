# Submission Checklist

## Required academic deliverables

- [ ] Final IEEE report PDF with generated results.
- [ ] Scientific poster PDF.
- [ ] Organized and commented Python source.
- [ ] Dataset provenance and license notes.
- [ ] Reproducibility instructions.
- [ ] Test evidence and coverage report.
- [ ] Result tables and figures generated from one run ID.
- [ ] ZIP opened and reproduced from a clean directory.

## Repository release gate

- [ ] No `.env`, API key, token, production prompt, or user identifier.
- [ ] `pytest` passes.
- [ ] Branch coverage is at least 85% for implemented modules.
- [ ] `ruff check .` passes.
- [ ] `mypy src tests` passes.
- [ ] GitHub Actions passes on Python 3.12 and 3.13.
- [ ] README contains no empirical claim unsupported by artifacts.
- [ ] Dataset license is compatible with redistribution, or data is downloaded by script.
