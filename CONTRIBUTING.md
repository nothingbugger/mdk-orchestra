# Contributing to MDK Orchestra

Thank you for your interest. This is a small, volunteer-maintained project — contributions that align with the existing architecture and style are very welcome.

## Getting set up

```bash
git clone https://github.com/nothingbugger/mdk-orchestra
cd mdk-orchestra
pip install -e ".[dev]"
pytest tests/
```

If you want to run the Orchestra end-to-end locally before contributing:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
mdk-orchestra demo --duration 3
```

## Development workflow

1. Open an issue first for anything non-trivial — it's cheaper to align on the shape than to rewrite a PR.
2. Branch off `main`, make focused commits, open a PR with a clear description of the change and its motivation.
3. Keep each PR small. "Refactor + feature + bug fix" in one PR is three PRs.
4. Add tests for behavior you've changed or added. See `tests/` for examples.

## Code style

- **Python 3.11+**, type hints on public functions.
- **Formatting**: `black` with line length 100. `isort` profile `black`.
- **Linting**: `ruff` (config in `pyproject.toml`).
- **Commits**: short, imperative subject line (under 72 chars). Body if non-obvious.

Run before pushing:

```bash
black .
isort .
ruff check .
pytest tests/
```

## Areas that welcome contributions

- **New LLM backends** for providers not yet covered. See [docs/extending.md](docs/extending.md) for the backend interface and registration.
- **New fault types** in the simulator. The fault injection framework in `simulator/faults.py` is designed to accept new fault kinds without touching the rest of the system.
- **Additional deterministic flaggers** (beyond the current rule engine + 2 XGBoost predictors). Validate with proper cross-miner splits before submitting.
- **Dashboard improvements** — the current TUI is functional but minimal.
- **Documentation** — clarifications, corrections, translations.

## What's out of scope (for v0.1)

- **Streaming LLM responses** — the current backend interface is blocking. Adding streaming support is a v0.2+ conversation.
- **Non-Bitcoin-mining domains** — the simulator, schemas, and specialist personalities are mining-specific. Adapting to a different domain is a fork, not a PR.
- **Production deployment tooling** — this is a demonstration repo. Packaging for k8s / systemd / etc. lives downstream.

## Filing a bug

Include:
- MDK Orchestra version (`pip show mdk-orchestra`)
- Python version (`python --version`)
- OS
- Full command that triggered the bug
- Full traceback or relevant log excerpt
- Whether you're using Anthropic, local LLM, or a third-party provider

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0, matching the rest of the project.
