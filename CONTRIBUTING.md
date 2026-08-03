# Contributing

Install development dependencies with `python -m pip install -e ".[dev]"` and
run `pytest`, `ruff check .`, and `ruff format --check .`. Changes to kernel
fusion must include an evaluation-mode numerical-equivalence test. Do not
commit datasets, checkpoints, generated runs, or credentials.
