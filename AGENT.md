AGENT guide for this repo

Build / run / test / lint

- Environment: Python >=3.12. Dependencies in [pyproject.toml](pyproject.toml). Uses a local [.env](.env) with `OPENAI_API_KEY` (+ optional `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT`).
- Install/run with uv (preferred): `uv run python -V`; run a script: `uv run python todos-test/normal_functions.py`.
- Alternative (venv already present): `python todos-test/normal_functions.py`.
- Tests: none configured. When pytest exists, single test: `pytest path::TestClass::test_name -q`.
- Lint/format/type: none configured. Recommended: `ruff check .`, `black .`, `mypy .`.

Architecture / structure

- Minimal Python project. Examples in [todos-test/](todos-test/):
  - [normal_functions.py](todos-test/normal_functions.py): OpenAI Responses API with JSON-function tools (`get_current_datetime`, `add_todo`).
  - [cfg_functions.py](todos-test/cfg_functions.py): Same flow using Lark grammars [get_current_datetime.lark](todos-test/get_current_datetime.lark) and [add_todo.lark](todos-test/add_todo.lark).
- Data store: [todos-test/output/todos.json](todos-test/output/todos.json) (created/overwritten each run).
- External API: OpenAI `responses.create`; tool-calling loop; secrets loaded from `.env`.

Code style / conventions

- Imports: stdlib, third-party, then local; one per line. Prefer `pathlib` over `os.path` (`from pathlib import Path`).
- Logging: `logging` at INFO; never log secrets.
- Types: optional; if added, prefer Python typing; later `mypy --strict`.
- Naming: snake_case (funcs/vars), UPPER_SNAKE (constants), CapWords (classes).
- Errors: raise precise exceptions; handle file I/O with `FileNotFoundError`; avoid broad `except`.
- Formatting: if added, Black (88 cols); keep lines short and functions small.

Tooling rules

- No Cursor/Claude/Windsurf/Cline/Goose/Copilot rule files present. If added later, mirror their guidance here.
