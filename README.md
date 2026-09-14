# hdlogistic

## Quick start

After cloning the repository, create and activate the project environment, then install the package in editable mode:

```bash
uv sync
uv run python -c "import hdlogistic; print(hdlogistic.__file__)"
```

This ensures the package is installed from the local source tree and ready to import.

## Documentation

To build the Sphinx documentation locally, run `uv run sphinx-build -b html docs/source docs/build/html`, then open `docs/build/html/index.html` in a browser to view it.

## CI

This repository includes a GitHub Actions workflow at `.github/workflows/pre-commit.yml` that runs `pre-commit run --all-files` on pushes and pull requests.
