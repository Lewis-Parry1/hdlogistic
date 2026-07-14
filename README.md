# to-be-titled

## Development setup

Install development dependencies and enable pre-commit hooks:

```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pre_commit install
```

## CI

This repository includes a GitHub Actions workflow at `.github/workflows/pre-commit.yml` that runs `pre-commit run --all-files` on pushes and pull requests.
