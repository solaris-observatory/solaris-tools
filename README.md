# Solaris Tools

A collection of scripts and modules developed within the **Solaris project**.
The main goal is to provide tools related to telescope pointing analysis and solar
observation support.

---

## Installation (for users)

### Requirements
- Python ≥ 3.9
- [Poetry](https://python-poetry.org/docs/#installation) installed

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/solaris-observatory/solaris-tools.git
   cd solaris-tools
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   poetry install
   ```

3. (Optional) Activate the shell environment:
   ```bash
   poetry shell
   ```

Now you can run the command-line tools such as:

```bash
poetry run check-errors-over-time   --start-time "2025-12-21T00:00:00"   --end-time "2025-12-21T03:00:00"   --frequency "30min"   --show-max-error   --plot   --save-plot result.png   --output results.csv
```

---

## Development Setup

### Install development dependencies
```bash
poetry install --with dev
```

### Run all tests
```bash
poetry run pytest
```

### Run test coverage
```bash
poetry run coverage run -m pytest
poetry run coverage report -m
```

### Run linting
```bash
poetry run ruff .
```

### Using Nox (optional automation)
```bash
poetry run nox
```

---

## Test Structure

- Unit tests and property-based tests (via Hypothesis) are located in the `tests/` directory.
- `pytest.ini` defines default behavior for test runs.
