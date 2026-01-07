# Oz Project

A Python project using `uv` for dependency management.

## Commands

- `uv sync` - Install dependencies
- `uv run pytest` - Run tests with coverage
- `uv run python main.py` - Run the main script
- `uv add <package>` - Add a runtime dependency
- `uv add --dev <package>` - Add a dev dependency

## Project Structure

- `main.py` - Main entry point
- `tests/` - Test files (pytest)
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions

## Testing

Tests use pytest with coverage enabled. Run `uv run pytest` to execute tests with coverage report. Coverage configuration is in `pyproject.toml`.

## Code Style

- Python 3.8+ compatible
- Keep functions small and focused
- Write tests for new functionality
