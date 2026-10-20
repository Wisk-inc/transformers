# Releasing helix-lm

The package on PyPI is `helix-lm` (`pip install helix-lm`). The artifacts are built from this directory with
hatchling; the sdist carries `src/`, `tests/`, `docs/`, the README and the licence files.

```bash
cd helix-lm
make style 2>/dev/null || ruff format src tests scripts && ruff check src tests scripts
PYTHONPATH=src python -m pytest tests -q -k "not two_processes"
rm -rf dist && python -m build -o dist .
python -m twine check dist/*
```

Publishing needs a PyPI API token with upload rights on the project. Keep it out of notebooks, chats and the
repository: put it in the machine's environment (for a Claude Code cloud session, an environment secret named
`PYPI_TOKEN`; locally, your shell profile or a keyring), then:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" python -m twine upload dist/helix_lm-<version>*
pip install --no-cache-dir "helix-lm==<version>" && helix info     # verify the published build
```

Each version can be uploaded once; bump `version` in `pyproject.toml` and `__version__` in
`src/helix_lm/__init__.py` together, and tag the commit (`git tag v<version>`).
