# debias-distill
A study of de-bias-ing LLMs using distillation

## Setup

Requires pyenv (Python 3.12, pinned in `.python-version`) and Poetry 2.x.

```bash
poetry env use "$(pyenv which python)"
poetry install --with eval,dev
poetry run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
