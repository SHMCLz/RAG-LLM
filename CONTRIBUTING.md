# Contributing

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code quality
- Add unit tests for new behavior
- Keep configuration in environment variables
- Avoid persisting any external evidence into the local knowledge base

## Pull request checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No secrets committed
