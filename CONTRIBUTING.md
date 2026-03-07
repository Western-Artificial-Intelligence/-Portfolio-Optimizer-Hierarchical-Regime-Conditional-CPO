# Contributing

Thank you for your interest in contributing to the Hierarchical Regime-Conditional CPO project.

## Getting Started

1. **Fork** the repository and clone your fork.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Bloomberg dataset from the [Google Drive link](https://drive.google.com/drive/folders/1cgXNTCe2qkbtR56CPsRL1uga2PDuqypV) and place the CSVs in `data/`.

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and test locally with `python main.py`
3. Ensure the full pipeline runs without errors
4. Submit a pull request with a clear description of your changes

## Project Structure

- **`src/`** — Core library modules (model, data, training, evaluation)
- **`main.py`** — Full pipeline entry point
- **`results/`** — GNN checkpoints and output metrics
- **`experimental/`** — Work-in-progress experiments (not part of the core paper)

## Code Style

- Python 3.10+
- Use type hints where practical
- Add docstrings to public functions
- Keep modules focused: one responsibility per file

## Areas for Contribution

- **Temporal encoder**: Extending the LSTM to bidirectional LSTM (as in CRISP)
- **Higher-frequency features**: Intraday data as node inputs during volatile periods
- **Cross-universe generalization**: Testing the learned attention topology on US equities or fixed income
- **Visualization**: Improved attention weight visualizations and interactive dashboards

## Reporting Issues

Open a GitHub Issue with:
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
