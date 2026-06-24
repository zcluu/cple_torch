# Installation

## Runtime Install

From the repository root:

```bash
python -m pip install .
```

For editable development:

```bash
python -m pip install -e .
```

## Development Install

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra installs test and packaging tools.

## Documentation Install

```bash
python -m pip install -e ".[docs]"
```

Build the documentation:

```bash
mkdocs build
```

Serve locally:

```bash
mkdocs serve
```

## Core Dependencies

CPLE depends on:

- PyTorch
- NumPy
- pandas
- PyYAML
- Sionna

The project reference example can also require the external model project path
and CUDA when running Mamba-based models.
