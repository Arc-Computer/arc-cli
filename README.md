# ARC Evaluation Platform

ARC is a comprehensive evaluation platform designed to assess AI agents' capabilities in handling complex, real-world scenarios across multiple domains.

## Overview

The ARC platform provides:
- Scenario generation and management
- AI agent evaluation framework
- Performance metrics and reporting
- Integration with multiple AI providers
- Comprehensive testing suite

## Project Structure

```bash
arc/
├── database/          # PostgreSQL database layer
│   ├── schema/        # Database schemas and migrations
│   ├── seed/          # Seed data and failure patterns
│   └── client.py      # AsyncPostgreSQLStorage implementation
├── sandbox/           # Simulation environment
│   ├── engine/        # Core simulation and trajectory capture
│   ├── scenarios/     # Scenario generation and quality scoring
│   └── evaluation/    # Judge and reliability scoring
├── recommendations/   # Configuration optimization
│   ├── diff_engine.py      # YAML configuration changes
│   └── failure_analyzer.py # ML-based clustering
├── api/               # CLI and API interfaces
│   ├── cli.py         # Primary CLI interface
│   └── models.py      # Pydantic schemas
├── core/              # Core models and utilities
│   ├── models/        # Data models
│   └── utils/         # Utility functions
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── e2e/           # End-to-end tests
├── scripts/           # Utility scripts
└── config/            # Configuration files
```

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis (for caching)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arc.git
cd arc

# Install dependencies
pip install -e .

# Set up the database
python scripts/setup_db.py

# Seed initial data
python scripts/seed_data.py
```

### Configuration

Copy the example configuration and update with your settings:

```bash
cp config/example.env .env
```

### Running the Application

```bash
# Development mode
python -m arc

# Production mode
gunicorn arc.api:app
```

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# With coverage
pytest --cov=arc
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

Run all checks:
```bash
make lint
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- [Database Schema](../docs/DATABASE.md)
- [Integration Guide](../docs/INTEGRATION_GUIDE.md)
- [Product Requirements](../docs/PRD.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.