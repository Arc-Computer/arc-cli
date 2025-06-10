# Arc CLI

Proactive capability assurance for AI agents. Arc tests what your AI system should be capable of doing before it fails in production.

## Overview

Arc CLI is a command-line tool that helps ML engineers and AI product teams quantify and improve the reliability of their AI agents through systematic testing. Instead of waiting for production failures, Arc proactively identifies capability gaps and provides actionable recommendations to improve agent performance.

## Key Features

- **Framework-agnostic agent testing** - Works with any AI agent configuration (OpenAI, Anthropic, Google, Meta, etc.)
- **Automated scenario generation** - Creates diverse test cases using pattern-based and LLM-generated approaches
- **Multi-model optimization** - Tests agents across 9+ model providers to find the optimal reliability/cost tradeoff
- **Failure pattern clustering** - Automatically groups and analyzes failures to identify systemic issues
- **Actionable recommendations** - Generates specific configuration changes to address identified problems
- **Local-first architecture** - All data stays on your machine; optionally use Modal for distributed execution

## Installation

```bash
# Clone the repository
git clone https://github.com/arc-computer/arc-cli.git
cd arc-cli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Arc CLI
pip install -e .
```

## Quick Start

1. **Create an agent configuration file** (`agent.yaml`):

```yaml
model: gpt-4.1-mini
temperature: 0.7
system_prompt: |
  You are a helpful assistant that processes financial transactions.
  Always use USD for currency unless specified otherwise.
tools:
  - name: calculator
  - name: currency_converter
```

2. **Test your agent**:

```bash
# Validate configuration
arc validate agent.yaml

# Run reliability testing (50 scenarios by default)
arc run agent.yaml

# Analyze failures
arc analyze

# Get improvement recommendations
arc recommend

# Check execution history
arc status
```

## Core Workflow

Arc follows a 5-step continuous improvement loop:

1. **Import** → Parse agent configuration
2. **Simulate** → Execute scenarios in sandboxed environment
3. **Analyze** → Cluster failures and identify patterns
4. **Recommend** → Generate configuration improvements
5. **Improve** → Apply changes and re-test

## Advanced Usage

### A/B Testing

Compare two agent configurations:

```bash
arc diff agent_v1.yaml agent_v2.yaml --scenarios 100
```

### Pattern-Based vs LLM Generation

Control scenario generation strategy:

```bash
# 100% pattern-based scenarios
arc run agent.yaml --pattern-ratio 1.0

# 100% LLM-generated scenarios
arc run agent.yaml --pattern-ratio 0.0

# Default: 70% pattern, 30% LLM
arc run agent.yaml --pattern-ratio 0.7
```

### JSON Output for Automation

```bash
arc run agent.yaml --json > results.json
arc analyze --json | jq '.failure_clusters'
arc recommend --json | jq '.recommendations'
```

### Distributed Execution with Modal

For faster execution with Modal (optional):

```bash
# Set up Modal authentication
modal setup

# Arc automatically uses Modal when available
arc run agent.yaml --scenarios 500
```

## Architecture

Arc uses a modular architecture optimized for extensibility:

- **Ingestion Layer** - Framework-agnostic parsing and normalization
- **Scenario Engine** - Two-stage generation pipeline (pattern selection → instantiation)
- **Sandbox Environment** - Isolated execution with tool behavior simulation
- **Analysis Pipeline** - ML-based failure clustering and pattern recognition
- **Recommendation Engine** - Multi-model optimization and configuration diffing

## Configuration Examples

See the `examples/configs/` directory for sample configurations:

- `minimal_agent.yaml` - Basic agent setup
- `finance_agent_v1.yaml` - Financial domain with USD assumptions
- `finance_agent_v2.yaml` - Enhanced with multi-currency support
- `audit_agent.yaml` - Compliance-focused configuration

## Development

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Code quality checks
make lint

# Format code
make format
```

## Environment Variables

- `OPENROUTER_API_KEY` - For LLM-based scenario generation
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` - For distributed execution (optional)
- `TIMESCALE_SERVICE_URL` - For production database deployment (optional)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## Support

- Documentation: `docs/`
- Issues: [GitHub Issues](https://github.com/arc-computer/arc-cli/issues)
- Discussions: [GitHub Discussions](https://github.com/arc-computer/arc-cli/discussions)