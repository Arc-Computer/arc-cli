## Executive Summary

Arc will open source 90% of its functionality to become the de facto standard for AI agent testing. We monetize through speed (1000x faster), scale (unlimited parallelism), and enterprise features (compliance, team collaboration).

## Technical Architecture Split

### Open Source Components

#### 1. **Full Testing & Simulation Engine**
**From**: `arc/core/`, `arc/sandbox/`, `arc/evaluation/`
**Includes**:
- Complete simulation engine with Dagger container orchestration
- All evaluation metrics and scoring algorithms
- Tool behavior simulation and mocking
- Trajectory capture and replay (local only)
- Multi-agent coordination testing
- Parallel local execution (10-20 containers)

#### 2. **Complete CLI with Local Web UI**
**From**: `arc/cli/`, new local UI
**Includes**:
- All CLI commands with full functionality
- Local web dashboard (localhost:8080) with:
  - Real-time test execution view
  - Failure analysis dashboards
  - Recommendation explorer
  - Test history (local SQLite/DuckDB)
  - Export to PDF/HTML reports
- Rich TUI with progress tracking
- Watch mode for continuous testing

#### 3. **All Scenarios & Generation**
**From**: `arc/scenarios/`, `arc/database/seed/failure_patterns/`
**Includes**:
- 30+ failure patterns across all categories
- LLM-powered scenario generation (BYOK)
- Scenario mutation and fuzzing
- Custom scenario DSL
- Scenario quality scoring
- Version control for scenarios
- Community scenario hub integration

#### 4. **Advanced Analysis & ML**
**From**: `arc/analysis/`, `arc/recommendations/`
**Includes**:
- `arc/analysis/clustering.py` - Advanced clustering with UMAP/HDBSCAN
- `arc/analysis/assumption_detector.py` - Full assumption analysis
- `arc/analysis/funnel_analyzer.py` - Failure funnel visualization
- `arc/recommendations/failure_analyzer.py` - ML-powered recommendations
- `arc/recommendations/diff_engine.py` - Configuration optimization
- Embedded ML models for offline analysis
- Root cause analysis with explanation

#### 5. **Complete Model Support**
**New**: `arc/adapters/` (right now we are using openrouter but likely need adapters for other providers)
**Includes**:
- All providers: OpenAI, Anthropic, Google, AWS, Azure, Groq, Together, Replicate, Ollama
- Advanced features:
  - Automatic failover and retry
  - Cost tracking and budgeting
  - Rate limit management
  - Response caching (local)
  - Model performance profiling
  - A/B testing framework

#### 6. **Developer Tools**
**New**: `arc/tools/`
**Includes**:
- CI/CD integrations (GitHub Actions, GitLab, CircleCI)

#### 7. **Local Infrastructure**
**New**: Using Dagger/Arrakis
**Includes**:
- Container-based parallel execution
- Local caching layer
- Resource management
- Cleanup and isolation
- Mock service orchestration

### Proprietary Components (Arc Cloud)

#### 1. **Scale & Speed**
- **1000x Faster**: GPU-accelerated distributed execution
- **Unlimited Parallelism**: 10,000+ simultaneous scenarios
- **Global Execution**: Run from 50+ edge locations
- **Smart Scheduling**: ML-based test prioritization
- **Incremental Testing**: Only run what changed

#### 2. **Team & Enterprise**
- **Collaboration**:
  - Shared dashboards and reports
  - Team workspaces
  - Comments and annotations
  - Change approvals workflow
- **SSO/SAML**: Enterprise authentication
- **RBAC**: Fine-grained permissions
- **Audit Logs**: Complete history for compliance

#### 3. **Compliance & Security**
- **SOC2 Type II**: Certified infrastructure
- **HIPAA Mode**: PHI-safe testing environment
- **Data Residency**: Region-specific deployment
- **Signed Attestations**: Legal reliability certificates
- **SBOM Generation**: Supply chain tracking
- **PII Redaction**: Automatic sensitive data handling

#### 4. **Operational Intelligence**
- **Cross-Org Insights**: Anonymized benchmarking
- **Trend Analysis**: Emerging failure patterns
- **Model Intelligence**: "GPT-4.1 fails 23% more on X"
- **Cost Optimization**: "Save $10k/mo by switching models"
- **Proactive Alerts**: "New failure pattern detected"

#### 5. **Integrations & Support**
- **Native Integrations**: JIRA, Linear, Slack, Teams, PagerDuty
- **Monitoring**: Datadog, New Relic, Prometheus export
- **Custom Webhooks**: Enterprise system integration
- **Priority Support**: 1-hour SLA
- **Professional Services**: Custom scenario development
- **Quarterly Reviews**: Strategic guidance

## Why This Split Works

### For Developers (OSS)
- **Complete Tool**: Can test any AI agent thoroughly
- **No Lock-in**: All core functionality available
- **Extensible**: Plugin architecture for customization
- **Fast Start**: <5 minutes to first test

### For Teams (Cloud)
- **Speed**: 1000x faster execution saves days of waiting
- **Collaboration**: Share results and insights
- **Zero Ops**: No infrastructure to manage
- **Scale**: Handle massive test suites

### For Enterprises (Cloud)
- **Compliance**: Meet regulatory requirements
- **Security**: Isolated, audited infrastructure
- **Support**: Get help when needed
- **Intelligence**: Learn from anonymized peer data