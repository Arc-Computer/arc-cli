# Arc-Eval 48-Hour Sprint Issues (Modal-Powered Demo)

**Sprint Goal:** Build a working CLI demo that showcases Arc's proactive capability assurance using existing Modal infrastructure.

**Target Persona:** Applied ML Engineer / AI Product Engineer (comfortable with CLI, not deep ML evaluation experts)
**Core Value Prop:** Time-to-insight < 5 minutes, +20pp reliability uplift, actionable code-level recommendations

**Database Note:** Database architecture is being handled by teammate - focus on CLI, Modal integration, and demo preparation.

## Day 1: Core CLI + Modal Integration (12 hours)

### Issue 1: Copy and adapt core experimental components to production
**Labels:** `migration`, `day-1`, `priority-1`, `mvp`

Copy essential experimental components to production codebase and remove experiments/ dependency for public release.

**Acceptance Criteria:**
- [ ] Copy `experiments/src/core/arc_eval_sandbox.py` → `arc/simulation/modal_engine.py`
- [ ] Copy `experiments/generation/enhanced_generator.py` → `arc/scenarios/enhanced_generator.py`
- [ ] Copy `experiments/generation/scenario_quality_scorer.py` → `arc/scenarios/quality_scorer.py`
- [ ] Copy `experiments/generation/pattern_based_generator.py` → `arc/scenarios/pattern_generator.py`
- [ ] Copy `experiments/src/evaluation/reliability_scorer.py` → `arc/evaluation/scorer.py`
- [ ] Copy `experiments/src/analysis/failure_clustering.py` → `arc/analysis/clustering.py`
- [ ] Copy `experiments/src/tracing/trajectory_capture.py` → `arc/tracing/capture.py`
- [ ] Copy `experiments/generation/failure_patterns/` → `arc/scenarios/failure_patterns/`
- [ ] Copy `experiments/generation/tool_behavior_profiles.json` → `arc/simulation/tool_profiles.json`
- [ ] Copy `experiments/config/agent_config*.yaml` → `configs/demo_agent_*.yaml`
- [ ] Update all imports to remove experiments/ references
- [ ] Maintain Modal integration and parallel execution capabilities
- [ ] Test core functionality works with production imports

**Tasks:**
- Create production directory structure: `arc/simulation/`, `arc/evaluation/`, `arc/analysis/`, `arc/scenarios/`, `arc/tracing/`
- Copy 31+ failure patterns from experiments/generation/failure_patterns/
- Copy tool behavior profiles for realistic simulation
- Update all internal imports to production paths
- Remove experimental/internal references and comments
- Test Modal integration still works with production code
- Update package dependencies in pyproject.toml

**Dependencies:** None
**Files to create:** `arc/simulation/modal_engine.py`, `arc/scenarios/enhanced_generator.py`, `arc/scenarios/quality_scorer.py`, `arc/scenarios/pattern_generator.py`, `arc/evaluation/scorer.py`, `arc/analysis/clustering.py`, `arc/tracing/capture.py`, `arc/scenarios/failure_patterns/`, `arc/simulation/tool_profiles.json`

---

### Issue 2: Design developer-first CLI
**Labels:** `cli`, `ux`, `day-1`, `priority-1`, `mvp`

Create a CLI that follows the continuous improvement workflow from PRD with developer-first experience targeting Applied ML Engineers.

**Acceptance Criteria:**
- [ ] CLI follows the 5-step workflow: Import → Simulate → Analyze → Recommend → Improve
- [ ] Commands: `arc run`, `arc analyze`, `arc recommend`, `arc diff`
- [ ] Rich text output with progress bars, tables, and color coding using `rich` library
- [ ] JSON output mode for programmatic use (`--output json`)
- [ ] Framework-agnostic agent profile ingestion (YAML/JSON)
- [ ] Time-to-insight < 5 minutes target (fast execution + clear output)
- [ ] Error handling with helpful messages for non-ML experts
- [ ] Cost transparency (show estimated and actual costs)
- [ ] Progress indicators for Modal parallel execution

**CLI Design Pattern (following tools like `kubectl`, `gh`, `docker`):**
```bash
# Core workflow commands (PRD 5-step process)
arc run <agent.yaml> [--scenarios N] [--output json|text] [--save-run ID]
arc analyze [--run-id ID] [--last] [--format table|json]
arc recommend [--run-id ID] [--cluster ID] [--apply]
arc diff <agent1.yaml> <agent2.yaml> [--expected-impact]

# Utility commands
arc status [--runs] [--costs]
arc validate <agent.yaml>
arc --version
```

**Output Format Requirements:**
- Reliability scores with visual progress bars
- Failure clustering with clear categorization
- Cost breakdown (estimated vs actual)
- Execution time and parallel speedup metrics
- Actionable recommendations with confidence scores

**Tasks:**
- Design CLI command structure following open source best practices
- Implement rich text formatting with tables, progress bars, colors
- Add real-time progress indicators for Modal execution
- Create JSON output mode for API integration
- Add comprehensive help text and examples
- Test CLI usability with Applied ML Engineer persona
- Implement cost transparency and estimation

**Dependencies:** Issue 1
**Files to modify:** `arc/cli.py`, create `arc/cli/commands/`

---

---

### Issue 3: Implement Modal orchestration with cost transparency
**Labels:** `modal`, `integration`, `day-1`, `priority-1`, `mvp`

Build Modal orchestration layer that provides cost transparency and execution monitoring for budget-conscious Applied ML Engineers.

**Acceptance Criteria:**
- [ ] `ModalOrchestrator` class that wraps production Modal functions from Issue 1
- [ ] Cost estimation BEFORE execution (critical for budget-conscious engineers)
- [ ] Real-time progress monitoring with container scaling visibility (50 containers max)
- [ ] Execution time tracking and parallel speedup metrics (target: 50 scenarios in ~45 seconds)
- [ ] Error handling with clear explanations for non-Modal experts
- [ ] Integration with CLI progress indicators and rich text output
- [ ] Support for existing Modal configuration (max_containers=50, buffer_containers=5)

**Cost Transparency Requirements (from PRD):**
- Show estimated cost before execution (~$0.05 per 50 scenarios)
- Track actual cost during execution with real-time updates
- Display cost per scenario and total cost breakdown
- Show parallel speedup vs sequential execution (target: 20x+ speedup)
- Provide cost optimization recommendations

**Modal Integration Requirements:**
- Leverage existing `arc_eval_sandbox.py` Modal functions
- Maintain parallel execution with `@modal.concurrent` decorators
- Support async execution with `run_evaluation_suite_parallel.remote.aio()`
- Handle Modal scaling configuration and warm containers
- Integrate with existing OpenTelemetry tracing

**Tasks:**
- Create `arc/simulation/modal_orchestrator.py` with cost-aware interface
- Implement pre-execution cost estimation based on scenario count and model pricing
- Add real-time progress monitoring with rich progress bars showing container scaling
- Track execution metrics (time, cost, speedup, container utilization)
- Add error handling with user-friendly messages for Modal API issues
- Test with production Modal functions from Issue 1
- Integrate with CLI for seamless user experience

**Dependencies:** Issue 1
**Files to create:** `arc/simulation/modal_orchestrator.py`

---

### Issue 4: Create demo agent configurations targeting currency assumption failures
**Labels:** `config`, `demo`, `day-1`, `priority-1`, `mvp`

Create realistic finance agent configurations that demonstrate currency assumption failures for compelling enterprise demos.

**Acceptance Criteria:**
- [ ] `configs/finance_agent_v1.yaml` - baseline with currency assumption bug (realistic for enterprise finance teams)
- [ ] `configs/finance_agent_v2.yaml` - fixed version with multi-currency handling protocol
- [ ] `configs/audit_agent.yaml` - different finance domain for variety (audit/compliance use case)
- [ ] Configurations follow framework-agnostic profile format from PRD
- [ ] Clear documentation of assumptions being tested
- [ ] Compatible with production Modal simulation engine
- [ ] Realistic enterprise finance use cases (inspired by LlamaIndex spreadsheet agent examples)

**Configuration Design (Enterprise Finance Focus):**
```yaml
# finance_agent_v1.yaml (Baseline - Currency Assumption Bug)
job: "Process financial data and generate reports for corporate finance teams"
assumptions:
  - "All financial amounts are in USD unless explicitly stated"  # ← This assumption will be violated
  - "Spreadsheet data follows standard accounting formats"
  - "Users provide complete transaction details"
  - "All dates are in MM/DD/YYYY format"
system_prompt: |
  You are a corporate finance assistant that processes financial data, analyzes spreadsheets, and generates reports.

  When processing financial amounts:
  - Assume all monetary values are in USD unless currency is explicitly specified
  - Use standard US accounting formats for calculations
  # ↑ This is the bug we'll demonstrate fixing - causes failures with EUR, GBP, JPY data

  Available tools:
  1. spreadsheet_analyzer - Parse and analyze Excel/CSV financial data
  2. financial_calculator - Perform financial calculations and conversions
  3. database_query - Query financial databases for historical data
```

**Enterprise Use Cases:**
- **Corporate Finance**: Budget consolidations, quarter-end close, cash flow forecasting
- **Tax Teams**: Income tax provisions, ERP extracts, mapping tax lines
- **Audit Firms**: Processing trial-balance and general-ledger files
- **Insurance**: Bordereaux ingestion, loss & premium processing

**Tasks:**
- Design finance agent v1 with currency assumption flaw affecting multi-currency financial data
- Design finance agent v2 with proper multi-currency handling and validation
- Create audit agent for compliance/regulatory use case variety
- Document specific financial assumptions being tested
- Test configurations with production Modal functions
- Ensure compatibility with framework-agnostic profile format
- Create realistic financial scenarios that trigger currency failures

**Dependencies:** Issue 1
**Files to create:** `configs/finance_agent_v1.yaml`, `configs/finance_agent_v2.yaml`, `configs/audit_agent.yaml`

---

---

### Issue 5: Build assumption-based scenario generation system
**Labels:** `scenarios`, `generation`, `day-1`, `priority-1`, `mvp`
**Assignee:** Claude Agent
**Estimate:** 3 hours

Implement the assumption-based scenario generation from V1 strategy to systematically test agent capabilities.

**Acceptance Criteria:**
- [ ] Assumption extraction from agent configurations
- [ ] Systematic violation scenario generation (matches V1 strategy approach)
- [ ] 50 scenarios total: 15 currency assumption violations + 35 general capability tests
- [ ] Scenario quality scoring and filtering
- [ ] Integration with production Modal simulation engine
- [ ] Reproducible scenario sets for consistent demo results

**Scenario Generation Pattern (from V1 Strategy):**
```python
# For assumption: "Budget includes all travel costs"
scenarios = [
    "Plan a trip for €2000",                    # Currency ambiguity
    "Plan a trip for $2000 excluding flights",  # Scope ambiguity
    "Plan a trip for 2000",                     # Currency missing
    "Plan a trip for $2000 including visa fees" # Scope expansion
]
```

**Tasks:**
- Adapt `experiments/generation/enhanced_generator.py` to production
- Implement assumption extraction from agent configs
- Create systematic violation scenario generation
- Add scenario quality scoring and filtering
- Generate demo-specific scenario sets
- Test integration with Modal simulation engine

**Dependencies:** Issue 1
**Files to modify:** `arc/scenarios/generator.py`

---

### Issue 6: Implement 5-dimensional reliability scoring with explanations
**Labels:** `evaluation`, `scoring`, `day-1`, `priority-1`, `mvp`
**Assignee:** Claude Agent
**Estimate:** 2 hours

Adapt the 5-dimensional reliability scorer to provide clear explanations for non-ML experts.

**Acceptance Criteria:**
- [ ] 5-dimensional scoring: Tool Execution (30%), Response Quality (25%), Error Handling (20%), Performance (15%), Completeness (10%)
- [ ] Clear explanations for each dimension score
- [ ] Composite reliability score calculation
- [ ] Score breakdown visualization for CLI output
- [ ] Integration with results formatting system
- [ ] Scoring explanations accessible to Applied ML Engineers (not deep ML experts)

**Scoring Output Format:**
```bash
┌─────────────────────────────────────────────┐
│ Reliability Score: 73% (36/50 scenarios)    │
├─────────────────────────────────────────────┤
│ Tool Execution        ████████░░ 8/10 (30%) │
│ Response Quality      ██████░░░░ 6/10 (25%) │
│ Error Handling        ████░░░░░░ 4/10 (20%) │
│ Performance           ███████░░░ 7/10 (15%) │
│ Completeness          █████████░ 9/10 (10%) │
└─────────────────────────────────────────────┘
```

**Tasks:**
- Adapt `experiments/src/evaluation/reliability_scorer.py` to production
- Add clear explanations for each scoring dimension
- Create CLI-friendly score visualization
- Implement composite score calculation with weights
- Add scoring rationale for non-experts
- Test with demo scenarios

**Dependencies:** Issue 1
**Files to modify:** `arc/evaluation/scorer.py`

---

---

## Day 2: Demo Polish + Advanced Features (6 hours)

### Issue 7: Build results processing and presentation system
**Labels:** `results`, `presentation`, `day-2`, `priority-1`, `mvp`
**Assignee:** Claude Agent
**Estimate:** 2 hours

Create results processing system that transforms Modal output into actionable insights for Applied ML Engineers.

**Acceptance Criteria:**
- [ ] `ResultsProcessor` class that transforms Modal execution results
- [ ] Rich text formatting with tables, progress bars, and color coding
- [ ] Clear failure categorization and clustering
- [ ] Performance metrics display (execution time, cost, parallel speedup)
- [ ] JSON export mode for programmatic integration
- [ ] Results presentation optimized for time-to-insight < 5 minutes

**Results Display Format (following PRD requirements):**
```
🚀 Arc-Eval Results Summary
├── Overall Reliability: 73% (36/50 scenarios passed)
├── Execution Time: 45 seconds (23.4x speedup vs sequential)
├── Total Cost: $0.0234
└── Key Issues Found: 3 failure clusters

📊 Reliability Breakdown:
[Detailed 5-dimensional scoring visualization]

🔍 Failure Analysis:
Cluster 1: "Currency Assumption Violations" (15 failures)
├── Root Cause: Agent assumes USD without confirmation
├── Impact: 30% reliability loss
└── Affected Scenarios: "Plan trip for €2000", "Tokyo for 200000"
```

**Tasks:**
- Create `arc/results/processor.py` with Modal results transformation
- Implement rich text formatting with `rich` library
- Add failure clustering and categorization
- Create performance metrics display
- Add JSON export capabilities
- Test with demo scenarios and Modal output

**Dependencies:** Issues 3, 6
**Files to create:** `arc/results/processor.py`, `arc/results/__init__.py`

---

### Issue 8: Implement configuration diff generator with impact prediction
**Labels:** `recommendations`, `diff`, `day-2`, `priority-1`, `mvp`
**Assignee:** Claude Agent
**Estimate:** 2 hours

Build the configuration diff system that generates actionable code-level recommendations (core value prop from PRD).

**Acceptance Criteria:**
- [ ] `ConfigurationDiff` class that generates specific YAML changes
- [ ] Mapping from failure clusters to configuration fixes
- [ ] Expected reliability impact calculations (target: +20pp improvement)
- [ ] Before/after configuration comparison
- [ ] Integration with CLI `recommend` command
- [ ] Recommendations must be actionable for non-ML experts

**Configuration Diff Output (Finance Agent Example):**
```diff
# Generated Configuration Diff
finance_agent_v1.yaml
@@ -12,8 +12,15 @@ system_prompt: |

   When processing financial amounts:
-  - Assume all monetary values are in USD unless currency is explicitly specified
-  - Use standard US accounting formats for calculations
+  - Always validate currency before processing monetary values
+  - Support multi-currency transactions with proper conversion rates
+  - Validate currency symbols and ISO codes (USD, EUR, GBP, JPY, etc.)
+
+  Multi-Currency Protocol:
+  1. Parse amount and detect currency indicators (€, £, ¥, $)
+  2. If currency unclear, ask: "What currency is this amount in?"
+  3. Apply appropriate formatting and conversion rules
+  4. Validate against supported currency list before processing

Expected Impact: +18% reliability improvement (73% → 91%)
Confidence: 94% (based on similar financial data processing patterns)
```

**Tasks:**
- Create `arc/recommendations/diff_generator.py`
- Implement failure cluster → configuration mapping
- Add YAML diff generation with before/after comparison
- Calculate expected reliability impact
- Add confidence scoring for recommendations
- Test with demo agent configurations

**Dependencies:** Issue 7
**Files to create:** `arc/recommendations/diff_generator.py`, `arc/recommendations/__init__.py`

---

### Issue 9: Create comprehensive demo script and validation system
**Labels:** `demo`, `validation`, `day-2`, `priority-1`, `mvp`
**Assignee:** Claude Agent
**Estimate:** 2 hours

Build complete demo flow that showcases the continuous improvement workflow from PRD.

**Acceptance Criteria:**
- [ ] Complete 6-minute demo script following the 5-step workflow
- [ ] Demo validation system to ensure consistent results
- [ ] Error handling and fallback procedures
- [ ] Customer-facing documentation and technical explanations
- [ ] Backup systems for live demo reliability

**Demo Script Structure (following PRD workflow):**
```bash
# 1. Agent Profile Ingestion (30 seconds)
arc run configs/finance_agent_v1.yaml --scenarios 50

# 2. Simulation Execution (45 seconds - Modal parallel execution)
[Real-time progress with container scaling visualization]

# 3. Automated Error Analysis (30 seconds)
arc analyze --last --format table

# 4. Reliability Analysis & Recommendations (2 minutes)
arc recommend --cluster currency_assumption

# 5. Iterative Improvement (2 minutes)
arc run configs/finance_agent_v2.yaml --scenarios 50
arc diff finance_agent_v1.yaml finance_agent_v2.yaml --expected-impact
```

**Tasks:**
- Create detailed demo script with exact timing
- Build demo validation system to ensure consistent results
- Add error handling and fallback procedures
- Prepare customer FAQ and technical explanations
- Create backup recording and manual fallback options
- Test complete demo flow multiple times

**Dependencies:** Issues #3-#10
**Files to create:** `demo/script.md`, `demo/validation.py`, `demo/README.md`

---

### Issue 15: Implement capability-centric architecture and assumption testing
**Labels:** `capability`, `architecture`, `day-2`, `priority-1`, `mvp`

Implement the core capability-centric approach that differentiates Arc from reactive debugging tools.

**Acceptance Criteria:**
- [ ] `CapabilityProfiler` class that extracts agent capabilities from configuration
- [ ] Assumption extraction and validation system
- [ ] Assumption-violation scenario generation (not just pattern-based)
- [ ] Capability coverage scoring and metrics
- [ ] Failure attribution to specific capability assumptions
- [ ] Integration with existing scenario generation system
- [ ] Capability-level recommendations (not just generic fixes)

**Core Value Proposition:**
This implements Arc's fundamental differentiation: **testing what agents are hired to do** rather than just checking if tools work.

**Tasks:**
- Create `arc/capabilities/profiler.py` with capability extraction
- Implement assumption-based scenario generation in `arc/scenarios/assumption_generator.py`
- Add capability coverage scoring to evaluation system
- Integrate with existing failure analysis for root cause attribution
- Create capability-level recommendation mapping
- Test with finance agent configurations to validate assumption violations

**Dependencies:** Issues #6, #7
**Files to create:** `arc/capabilities/profiler.py`, `arc/scenarios/assumption_generator.py`

---

### Issue 17: Implement TimescaleDB client integration
**Labels:** `database`, `integration`, `day-1`, `priority-1`, `mvp`

Implement the database client that connects our Modal execution results to TimescaleDB for persistence and real-time CLI updates.

**Acceptance Criteria:**
- [ ] Complete `ArcEvalDBClient` class with async connection pooling
- [ ] `record_execution_outcome()` method for storing Modal results
- [ ] `record_simulation()` method for tracking complete runs
- [ ] `record_failure_pattern()` method for failure analysis
- [ ] `get_simulation_status()` method for real-time CLI progress updates
- [ ] `get_historical_trends()` method for CLI analytics display
- [ ] Integration with existing Modal orchestrator
- [ ] TimescaleDB hypertable optimization for outcomes
- [ ] Connection to TimescaleDB Cloud instance
- [ ] **Real-time data capabilities for CLI interface updates**

**CLI Real-Time Integration:**
Even with CLI interface, we need real-time capabilities for:
- **Live progress updates**: `arc run` shows real-time execution progress
- **Historical comparisons**: `arc analyze` shows trends over time
- **Configuration evolution**: `arc diff` shows improvement history
- **Cost tracking**: Real-time cost updates during execution

**Tasks:**
- Complete `arc/database/client.py` with production-ready implementation
- Implement async connection pooling with asyncpg
- Add TimescaleDB-specific optimizations (hypertables, compression)
- Create real-time query methods for CLI interface
- Add error handling and retry logic for database operations
- Test with TimescaleDB Cloud instance
- **Integrate with CLI commands for real-time data display**

**Dependencies:** Teammate's TimescaleDB Cloud setup
**Files to modify:** `arc/database/client.py`, `arc/database/schema/tables.sql`

---

### Issue 18: Connect Modal orchestrator to TimescaleDB
**Labels:** `modal`, `database`, `integration`, `day-1`, `priority-1`, `mvp`

Connect the Modal execution pipeline to persist results in TimescaleDB and provide real-time CLI updates.

**Acceptance Criteria:**
- [ ] Modal execution results automatically stored in TimescaleDB
- [ ] Simulation tracking with real-time status updates for CLI
- [ ] Trajectory data stored with embeddings for similarity search
- [ ] Tool usage analytics captured for CLI analytics
- [ ] Cost tracking integrated with database
- [ ] Error handling for database connection failures
- [ ] **Real-time CLI progress updates during Modal execution**
- [ ] **Historical data integration for CLI trend analysis**

**Integration Flow:**
```
CLI Command → Modal Orchestrator → Parallel Execution → TimescaleDB → CLI Display
     ↓              ↓                    ↓               ↓           ↓
arc run        Start simulation     Execute scenarios   Store results  Show progress
arc analyze    Query database       N/A                Get trends     Display analytics
arc recommend  Query patterns       N/A                Get history    Show recommendations
```

**Tasks:**
- Enhance `arc/simulation/modal_orchestrator.py` with database integration
- Add real-time status tracking during Modal execution
- Implement trajectory storage with vector embeddings
- Add cost tracking integration with TimescaleDB
- **Integrate real-time progress updates with CLI interface**
- **Add historical data queries for CLI analytics**
- Test end-to-end Modal → TimescaleDB → CLI flow

**Dependencies:** Issue #17 (Database Client), Issue #5 (Modal Orchestrator)
**Files to modify:** `arc/simulation/modal_orchestrator.py`, `arc/cli.py`

---

### Issue 19: Implement configuration versioning and evolution tracking
**Labels:** `config`, `versioning`, `day-2`, `priority-2`, `enhancement`

Implement configuration versioning system that tracks agent evolution and enables historical analysis for CLI commands.

**Acceptance Criteria:**
- [ ] `ConfigVersionManager` class for tracking configuration changes
- [ ] Automatic versioning when configurations are modified
- [ ] Configuration diff tracking in TimescaleDB
- [ ] Historical evolution analysis for CLI display
- [ ] Integration with `arc diff` command for impact analysis
- [ ] Configuration similarity search using embeddings
- [ ] **CLI historical context**: "This config improved 23% over last week"

**CLI Integration Examples:**
```bash
# Enhanced arc run with historical context
$ arc run finance_agent_v2.yaml
📊 Historical Context:
├── Previous reliability: 73% → Expected: 91% (+18%)
├── Improvement trend: +2.3% per day over last week
└── Similar configs achieved: 89% average reliability

# Enhanced arc diff with impact prediction
$ arc diff finance_agent_v1.yaml finance_agent_v2.yaml
📈 Predicted Impact (based on similar changes):
├── Expected reliability improvement: +18%
├── Confidence: 94% (based on 23 similar configurations)
└── Historical success rate: 89% of similar changes succeeded
```

**Tasks:**
- Create `arc/config/version_manager.py` with configuration tracking
- Implement configuration embedding generation for similarity search
- Add automatic versioning when configurations change
- **Integrate with CLI commands for historical context display**
- **Add configuration evolution queries to database client**
- Implement similarity search for impact prediction

**Dependencies:** Issue #17 (Database Client), Issue #4 (CLI)
**Files to create:** `arc/config/version_manager.py`, `arc/config/similarity.py`

---

## Supporting Issues (If Time Permits)

### Issue 12: Implement advanced failure clustering with ML
**Labels:** `analysis`, `ml`, `day-2`, `priority-3`, `nice-to-have`

Enhance failure analysis with ML-powered clustering from experimental implementation.

**Acceptance Criteria:**
- [ ] TF-IDF vectorization for failure text analysis
- [ ] DBSCAN clustering for similar failure patterns
- [ ] Human-readable cluster naming
- [ ] Root cause attribution with confidence scoring
- [ ] Integration with experimental failure patterns
- [ ] Pattern recognition for currency assumptions and other common failures

**Tasks:**
- Adapt `experiments/src/analysis/failure_clustering.py` to production
- Implement TF-IDF vectorization for failure texts
- Add DBSCAN clustering with optimal parameters
- Create human-readable cluster names and descriptions
- Add root cause attribution logic
- Test with demo scenarios and validate clustering quality

**Dependencies:** Issue #9
**Files to modify:** `arc/analysis/clustering.py`

---

### Issue 13: Add real-time execution monitoring and cost tracking
**Labels:** `monitoring`, `ux`, `day-2`, `priority-3`, `nice-to-have`

Enhance the demo experience with real-time monitoring that showcases Modal's parallel execution capabilities.

**Acceptance Criteria:**
- [ ] Real-time progress bars showing Modal container scaling
- [ ] Live cost tracking during execution
- [ ] Execution timeline with performance metrics
- [ ] Container utilization visualization
- [ ] Error monitoring and recovery procedures
- [ ] Integration with CLI rich text output

**Real-time Display Format:**
```bash
🚀 Executing 50 scenarios via Modal...
├── Containers: ████████████████████ 25/50 active
├── Progress:   ████████████░░░░░░░░ 32/50 complete (64%)
├── Cost:       $0.0156 (estimated: $0.0234)
├── Time:       00:32 (estimated: 00:45)
└── Failures:   3 detected, clustering in progress...
```

**Tasks:**
- Enhance Modal orchestrator with real-time monitoring
- Add progress visualization with rich library
- Implement live cost tracking and estimation
- Create container scaling visualization
- Add execution timeline and performance metrics
- Test real-time updates during Modal execution

**Dependencies:** Issue #5
**Files to modify:** `arc/simulation/modal_orchestrator.py`, `arc/cli.py`

---

## Sprint Success Criteria

### Technical Success Metrics (Must Achieve)
- [ ] CLI executes complete 5-step workflow in under 6 minutes (PRD requirement: time-to-insight < 5 min)
- [ ] Modal integration scales to 50 scenarios in ~45 seconds with parallel execution
- [ ] Reliability improvement demo shows 73% → 91% (+18pp, meets PRD target of +20pp)
- [ ] Currency assumption failures reduced from 15 → 0 (demonstrates proactive capability assurance)
- [ ] All CLI commands work without errors and provide rich text output
- [ ] Cost transparency: show estimated and actual costs (~$0.05 per evaluation)
- [ ] Framework-agnostic agent profile ingestion (YAML/JSON)
- [ ] **TimescaleDB integration**: All results persisted with real-time CLI updates
- [ ] **Historical context**: CLI shows configuration evolution and trends
- [ ] **Database connectivity**: Successful connection to TimescaleDB Cloud instance

### Business Success Metrics (Customer Demo Goals)
- [ ] Clear value proposition: proactive vs reactive reliability testing
- [ ] Demonstrates continuous improvement workflow from PRD
- [ ] Shows actionable, code-level recommendations (core differentiator)
- [ ] Showcases Modal parallel execution scalability
- [ ] Cost efficiency and transparency for budget-conscious engineers
- [ ] Developer-first experience accessible to Applied ML Engineers

### Demo Readiness Checklist
- [ ] All 12 core issues (Issues #3-#11, #15, #17-#18) completed and tested
- [ ] Demo script rehearsed 3+ times with consistent results
- [ ] Backup recording and manual fallback procedures ready
- [ ] Customer FAQ covering technical questions prepared
- [ ] Modal infrastructure tested and validated in demo environment
- [ ] Error handling tested with graceful failure recovery
- [ ] Production codebase clean (no experiments/ dependencies)

### Target Persona Validation
- [ ] CLI accessible to Applied ML Engineers (not deep ML experts)
- [ ] Time-to-insight < 5 minutes achieved
- [ ] Actionable recommendations that engineers will trust to modify production code
- [ ] Cost transparency and optimization recommendations
- [ ] Framework-agnostic approach (no vendor lock-in)

---

## Post-Sprint Scaling Path

### Week 1: Production Hardening
- Database integration with teammate's PostgreSQL implementation
- Enhanced error handling and monitoring
- Multi-model testing across 9+ providers via OpenRouter
- Advanced ML-powered failure clustering
- REST API endpoints for programmatic integration

### Month 1: Enterprise Features (PRD Phase 2)
- Web dashboard for results visualization
- Multi-agent system support
- Industry-specific capability models
- Governance & compliance features (signed reliability attestations)
- Advanced configuration optimization engine

### Month 3: Platform Vision (Long-term Defensibility)
- Configuration→Outcome graph for predictive modeling
- Network effects through shared failure patterns
- Reinforcement learning optimization
- Enterprise GRC workflow integration
- API ecosystem for third-party integrations