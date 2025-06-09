# Arc-Eval GitHub Issues

## Phase 1: Database Foundation Issues

### Issue 1: Deploy PostgreSQL with pgvector extension
**Labels:** `database`, `infrastructure`, `phase-1`, `mvp`

Set up PostgreSQL instance with pgvector extension for embeddings support. Configure for 12.5x write performance optimization with JSONB and GIN indexes.

**Tasks:**
- Install PostgreSQL with pgvector extension
- Configure connection pooling
- Set up database credentials
- Test vector similarity search functionality

---

### Issue 2: Implement database schema from Database_arc.md
**Labels:** `database`, `phase-1`, `mvp`

Execute complete schema including all core tables: configurations, config_versions, scenarios, simulations, outcomes, failure_patterns, tool_usage, recommendations, audit_log.

**Tasks:**
- Create tables.sql with full schema
- Add JSONB columns with GIN indexes
- Configure pgvector columns for embeddings
- Set up row-level security for multi-tenancy

---

### Issue 3: Import 31 failure patterns as seed data
**Labels:** `database`, `data`, `phase-1`

Load the 31 failure patterns from experiments across 9 categories into failure_patterns table.

**Tasks:**
- Create failure pattern seed data from experiments
- Structure patterns with proper categorization
- Load into failure_patterns table
- Verify pattern embeddings generated correctly

---

### Issue 4: Create sample agent configurations
**Labels:** `database`, `data`, `phase-1`

Create 3 sample agent configs (weather, database, calculator) for demo and testing.

**Tasks:**
- Design weather agent YAML config
- Design database agent YAML config  
- Design calculator agent YAML config
- Store configs with proper versioning

---

### Issue 5: Port AsyncPostgreSQLStorage from experiments
**Labels:** `database`, `backend`, `phase-1`, `mvp`

Implement async PostgreSQL client with connection pooling, retry logic, and optimized query patterns.

**Tasks:**
- Port AsyncPostgreSQLStorage class
- Implement connection pooling with asyncpg
- Add retry logic for transient failures
- Create unit tests for storage operations

---

## Phase 2: Sandbox Environment Issues

### Issue 6: Port simulation engine from arc_eval_sandbox.py
**Labels:** `sandbox`, `backend`, `phase-2`, `mvp`

Implement core simulation logic with secure Docker-based execution environment.

**Tasks:**
- Port simulation orchestration logic
- Set up Docker container isolation
- Implement timeout and resource limits
- Add OpenTelemetry instrumentation

---

### Issue 7: Implement 4-level trajectory capture system
**Labels:** `sandbox`, `backend`, `phase-2`, `mvp`

Build trajectory capture for execution traces, decision points, behavioral patterns, and meta-cognitive signals.

**Tasks:**
- Implement TrajectoryCapture class
- Capture tool calls with timing (Level 1)
- Extract reasoning chains (Level 2)
- Identify strategy shifts (Level 3)
- Track uncertainty expressions (Level 4)

---

### Issue 8: Create realistic tool behavior profiles
**Labels:** `sandbox`, `tools`, `phase-2`

Implement weather, database, and calculator tool behaviors with realistic failure modes.

**Tasks:**
- Create weather API tool with rate limits
- Create database tool with connection failures
- Create calculator tool with precision limits
- Add configurable failure injection

---

### Issue 9: Implement 5-dimensional reliability scorer
**Labels:** `sandbox`, `evaluation`, `phase-2`, `mvp`

Build reliability scoring across tool execution, response quality, error handling, performance, and completeness dimensions.

**Tasks:**
- Create ReliabilityDimension enum with weights
- Implement scoring logic for each dimension
- Create composite score calculation
- Add scoring explanation generation

---

### Issue 10: Set up async execution pipeline
**Labels:** `sandbox`, `infrastructure`, `phase-2`

Build basic async execution without Modal for MVP, preparing for future scale.

**Tasks:**
- Implement asyncio-based execution pool
- Add concurrent simulation support
- Create execution monitoring
- Design Modal migration path

---

## Phase 3: Integration Issues

### Issue 11: Connect sandbox to PostgreSQL storage
**Labels:** `integration`, `backend`, `phase-3`, `mvp`

Wire simulation engine to persist trajectories, outcomes, and metrics to PostgreSQL.

**Tasks:**
- Integrate AsyncPostgreSQLStorage with simulator
- Implement trajectory serialization to JSONB
- Store outcomes with proper relationships
- Add transaction management

---

### Issue 12: Implement ML-powered failure clustering
**Labels:** `integration`, `ml`, `phase-3`

Build DBSCAN-based failure clustering with TF-IDF vectorization.

**Tasks:**
- Implement FailureClusterer class
- Add TF-IDF vectorization for failure texts
- Configure DBSCAN clustering parameters
- Generate human-readable cluster names

---

### Issue 13: Build configuration diff generator
**Labels:** `integration`, `recommendations`, `phase-3`, `mvp`

Create system to generate specific YAML configuration changes based on failure analysis.

**Tasks:**
- Implement ConfigurationDiff class
- Map failure clusters to config changes
- Calculate expected reliability impact
- Generate minimal reproduction scenarios

---

### Issue 14: Create CLI interface
**Labels:** `integration`, `cli`, `phase-3`, `mvp`

Build command-line interface for arc-eval with rich output support.

**Tasks:**
- Implement main CLI commands (run, analyze, recommend)
- Add rich text formatting for outputs
- Create progress indicators for long operations
- Support JSON output mode

---

### Issue 15: End-to-end integration testing
**Labels:** `integration`, `testing`, `phase-3`, `mvp`

Verify complete flow from agent import to recommendations.

**Tasks:**
- Create integration test suite
- Test full simulation pipeline
- Verify recommendation generation
- Validate reliability score improvements

---

## Phase 4: Demo Preparation Issues

### Issue 16: Create Wednesday demo script and materials
**Labels:** `demo`, `phase-4`

Prepare complete demo flow showing failing agent, analysis, recommendations, and improvement.

**Tasks:**
- Create weather_agent_v1.yaml with currency assumption bug
- Prepare demo commands sequence
- Generate expected outputs
- Create backup demo recordings

---

## Additional Setup Issues

### Issue 17: Set up development environment
**Labels:** `setup`, `infrastructure`

Configure local development environment with all dependencies.

**Tasks:**
- Document Python version requirements
- Create virtual environment setup script
- Configure IDE settings
- Set up pre-commit hooks

---

### Issue 18: Configure CI/CD pipeline
**Labels:** `setup`, `infrastructure`, `devops`

Set up automated testing and deployment pipeline.

**Tasks:**
- Configure GitHub Actions workflows
- Set up automated testing on PR
- Add code coverage reporting
- Create deployment scripts

---

### Issue 19: Implement monitoring and observability
**Labels:** `setup`, `monitoring`, `infrastructure`

Set up application monitoring with OpenTelemetry.

**Tasks:**
- Configure OpenTelemetry collectors
- Set up trace sampling
- Create performance dashboards
- Add error alerting

---

### Issue 20: Create API documentation
**Labels:** `documentation`, `api`

Generate OpenAPI/Swagger documentation for REST endpoints.

**Tasks:**
- Add OpenAPI annotations
- Configure Swagger UI
- Document authentication flow
- Create API usage examples

---

### Issue 21: Set up security foundations
**Labels:** `security`, `infrastructure`

Implement basic security measures for MVP.

**Tasks:**
- Configure environment variable management
- Implement basic API authentication
- Set up rate limiting
- Add input validation

---

### Issue 22: Prepare for Modal integration
**Labels:** `infrastructure`, `scale`

Design architecture for future Modal compute integration.

**Tasks:**
- Research Modal platform requirements
- Design job submission interface
- Plan migration strategy
- Estimate scaling requirements