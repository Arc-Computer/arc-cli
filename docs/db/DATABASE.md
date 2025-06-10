# Arc-Eval Database Architecture (TIMESCALEDB)

**🚀 TIMESCALEDB OPTIMIZED:** Enhanced for time-series workloads, modal sandbox integration, and managed cloud deployment.

Based on the Figma diagram flow: **Modal Sandbox → Execution → Outcomes → Database → LLM Recommendations → Dashboard**, this architecture is optimized for TimescaleDB's time-series capabilities and managed cloud features.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL SANDBOX ENVIRONMENT                         │
├─────────────────┬───────────────────┬───────────────────┬─────────────────┤
│   User + Yard   │    Simulators     │    Executions     │   Outcomes      │
│   Environment   │                   │                   │  (Pass/Fail)    │
└─────────────────┴───────────────────┴───────────────────┴─────────────────┘
         │                    │                    │                 │
         ▼                    ▼                    ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TIMESCALEDB DATABASE                                 │
├─────────────────┬───────────────────┬───────────────────┬─────────────────┤
│ configurations  │    simulations    │     outcomes      │ failure_patterns│
│ config_versions │                   │   (hypertable)    │                 │
│ config_diffs    │   simulations_    │                   │ recommendations │
│ scenarios       │   scenarios       │                   │                 │
└─────────────────┴───────────────────┴───────────────────┴─────────────────┘
         │                                        │                 │
         ▼                                        ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLM RECOMMENDATIONS ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Based on historical outcomes data → Generate config recommendations        │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DASHBOARD                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Show recommendations to users → Config v4 suggestions                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### TimescaleDB Features

| **Feature** | **Why Essential** | **TimescaleDB Advantage** |
|-------------|-------------------|---------------------------|
| **Time-Series Outcomes** | Track execution results over time | Automatic partitioning, compression |
| **Modal Integration** | Seamless sandbox execution tracking | High-throughput ingestion |
| **LLM Recommendations** | Historical pattern analysis | Fast time-based aggregations |
| **Dashboard Analytics** | Real-time performance insights | Continuous aggregates |
| **Managed Security** | Production-ready without ops overhead | Built-in encryption, backups |
| **Auto-scaling** | Handle variable execution loads | Elastic compute resources |

---

## 2. TimescaleDB Schema Definitions

```sql
-- TimescaleDB extensions (automatically available in TimescaleDB Cloud)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embeddings

-- 1. CONFIGURATIONS: High-level configuration metadata
CREATE TABLE IF NOT EXISTS configurations (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL CHECK (length(name) >= 3 AND length(name) <= 100),
    user_id UUID NOT NULL,                  -- Required for multi-tenant security
    latest_version_id UUID,                 -- Points to current config_versions
    is_active BOOLEAN DEFAULT true,         -- Soft delete support
    modal_app_id TEXT,                      -- Modal application identifier
    sandbox_config JSONB DEFAULT '{}',     -- Modal sandbox configuration
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Add constraints for data integrity
    CONSTRAINT valid_name CHECK (name ~ '^[a-zA-Z0-9_\-\s]+$'),
    CONSTRAINT valid_timestamps CHECK (updated_at >= created_at)
);

-- 2. CONFIG_VERSIONS: Immutable YAML snapshots with embeddings
CREATE TABLE IF NOT EXISTS config_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_id UUID NOT NULL,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    raw_yaml TEXT NOT NULL CHECK (length(raw_yaml) <= 1048576), -- 1MB limit
    parsed_config JSONB NOT NULL,
    config_hash TEXT NOT NULL CHECK (length(config_hash) = 64), -- SHA256
    embedding VECTOR(1536),                 -- For config similarity search
    generated_by TEXT DEFAULT 'manual' CHECK (generated_by IN ('manual', 'llm_recommendation', 'diff_engine', 'api')),
    modal_deployment_id TEXT,               -- Modal deployment reference
    is_deleted BOOLEAN DEFAULT false,       -- Soft delete for audit trail
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    FOREIGN KEY (config_id) REFERENCES configurations(config_id) ON DELETE CASCADE,
    UNIQUE(config_id, version_number),
    
    -- Validate YAML structure contains required fields
    CONSTRAINT valid_parsed_config CHECK (
        parsed_config ? 'model' AND 
        parsed_config ? 'temperature' AND
        (parsed_config->>'temperature')::numeric BETWEEN 0 AND 2
    )
);

-- 3. CONFIG_DIFFS: Track YAML changes between versions
CREATE TABLE IF NOT EXISTS config_diffs (
    diff_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_id UUID NOT NULL,
    version_before UUID NOT NULL,
    version_after UUID NOT NULL,
    diff_yaml TEXT NOT NULL CHECK (length(diff_yaml) <= 524288), -- 512KB limit
    change_summary JSONB,
    impact_prediction JSONB,
    llm_generated BOOLEAN DEFAULT false,    -- Track if LLM generated this diff
    approval_status TEXT DEFAULT 'pending' CHECK (approval_status IN ('pending', 'approved', 'rejected')),
    approved_by UUID,                       -- User who approved the change
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    FOREIGN KEY (config_id) REFERENCES configurations(config_id) ON DELETE CASCADE,
    FOREIGN KEY (version_before) REFERENCES config_versions(version_id),
    FOREIGN KEY (version_after) REFERENCES config_versions(version_id),
    
    -- Ensure versions are different and ordered correctly
    CONSTRAINT different_versions CHECK (version_before != version_after)
);

-- 4. SCENARIOS: Test scenarios with embeddings for clustering
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id TEXT PRIMARY KEY CHECK (length(scenario_id) <= 100),
    name TEXT NOT NULL CHECK (length(name) >= 3 AND length(name) <= 200),
    task_prompt TEXT NOT NULL CHECK (length(task_prompt) >= 10),
    ground_truth JSONB,
    difficulty_level TEXT DEFAULT 'medium' CHECK (difficulty_level IN ('easy', 'medium', 'hard')),
    tags TEXT[] CHECK (array_length(tags, 1) <= 20),
    scenario_embedding VECTOR(1536),
    expected_tools JSONB,
    modal_function_name TEXT,               -- Modal function for this scenario
    is_active BOOLEAN DEFAULT true,
    estimated_duration_ms INTEGER CHECK (estimated_duration_ms > 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Validate scenario_id format
    CONSTRAINT valid_scenario_id CHECK (scenario_id ~ '^[a-z0-9_]+$')
);

-- 5. SIMULATIONS: Complete evaluation runs with Modal integration
CREATE TABLE IF NOT EXISTS simulations (
    simulation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_version_id UUID NOT NULL,
    scenario_set TEXT[] NOT NULL CHECK (array_length(scenario_set, 1) > 0),
    simulation_name TEXT CHECK (length(simulation_name) <= 200),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    total_scenarios INTEGER NOT NULL CHECK (total_scenarios > 0),
    completed_scenarios INTEGER DEFAULT 0 CHECK (completed_scenarios >= 0),
    overall_score REAL CHECK (overall_score BETWEEN 0 AND 1),
    execution_time_ms INTEGER CHECK (execution_time_ms >= 0),
    total_cost_usd REAL CHECK (total_cost_usd >= 0),
    metadata JSONB DEFAULT '{}',
    
    -- Modal-specific fields
    modal_app_id TEXT,                      -- Modal application ID
    modal_environment TEXT DEFAULT 'production', -- Modal environment (dev/staging/prod)
    sandbox_instances INTEGER DEFAULT 1,    -- Number of parallel sandbox instances
    
    timeout_ms INTEGER DEFAULT 300000,      -- 5 minute default timeout
    max_retries INTEGER DEFAULT 3,
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    FOREIGN KEY (config_version_id) REFERENCES config_versions(version_id),
    
    -- Logical constraints
    CONSTRAINT valid_completion CHECK (
        (status = 'completed' AND completed_at IS NOT NULL) OR
        (status != 'completed' AND completed_at IS NULL)
    ),
    CONSTRAINT completed_scenarios_limit CHECK (completed_scenarios <= total_scenarios),
    CONSTRAINT valid_timestamps CHECK (
        started_at IS NULL OR started_at >= created_at
    )
);

-- 6. SIMULATIONS_SCENARIOS: Junction table for M:N relationship
CREATE TABLE IF NOT EXISTS simulations_scenarios (
    simulation_id UUID NOT NULL,
    scenario_id TEXT NOT NULL,
    execution_order INTEGER NOT NULL CHECK (execution_order > 0),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    
    -- Modal execution tracking
    modal_call_id TEXT,                     -- Modal function call identifier
    modal_execution_logs TEXT,              -- Modal execution logs
    
    -- Execution metadata
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
    
    PRIMARY KEY (simulation_id, scenario_id),
    FOREIGN KEY (simulation_id) REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id),
    
    CONSTRAINT valid_execution_timestamps CHECK (
        completed_at IS NULL OR started_at IS NOT NULL
    )
);

-- 7. OUTCOMES: Individual scenario results (HYPERTABLE for time-series optimization)
CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id UUID DEFAULT uuid_generate_v4(),
    simulation_id UUID NOT NULL,
    scenario_id TEXT NOT NULL,
    execution_time TIMESTAMPTZ NOT NULL,    -- Primary time dimension
    status TEXT NOT NULL CHECK (status IN ('success', 'error', 'timeout', 'cancelled')),
    reliability_score REAL NOT NULL CHECK (reliability_score BETWEEN 0 AND 1),
    execution_time_ms REAL NOT NULL CHECK (execution_time_ms >= 0),
    tokens_used INTEGER NOT NULL CHECK (tokens_used >= 0),
    cost_usd REAL NOT NULL CHECK (cost_usd >= 0),
    trajectory JSONB NOT NULL,
    trajectory_embedding VECTOR(1536),
    metrics JSONB DEFAULT '{}',
    
    -- Modal sandbox tracking
    modal_call_id TEXT,                     -- Link to Modal execution
    sandbox_id TEXT,                        -- Modal sandbox identifier
    sandbox_logs TEXT,                      -- Full sandbox execution logs
    
    -- Enhanced failure tracking
    retry_count INTEGER DEFAULT 0 CHECK (retry_count >= 0),
    error_code TEXT,                        -- Structured error classification
    error_category TEXT CHECK (error_category IN ('timeout', 'tool_error', 'model_error', 'validation_error', 'system_error', 'sandbox_error')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    FOREIGN KEY (simulation_id) REFERENCES simulations(simulation_id) ON DELETE CASCADE,
    FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id),
    
    -- Validate trajectory contains required fields
    CONSTRAINT valid_trajectory CHECK (
        trajectory ? 'start_time' AND 
        trajectory ? 'status'
    ),
    
    -- Primary key includes time for TimescaleDB partitioning
    PRIMARY KEY (outcome_id, execution_time)
);

-- Convert outcomes to TimescaleDB hypertable (optimized for time-series queries)
SELECT create_hypertable('outcomes', 'execution_time', 
    chunk_time_interval => INTERVAL '1 day',
    create_default_indexes => FALSE
);

-- 8. FAILURE_PATTERNS: Structured failure analysis with time-based clustering
CREATE TABLE IF NOT EXISTS failure_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    outcome_id UUID NOT NULL,
    execution_time TIMESTAMPTZ NOT NULL,   -- Time dimension for pattern analysis
    failure_type TEXT NOT NULL CHECK (length(failure_type) <= 100),
    failure_category TEXT NOT NULL CHECK (failure_category IN 
        ('timeout', 'tool_error', 'model_error', 'validation_error', 'system_error', 'sandbox_error', 'modal_error', 'unknown')),
    error_message TEXT CHECK (length(error_message) <= 10000),
    error_embedding VECTOR(1536),
    
    -- Pattern clustering metadata
    pattern_cluster_id UUID,               -- Cluster this pattern belongs to
    cluster_confidence REAL CHECK (cluster_confidence BETWEEN 0 AND 1),
    similar_failures_count INTEGER DEFAULT 0 CHECK (similar_failures_count >= 0),
    
    -- Recovery and resolution
    recovery_attempted BOOLEAN DEFAULT FALSE,
    recovery_successful BOOLEAN DEFAULT FALSE,
    resolution_steps TEXT,                  -- How to fix this type of error
    llm_analysis JSONB,                    -- LLM-generated failure analysis
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Recovery logic validation
    CONSTRAINT valid_recovery CHECK (
        NOT recovery_successful OR recovery_attempted = true
    ),
    
    -- Primary key includes time for potential hypertable conversion
    PRIMARY KEY (pattern_id, execution_time)
);

-- 9. TOOL_USAGE: Tool performance analytics with time-series optimization
CREATE TABLE IF NOT EXISTS tool_usage (
    usage_id UUID DEFAULT uuid_generate_v4(),
    outcome_id UUID NOT NULL,
    execution_time TIMESTAMPTZ NOT NULL,   -- Time dimension
    tool_name TEXT NOT NULL CHECK (length(tool_name) <= 100),
    call_count INTEGER NOT NULL CHECK (call_count > 0),
    avg_duration_ms REAL NOT NULL CHECK (avg_duration_ms >= 0),
    success_rate REAL NOT NULL CHECK (success_rate BETWEEN 0 AND 1),
    tool_sequence JSONB,
    total_cost_usd REAL DEFAULT 0 CHECK (total_cost_usd >= 0),
    error_types TEXT[],                     -- Common error patterns for this tool
    
    -- Modal-specific tool tracking
    modal_function_calls JSONB,            -- Track Modal function interactions
    sandbox_tool_logs TEXT,                -- Tool execution logs from sandbox
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (usage_id, execution_time)
);

-- 10. RECOMMENDATIONS: LLM-generated improvement suggestions
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type TEXT NOT NULL CHECK (source_type IN ('simulation', 'outcome', 'config_diff', 'pattern_analysis', 'time_series_analysis')),
    source_id UUID NOT NULL,
    recommendation_type TEXT NOT NULL CHECK (recommendation_type IN 
        ('config_change', 'scenario_addition', 'tool_optimization', 'performance_tuning', 'modal_optimization')),
    description TEXT NOT NULL CHECK (length(description) >= 10),
    suggested_yaml_changes TEXT,
    impact_estimate JSONB,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    
    -- LLM recommendation metadata
    llm_model_used TEXT,                   -- Which LLM generated this recommendation
    llm_prompt_hash TEXT,                  -- Hash of the prompt used
    historical_data_window INTERVAL,       -- Time window of data analyzed
    
    -- Application tracking
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMPTZ,
    applied_by UUID,                       -- User who applied the recommendation
    feedback_score INTEGER CHECK (feedback_score BETWEEN 1 AND 5),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Applied logic validation
    CONSTRAINT valid_application CHECK (
        (applied = false AND applied_at IS NULL) OR
        (applied = true AND applied_at IS NOT NULL)
    )
);

-- 11. LLM_RECOMMENDATION_SESSIONS: Track LLM analysis sessions
CREATE TABLE IF NOT EXISTS llm_recommendation_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    analysis_start_time TIMESTAMPTZ NOT NULL,
    analysis_end_time TIMESTAMPTZ NOT NULL,
    data_window_start TIMESTAMPTZ NOT NULL,  -- Historical data analyzed from
    data_window_end TIMESTAMPTZ NOT NULL,    -- Historical data analyzed to
    
    -- LLM configuration
    llm_model TEXT NOT NULL,
    llm_temperature REAL DEFAULT 0.1,
    prompt_template_version TEXT,
    
    -- Analysis results
    total_outcomes_analyzed INTEGER NOT NULL CHECK (total_outcomes_analyzed >= 0),
    patterns_identified INTEGER DEFAULT 0,
    recommendations_generated INTEGER DEFAULT 0,
    analysis_cost_usd REAL DEFAULT 0 CHECK (analysis_cost_usd >= 0),
    
    -- Session metadata
    session_status TEXT DEFAULT 'completed' CHECK (session_status IN ('running', 'completed', 'failed')),
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_analysis_window CHECK (analysis_end_time >= analysis_start_time),
    CONSTRAINT valid_data_window CHECK (data_window_end >= data_window_start)
);

-- 12. AUDIT_LOG: Security and compliance tracking (no foreign key constraints)
CREATE TABLE IF NOT EXISTS audit_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    user_id UUID,
    old_data JSONB,
    new_data JSONB,
    ip_address INET,
    user_agent TEXT,
    modal_call_context JSONB,              -- Modal execution context if applicable
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Add foreign key constraint for configurations (deferred)
ALTER TABLE configurations 
ADD CONSTRAINT fk_latest_version 
FOREIGN KEY (latest_version_id) REFERENCES config_versions(version_id) DEFERRABLE INITIALLY DEFERRED;
```

---

## 3. TimescaleDB Optimizations

### A. Hypertables and Compression
```sql
-- Enable compression on outcomes hypertable (TimescaleDB feature)
ALTER TABLE outcomes SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'simulation_id, scenario_id',
    timescaledb.compress_orderby = 'execution_time DESC'
);

-- Automatically compress chunks older than 7 days
SELECT add_compression_policy('outcomes', INTERVAL '7 days');

-- Optional: Convert failure_patterns to hypertable for time-based analysis
SELECT create_hypertable('failure_patterns', 'execution_time', 
    chunk_time_interval => INTERVAL '7 days',
    create_default_indexes => FALSE
);

-- Optional: Convert tool_usage to hypertable
SELECT create_hypertable('tool_usage', 'execution_time', 
    chunk_time_interval => INTERVAL '1 day',
    create_default_indexes => FALSE
);
```

### B. Continuous Aggregates for Dashboard
```sql
-- Real-time dashboard metrics with continuous aggregates
CREATE MATERIALIZED VIEW simulation_performance_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', execution_time) AS hour,
    simulation_id,
    COUNT(*) AS total_outcomes,
    AVG(reliability_score) AS avg_reliability,
    AVG(execution_time_ms) AS avg_execution_time,
    SUM(cost_usd) AS total_cost,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count
FROM outcomes
GROUP BY hour, simulation_id;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('simulation_performance_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Failure pattern trends for LLM analysis
CREATE MATERIALIZED VIEW failure_trends_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', execution_time) AS day,
    failure_category,
    COUNT(*) AS failure_count,
    AVG(cluster_confidence) AS avg_confidence,
    COUNT(DISTINCT pattern_cluster_id) AS unique_patterns
FROM failure_patterns
GROUP BY day, failure_category;

-- Tool performance aggregation
CREATE MATERIALIZED VIEW tool_performance_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', execution_time) AS day,
    tool_name,
    AVG(success_rate) AS avg_success_rate,
    AVG(avg_duration_ms) AS avg_duration,
    SUM(total_cost_usd) AS daily_cost,
    COUNT(*) AS usage_count
FROM tool_usage
GROUP BY day, tool_name;
```sql

### C. Optimized Indexes for TimescaleDB
```sql
-- Time-based indexes optimized for TimescaleDB
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_time_status 
    ON outcomes(execution_time DESC, status) WHERE status != 'success';
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_simulation_time 
    ON outcomes(simulation_id, execution_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_scenario_time 
    ON outcomes(scenario_id, execution_time DESC);

-- Modal-specific indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_modal_call 
    ON outcomes(modal_call_id) WHERE modal_call_id IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_simulations_modal_app 
    ON simulations(modal_app_id, created_at DESC);

-- Configuration and versioning indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_config_versions_config_version 
    ON config_versions(config_id, version_number DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_config_diffs_llm_generated 
    ON config_diffs(llm_generated, created_at DESC) WHERE llm_generated = true;

-- Vector similarity indexes for recommendations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_config_embedding_cosine 
    ON config_versions USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100) WHERE embedding IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trajectory_embedding_cosine 
    ON outcomes USING ivfflat (trajectory_embedding vector_cosine_ops) 
    WITH (lists = 500) WHERE trajectory_embedding IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scenario_embedding_cosine 
    ON scenarios USING ivfflat (scenario_embedding vector_cosine_ops) 
    WITH (lists = 50) WHERE scenario_embedding IS NOT NULL;

-- JSONB indexes for complex queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_outcomes_trajectory_gin 
    ON outcomes USING GIN (trajectory) WHERE trajectory IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_config_versions_parsed_gin 
    ON config_versions USING GIN (parsed_config);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_impact_gin 
    ON recommendations USING GIN (impact_estimate) WHERE impact_estimate IS NOT NULL;

-- Dashboard and analytics indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendations_pending_confidence 
    ON recommendations(applied, confidence DESC, created_at DESC) 
    WHERE applied = false;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failure_patterns_cluster_time 
    ON failure_patterns(pattern_cluster_id, execution_time DESC) 
    WHERE pattern_cluster_id IS NOT NULL;
```sql

---

## 4. TimescaleDB Deployment Configuration

### A. Connection Configuration
```python
# TimescaleDB connection settings
TIMESCALEDB_CONFIG = {
    "host": "your-instance.timescaledb.cloud",  # TimescaleDB Cloud endpoint
    "port": 5432,
    "database": "arc_eval_production",
    "user": "tsdbadmin",  # Default TimescaleDB admin user
    "password": "your-secure-password",
    "sslmode": "require",  # Always use SSL with TimescaleDB Cloud
    "connect_timeout": 10,
    "command_timeout": 60,
    "server_settings": {
        "jit": "off",  # Recommended for analytical workloads
        "timezone": "UTC"
    }
}

# Connection pooling for high-throughput Modal executions
POOL_CONFIG = {
    "min_size": 10,
    "max_size": 50,
    "max_queries": 50000,
    "max_inactive_connection_lifetime": 300
}
```sql

### B. Modal Integration Settings
```python
# Modal configuration for sandbox executions
MODAL_CONFIG = {
    "app_name": "arc-eval-sandbox",
    "environment": "production",  # dev, staging, production
    "image": "arc-eval:latest",
    "secrets": ["timescaledb-credentials"],
    "timeout": 300,  # 5 minutes per scenario
    "cpu": 2.0,
    "memory": 4096,  # 4GB per sandbox instance
    "retries": 3,
    "concurrency_limit": 100  # Max parallel executions
}
```sql

### C. Retention Policies
```sql
-- Automatic data retention with TimescaleDB
SELECT add_retention_policy('outcomes', INTERVAL '2 years');
SELECT add_retention_policy('failure_patterns', INTERVAL '1 year');
SELECT add_retention_policy('tool_usage', INTERVAL '6 months');
SELECT add_retention_policy('audit_log', INTERVAL '1 year');

-- Compress old data to save storage costs
SELECT add_compression_policy('outcomes', INTERVAL '30 days');
SELECT add_compression_policy('failure_patterns', INTERVAL '14 days');
SELECT add_compression_policy('tool_usage', INTERVAL '7 days');
```

---

## 5. LLM Recommendation Engine Integration

### A. Time-Series Analysis Views for LLM
```sql
-- View for LLM to analyze failure trends over time
CREATE OR REPLACE VIEW v_llm_failure_analysis AS
SELECT 
    date_trunc('day', execution_time) as day,
    failure_category,
    COUNT(*) as daily_failures,
    AVG(cluster_confidence) as avg_confidence,
    ARRAY_AGG(DISTINCT error_message) FILTER (WHERE error_message IS NOT NULL) as sample_errors,
    ARRAY_AGG(DISTINCT scenario_id) as affected_scenarios
FROM failure_patterns fp
JOIN outcomes o ON fp.outcome_id = o.outcome_id
WHERE fp.execution_time > NOW() - INTERVAL '30 days'
GROUP BY day, failure_category
ORDER BY day DESC, daily_failures DESC;

-- Configuration performance over time for recommendations
CREATE OR REPLACE VIEW v_llm_config_performance AS
SELECT 
    cv.config_id,
    cv.version_number,
    cv.parsed_config,
    date_trunc('day', o.execution_time) as day,
    COUNT(*) as total_executions,
    AVG(o.reliability_score) as avg_reliability,
    AVG(o.execution_time_ms) as avg_execution_time,
    SUM(o.cost_usd) as daily_cost,
    COUNT(*) FILTER (WHERE o.status = 'error') as error_count
FROM config_versions cv
JOIN simulations s ON cv.version_id = s.config_version_id
JOIN outcomes o ON s.simulation_id = o.simulation_id
WHERE o.execution_time > NOW() - INTERVAL '90 days'
GROUP BY cv.config_id, cv.version_number, cv.parsed_config, day
ORDER BY day DESC, avg_reliability DESC;

-- Scenario difficulty trends for LLM optimization
CREATE OR REPLACE VIEW v_llm_scenario_trends AS
SELECT 
    scenario_id,
    date_trunc('week', execution_time) as week,
    COUNT(*) as executions,
    AVG(reliability_score) as avg_reliability,
    STDDEV(reliability_score) as reliability_variance,
    AVG(execution_time_ms) as avg_duration,
    COUNT(*) FILTER (WHERE status = 'timeout') as timeout_count
FROM outcomes
WHERE execution_time > NOW() - INTERVAL '6 months'
GROUP BY scenario_id, week
ORDER BY week DESC, avg_reliability ASC;
```sql

### B. LLM Recommendation Functions
```sql
-- Function to trigger LLM analysis based on recent performance
CREATE OR REPLACE FUNCTION trigger_llm_recommendation_analysis(
    analysis_window INTERVAL DEFAULT '7 days',
    user_id_param UUID DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    session_id UUID;
    outcomes_count INTEGER;
    patterns_count INTEGER;
BEGIN
    -- Create new analysis session
    INSERT INTO llm_recommendation_sessions (
        session_id, user_id, analysis_start_time, analysis_end_time,
        data_window_start, data_window_end, llm_model,
        total_outcomes_analyzed, session_status
    ) VALUES (
        uuid_generate_v4(), user_id_param, NOW(), NOW(),
        NOW() - analysis_window, NOW(), 'gpt-4-turbo',
        0, 'running'
    ) RETURNING llm_recommendation_sessions.session_id INTO session_id;
    
    -- Count outcomes in analysis window
    SELECT COUNT(*) INTO outcomes_count
    FROM outcomes 
    WHERE execution_time > (NOW() - analysis_window);
    
    -- Count failure patterns in analysis window
    SELECT COUNT(*) INTO patterns_count
    FROM failure_patterns 
    WHERE execution_time > (NOW() - analysis_window);
    
    -- Update session with counts
    UPDATE llm_recommendation_sessions 
    SET total_outcomes_analyzed = outcomes_count,
        patterns_identified = patterns_count,
        session_status = 'completed'
    WHERE llm_recommendation_sessions.session_id = session_id;
    
    RETURN session_id;
END;
$$ LANGUAGE plpgsql;
```sql

---

## 6. Dashboard Analytics Views

### A. Real-Time Performance Dashboard
```sql
-- Active simulations with progress tracking
CREATE OR REPLACE VIEW v_dashboard_active_simulations AS
SELECT 
    s.simulation_id,
    s.simulation_name,
    s.status,
    s.completed_scenarios,
    s.total_scenarios,
    ROUND((s.completed_scenarios::float / s.total_scenarios * 100), 2) as progress_pct,
    EXTRACT(EPOCH FROM NOW() - s.started_at) as runtime_seconds,
    s.modal_app_id,
    cv.parsed_config->>'model' as model_name,
    COUNT(o.outcome_id) as outcomes_recorded
FROM simulations s
JOIN config_versions cv ON s.config_version_id = cv.version_id
LEFT JOIN outcomes o ON s.simulation_id = o.simulation_id
WHERE s.status IN ('pending', 'running')
GROUP BY s.simulation_id, s.simulation_name, s.status, s.completed_scenarios, 
         s.total_scenarios, s.started_at, s.modal_app_id, cv.parsed_config
ORDER BY s.created_at DESC;

-- Recent recommendations ready for user review
CREATE OR REPLACE VIEW v_dashboard_pending_recommendations AS
SELECT 
    r.recommendation_id,
    r.recommendation_type,
    r.description,
    r.confidence,
    r.impact_estimate,
    r.suggested_yaml_changes,
    lrs.llm_model as generated_by_model,
    lrs.data_window_start,
    lrs.data_window_end,
    r.created_at
FROM recommendations r
JOIN llm_recommendation_sessions lrs ON r.created_at BETWEEN lrs.analysis_start_time AND lrs.analysis_end_time
WHERE r.applied = false 
  AND r.confidence > 0.7  -- Only show high-confidence recommendations
ORDER BY r.confidence DESC, r.created_at DESC
LIMIT 10;

-- Performance trends for the last 30 days
CREATE OR REPLACE VIEW v_dashboard_performance_trends AS
SELECT 
    date_trunc('day', execution_time) as day,
    COUNT(*) as total_executions,
    AVG(reliability_score) as avg_reliability,
    AVG(execution_time_ms) as avg_duration_ms,
    SUM(cost_usd) as daily_cost,
    COUNT(*) FILTER (WHERE status = 'error') as error_count,
    ROUND((COUNT(*) FILTER (WHERE status = 'error')::float / COUNT(*) * 100), 2) as error_rate_pct
FROM outcomes
WHERE execution_time > NOW() - INTERVAL '30 days'
GROUP BY day
ORDER BY day DESC;
```sql

### B. Cost and Usage Analytics
```sql
-- Daily cost breakdown by model and configuration
CREATE OR REPLACE VIEW v_dashboard_cost_analysis AS
SELECT 
    date_trunc('day', o.execution_time) as day,
    cv.parsed_config->>'model' as model,
    COUNT(*) as executions,
    SUM(o.cost_usd) as total_cost,
    AVG(o.cost_usd) as avg_cost_per_execution,
    SUM(o.tokens_used) as total_tokens,
    AVG(o.tokens_used) as avg_tokens_per_execution
FROM outcomes o
JOIN simulations s ON o.simulation_id = s.simulation_id
JOIN config_versions cv ON s.config_version_id = cv.version_id
WHERE o.execution_time > NOW() - INTERVAL '30 days'
GROUP BY day, cv.parsed_config->>'model'
ORDER BY day DESC, total_cost DESC;
```sql

---

## 7. TimescaleDB Migration Guide

### A. From Self-Hosted PostgreSQL
```sql
-- 1. Export existing data using pg_dump
-- pg_dump -h localhost -U postgres -d arc_eval --data-only --inserts > arc_eval_data.sql

-- 2. Create TimescaleDB instance and run schema
-- psql -h your-instance.timescaledb.cloud -U tsdbadmin -d arc_eval_production -f timescaledb_schema.sql

-- 3. Import data (outcomes table will need special handling for hypertable)
-- First import non-time-series tables
-- psql -h your-instance.timescaledb.cloud -U tsdbadmin -d arc_eval_production -c "
-- COPY configurations FROM 'configurations.csv' WITH CSV HEADER;
-- COPY config_versions FROM 'config_versions.csv' WITH CSV HEADER;
-- -- ... other tables
-- "

-- 4. Import outcomes data in time-ordered batches
-- psql -h your-instance.timescaledb.cloud -U tsdbadmin -d arc_eval_production -c "
-- COPY outcomes FROM 'outcomes.csv' WITH CSV HEADER;
-- "
```sql

### B. Production Checklist for TimescaleDB
- [ ] **TimescaleDB Cloud Instance**: Created with appropriate tier (>=4GB RAM)
- [ ] **Extensions Enabled**: timescaledb, uuid-ossp, vector
- [ ] **Hypertables Created**: outcomes converted to hypertable
- [ ] **Compression Enabled**: 7-day compression policy on outcomes
- [ ] **Continuous Aggregates**: Dashboard materialized views created
- [ ] **Retention Policies**: Configured for cost optimization
- [ ] **Connection Pooling**: Application connection pool configured
- [ ] **Monitoring**: TimescaleDB Cloud monitoring enabled
- [ ] **Backups**: Automated backups configured
- [ ] **Security**: SSL enforced, user access configured

This TimescaleDB-optimized schema is now ready for:
- ✅ **Modal Sandbox Integration**: Seamless execution tracking
- ✅ **Time-Series Performance**: Optimized for high-throughput outcomes
- ✅ **LLM Recommendations**: Historical pattern analysis
- ✅ **Real-Time Dashboard**: Live performance monitoring
- ✅ **Managed Infrastructure**: No ops overhead with TimescaleDB Cloud
- ✅ **Cost Optimization**: Automatic compression and retention
- ✅ **Scalability**: Auto-scaling compute and storage

---
*Updated: June 9 2024 - Optimized for TimescaleDB Cloud and Modal integration*

## 8. API-First Integration Design

### A. Database Client Interface
```python
# Flexible database client that your Modal sandbox team can use
class ArcEvalDBClient:
    """
    Flexible database client designed for Modal sandbox integration.
    This interface remains stable while internal implementation can evolve.
    """
    
    async def record_execution_start(self, execution_data: dict) -> str:
        """
        Record when a sandbox execution begins.
        Returns execution_id for tracking.
        """
        pass
    
    async def record_execution_outcome(self, outcome_data: dict) -> str:
        """
        Record the final outcome of a sandbox execution.
        Handles both success and failure cases.
        """
        pass
    
    async def record_tool_usage(self, tool_events: List[dict]) -> None:
        """
        Batch record tool usage events from sandbox execution.
        """
        pass
    
    async def record_failure_pattern(self, failure_data: dict) -> None:
        """
        Record structured failure information for pattern analysis.
        """
        pass
    
    async def get_active_configurations(self) -> List[dict]:
        """
        Get current active configurations for sandbox execution.
        """
        pass

# Example usage in Modal sandbox
async def execute_scenario(scenario_id: str, config_id: str):
    db_client = ArcEvalDBClient()
    
    # Start execution tracking
    execution_id = await db_client.record_execution_start({
        "scenario_id": scenario_id,
        "config_id": config_id,
        "modal_call_id": current_modal_call_id(),
        "started_at": datetime.utcnow()
    })
    
    try:
        # Run scenario in sandbox
        result = await run_scenario_in_sandbox(scenario_id, config_id)
        
        # Record outcome
        await db_client.record_execution_outcome({
            "execution_id": execution_id,
            "status": "success" if result.success else "error",
            "reliability_score": result.score,
            "trajectory": result.trajectory,
            "cost_usd": result.cost,
            "execution_time_ms": result.duration_ms
        })
        
    except Exception as e:
        # Record failure
        await db_client.record_failure_pattern({
            "execution_id": execution_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "failure_category": classify_error(e)
        })
```sql

### B. Event-Driven Data Flow
```python
# Flexible event-driven pattern for sandbox-to-database integration
class ExecutionEventHandler:
    """
    Handle events from Modal sandbox executions.
    Designed to be extensible as your data needs evolve.
    """
    
    async def handle_execution_started(self, event: dict):
        """Handle sandbox execution start events."""
        await self.db.create_simulation_record({
            "simulation_id": event["simulation_id"],
            "modal_app_id": event["modal_app_id"],
            "status": "running",
            "started_at": event["timestamp"]
        })
    
    async def handle_scenario_completed(self, event: dict):
        """Handle individual scenario completion events."""
        await self.db.record_outcome({
            "outcome_id": event["outcome_id"],
            "simulation_id": event["simulation_id"],
            "scenario_id": event["scenario_id"],
            "execution_time": event["timestamp"],
            "status": event["status"],
            "trajectory": event["trajectory"],
            "modal_call_id": event["modal_call_id"],
            "sandbox_logs": event.get("logs", "")
        })
    
    async def handle_execution_completed(self, event: dict):
        """Handle complete simulation finish events."""
        await self.db.finalize_simulation({
            "simulation_id": event["simulation_id"],
            "status": "completed",
            "completed_at": event["timestamp"],
            "overall_score": event["overall_score"]
        })

# Integration with Modal
@modal.function
async def run_scenario_evaluation(scenario_id: str, config_id: str):
    event_handler = ExecutionEventHandler()
    
    # Emit events that database layer can handle
    await event_handler.handle_execution_started({
        "simulation_id": simulation_id,
        "modal_app_id": modal.current_app_id(),
        "timestamp": datetime.utcnow()
    })
    
    # ... run scenario ...
    
    await event_handler.handle_scenario_completed({
        "outcome_id": outcome_id,
        "simulation_id": simulation_id,
        "scenario_id": scenario_id,
        "timestamp": datetime.utcnow(),
        "status": result.status,
        "trajectory": result.trajectory,
        "modal_call_id": modal.current_call_id()
    })
```sql

### C. Configuration Management API
```python
# Flexible configuration management for Modal sandbox
class ConfigurationManager:
    """
    Manage agent configurations with versioning.
    Modal sandbox can request configs without knowing internal structure.
    """
    
    async def get_active_config(self, config_name: str) -> dict:
        """Get the latest active configuration."""
        result = await self.db.query("""
            SELECT cv.parsed_config, cv.version_id
            FROM configurations c
            JOIN config_versions cv ON c.latest_version_id = cv.version_id
            WHERE c.name = $1 AND c.is_active = true
        """, config_name)
        
        return {
            "config": result["parsed_config"],
            "version_id": result["version_id"],
            "modal_deployment_id": result.get("modal_deployment_id")
        }
    
    async def create_config_version(self, config_data: dict) -> str:
        """Create new configuration version."""
        # Hash the config for deduplication
        config_hash = hashlib.sha256(
            json.dumps(config_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Check if this exact config already exists
        existing = await self.db.fetch_one("""
            SELECT version_id FROM config_versions 
            WHERE config_hash = $1
        """, config_hash)
        
        if existing:
            return existing["version_id"]
        
        # Create new version
        return await self.db.create_config_version({
            "config_hash": config_hash,
            "parsed_config": config_data,
            "generated_by": "modal_sandbox"
        })
```sql

---

## 9. Integration Patterns for Modal Sandbox

### A. Asynchronous Data Ingestion
```python
# High-throughput data ingestion pattern
class BatchExecutionRecorder:
    """
    Batch multiple execution outcomes for efficient database writes.
    Critical for high-volume Modal sandbox executions.
    """
    
    def __init__(self, batch_size: int = 100, flush_interval: int = 30):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.outcome_buffer = []
        self.tool_usage_buffer = []
        self.failure_buffer = []
    
    async def record_outcome(self, outcome: dict):
        """Buffer outcome for batch insert."""
        self.outcome_buffer.append({
            **outcome,
            "created_at": datetime.utcnow()
        })
        
        if len(self.outcome_buffer) >= self.batch_size:
            await self.flush_outcomes()
    
    async def flush_outcomes(self):
        """Batch insert outcomes to TimescaleDB."""
        if not self.outcome_buffer:
            return
            
        await self.db.execute_many("""
            INSERT INTO outcomes (
                outcome_id, simulation_id, scenario_id, execution_time,
                status, reliability_score, execution_time_ms, tokens_used,
                cost_usd, trajectory, modal_call_id, sandbox_id, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """, self.outcome_buffer)
        
        self.outcome_buffer.clear()
```sql

### B. Flexible Schema Evolution
```sql
-- Design tables to handle schema evolution
-- Modal sandbox can send additional fields without breaking

-- Use JSONB for flexible metadata that can evolve
ALTER TABLE outcomes ADD COLUMN IF NOT EXISTS execution_metadata JSONB DEFAULT '{}';
ALTER TABLE simulations ADD COLUMN IF NOT EXISTS sandbox_metadata JSONB DEFAULT '{}';

-- Create indexes that support flexible queries
CREATE INDEX IF NOT EXISTS idx_outcomes_metadata_gin 
    ON outcomes USING GIN (execution_metadata);

-- Example: Modal sandbox can add new fields dynamically
-- Original outcome record:
{
    "outcome_id": "uuid",
    "status": "success",
    "execution_metadata": {
        "modal_version": "0.1.0",
        "sandbox_type": "docker"
    }
}

-- Later, Modal team adds new tracking:
{
    "outcome_id": "uuid", 
    "status": "success",
    "execution_metadata": {
        "modal_version": "0.2.0",
        "sandbox_type": "modal_container",
        "resource_usage": {"cpu": 0.5, "memory": 1024},
        "new_feature_flag": true
    }
}
```sql

### C. Health Check and Monitoring
```python
# Database health checks for Modal sandbox integration
class DatabaseHealthChecker:
    """
    Monitor database health and performance for Modal sandbox integration.
    """
    
    async def check_write_performance(self) -> dict:
        """Check if database can handle current write load."""
        start_time = time.time()
        
        # Test batch insert performance
        test_outcomes = [
            {
                "outcome_id": uuid.uuid4(),
                "simulation_id": uuid.uuid4(),
                "scenario_id": "test_scenario",
                "execution_time": datetime.utcnow(),
                "status": "success",
                "reliability_score": 0.9,
                "execution_time_ms": 1000,
                "tokens_used": 100,
                "cost_usd": 0.01,
                "trajectory": {"test": True}
            }
            for _ in range(100)
        ]
        
        await self.db.insert_batch_outcomes(test_outcomes)
        
        write_time = time.time() - start_time
        throughput = len(test_outcomes) / write_time
        
        return {
            "status": "healthy" if throughput > 50 else "degraded",
            "throughput_per_second": throughput,
            "batch_write_time_ms": write_time * 1000
        }
    
    async def check_hypertable_health(self) -> dict:
        """Check TimescaleDB hypertable health."""
        result = await self.db.fetch_one("""
            SELECT 
                hypertable_name,
                num_chunks,
                total_bytes,
                compressed_bytes,
                compression_ratio
            FROM timescaledb_information.hypertables h
            LEFT JOIN timescaledb_information.chunks c ON h.hypertable_name = c.hypertable_name
            WHERE h.hypertable_name = 'outcomes'
        """)
        
        return {
            "hypertable_status": "healthy",
            "chunk_count": result["num_chunks"],
            "compression_ratio": result["compression_ratio"],
            "storage_efficiency": "good" if result["compression_ratio"] > 2 else "needs_attention"
        }
```sql

---

## 10. Implementation Checklist

### A. TimescaleDB Cloud Setup
```bash
# 1. Create TimescaleDB Cloud account
# Go to: https://console.cloud.timescale.com/

# 2. Create new service with these specs:
# - Database: arc_eval_production
# - Region: Choose closest to your Modal deployments
# - Tier: Minimum 4GB RAM, 2 CPU cores
# - Extensions: timescaledb, vector, uuid-ossp

# 3. Get connection details
HOST="your-instance-id.timescaledb.cloud"
PORT="5432" 
DATABASE="arc_eval_production"
USER="tsdbadmin"
PASSWORD="your-generated-password"
```sql

### B. Schema Deployment
```bash
# 1. Save the schema to a file
cat > timescaledb_schema.sql << 'EOF'
# [Include the full schema from section 2 above]
EOF

# 2. Deploy to TimescaleDB
psql "postgresql://tsdbadmin:${PASSWORD}@${HOST}:${PORT}/${DATABASE}?sslmode=require" \
     -f timescaledb_schema.sql

# 3. Verify hypertable creation
psql "postgresql://tsdbadmin:${PASSWORD}@${HOST}:${PORT}/${DATABASE}?sslmode=require" \
     -c "SELECT * FROM timescaledb_information.hypertables;"
```sql

### C. Environment Configuration
```python
# Create .env file for your project
TIMESCALEDB_HOST="your-instance-id.timescaledb.cloud"
TIMESCALEDB_PORT="5432"
TIMESCALEDB_DATABASE="arc_eval_production"
TIMESCALEDB_USER="tsdbadmin"
TIMESCALEDB_PASSWORD="your-password"
TIMESCALEDB_SSL_MODE="require"

# Connection pool settings
DB_POOL_MIN_SIZE=10
DB_POOL_MAX_SIZE=50
DB_POOL_MAX_QUERIES=50000
```sql

### D. Basic Client Implementation
```python
# Create arc_eval_db/client.py
import asyncio
import asyncpg
from typing import Dict, List, Optional
import json
from datetime import datetime

class ArcEvalDBClient:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=10,
            max_size=50,
            command_timeout=60
        )
    
    async def record_execution_start(self, execution_data: dict) -> str:
        """Record simulation start for Modal sandbox"""
        async with self.pool.acquire() as conn:
            simulation_id = await conn.fetchval("""
                INSERT INTO simulations (
                    simulation_id, config_version_id, scenario_set,
                    status, total_scenarios, modal_app_id, modal_environment,
                    created_at, started_at
                ) VALUES (
                    gen_random_uuid(), $1, $2, 'running', $3, $4, $5, NOW(), NOW()
                ) RETURNING simulation_id
            """, 
                execution_data["config_version_id"],
                execution_data["scenario_set"],
                len(execution_data["scenario_set"]),
                execution_data.get("modal_app_id"),
                execution_data.get("modal_environment", "production")
            )
            return str(simulation_id)
    
    async def record_execution_outcome(self, outcome_data: dict) -> str:
        """Record individual scenario outcome"""
        async with self.pool.acquire() as conn:
            outcome_id = await conn.fetchval("""
                INSERT INTO outcomes (
                    outcome_id, simulation_id, scenario_id, execution_time,
                    status, reliability_score, execution_time_ms, tokens_used,
                    cost_usd, trajectory, modal_call_id, sandbox_id,
                    execution_metadata, created_at
                ) VALUES (
                    gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW()
                ) RETURNING outcome_id
            """,
                outcome_data["simulation_id"],
                outcome_data["scenario_id"], 
                outcome_data.get("execution_time", datetime.utcnow()),
                outcome_data["status"],
                outcome_data["reliability_score"],
                outcome_data["execution_time_ms"],
                outcome_data["tokens_used"],
                outcome_data["cost_usd"],
                json.dumps(outcome_data["trajectory"]),
                outcome_data.get("modal_call_id"),
                outcome_data.get("sandbox_id"),
                json.dumps(outcome_data.get("metadata", {}))
            )
            return str(outcome_id)
    
    async def get_active_configurations(self) -> List[dict]:
        """Get active configurations for Modal sandbox"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    c.config_id,
                    c.name,
                    cv.version_id,
                    cv.parsed_config,
                    cv.modal_deployment_id
                FROM configurations c
                JOIN config_versions cv ON c.latest_version_id = cv.version_id
                WHERE c.is_active = true
            """)
            
            return [
                {
                    "config_id": str(row["config_id"]),
                    "config_name": row["name"],
                    "version_id": str(row["version_id"]),
                    "config": row["parsed_config"],
                    "modal_deployment_id": row["modal_deployment_id"]
                }
                for row in rows
            ]

# Usage example for Modal team
async def main():
    client = ArcEvalDBClient(
        "postgresql://tsdbadmin:password@host:5432/database"
    )
    await client.initialize()
    
    # Modal team can use this interface
    simulation_id = await client.record_execution_start({
        "config_version_id": "uuid-here",
        "scenario_set": ["scenario1", "scenario2"],
        "modal_app_id": "arc-eval-sandbox"
    })
    
    # Record outcome after scenario completes
    await client.record_execution_outcome({
        "simulation_id": simulation_id,
        "scenario_id": "scenario1",
        "status": "success",
        "reliability_score": 0.95,
        "execution_time_ms": 2500,
        "tokens_used": 1500,
        "cost_usd": 0.02,
        "trajectory": {"steps": [...], "result": "success"},
        "modal_call_id": "modal-call-123"
    })
```sql

### E. Testing and Validation
```python
# Create tests/test_db_integration.py
import pytest
import asyncio
from arc_eval_db.client import ArcEvalDBClient

@pytest.mark.asyncio
async def test_execution_recording():
    """Test that Modal sandbox integration works end-to-end"""
    client = ArcEvalDBClient("your-connection-string")
    await client.initialize()
    
    # Test simulation creation
    simulation_id = await client.record_execution_start({
        "config_version_id": "test-config-uuid",
        "scenario_set": ["test_scenario_1", "test_scenario_2"],
        "modal_app_id": "test-app"
    })
    
    assert simulation_id is not None
    
    # Test outcome recording
    outcome_id = await client.record_execution_outcome({
        "simulation_id": simulation_id,
        "scenario_id": "test_scenario_1", 
        "status": "success",
        "reliability_score": 0.9,
        "execution_time_ms": 1000,
        "tokens_used": 100,
        "cost_usd": 0.01,
        "trajectory": {"test": True}
    })
    
    assert outcome_id is not None

# Run tests
# pytest tests/test_db_integration.py -v
```sql

### F. Documentation for Modal Team
```markdown
# Arc-Eval Database Integration Guide

## Quick Start for Modal Team

### 1. Install the client
```bash
pip install arc-eval-db
```sql

### 2. Initialize in your Modal app
```python
from arc_eval_db import ArcEvalDBClient

# In your Modal function
db_client = ArcEvalDBClient(os.environ["TIMESCALEDB_CONNECTION_STRING"])
await db_client.initialize()
```sql

### 3. Record execution data
```python
# Start simulation
simulation_id = await db_client.record_execution_start({
    "config_version_id": config_id,
    "scenario_set": scenario_list,
    "modal_app_id": modal.current_app_id()
})

# Record each outcome
for scenario_result in results:
    await db_client.record_execution_outcome({
        "simulation_id": simulation_id,
        "scenario_id": scenario_result.scenario_id,
        "status": scenario_result.status,
        "reliability_score": scenario_result.score,
        "execution_time_ms": scenario_result.duration,
        "tokens_used": scenario_result.tokens,
        "cost_usd": scenario_result.cost,
        "trajectory": scenario_result.trajectory,
        "modal_call_id": modal.current_call_id()
    })
```sql

## API Reference
[Include detailed API documentation]
```sql

---

*Your TimescaleDB architecture is now ready for flexible Modal sandbox integration!*


