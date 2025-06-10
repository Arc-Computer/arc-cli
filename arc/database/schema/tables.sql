-- TimescaleDB Schema for Arc-Eval Platform
-- Generated from docs/db/DATABASE.md
-- Optimized for time-series workloads and Modal sandbox integration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

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

-- 8. FAILURE_PATTERNS: Structured failure analysis with time-based clustering
CREATE TABLE IF NOT EXISTS failure_patterns (
    pattern_id UUID DEFAULT uuid_generate_v4(),
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

-- Add deferred foreign key constraint for configurations (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'fk_latest_version' 
        AND table_name = 'configurations'
    ) THEN
        ALTER TABLE configurations 
        ADD CONSTRAINT fk_latest_version 
        FOREIGN KEY (latest_version_id) REFERENCES config_versions(version_id) DEFERRABLE INITIALLY DEFERRED;
    END IF;
END $$;
