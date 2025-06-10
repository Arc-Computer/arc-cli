-- TimescaleDB Optimizations for Arc Platform
-- Hypertables, Indexes, Compression, and Continuous Aggregates

-- ============================================================================
-- HYPERTABLES: Convert time-series tables
-- ============================================================================

-- Convert outcomes to TimescaleDB hypertable (main time-series table)
-- Only create if table exists and is not already a hypertable
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'outcomes') THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'outcomes') THEN
            PERFORM create_hypertable('outcomes', 'execution_time', 
                chunk_time_interval => INTERVAL '1 day',
                create_default_indexes => FALSE
            );
        END IF;
    END IF;
END $$;

-- Optional: Convert failure_patterns to hypertable for time-based analysis
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'failure_patterns') THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'failure_patterns') THEN
            PERFORM create_hypertable('failure_patterns', 'execution_time', 
                chunk_time_interval => INTERVAL '7 days',
                create_default_indexes => FALSE
            );
        END IF;
    END IF;
END $$;

-- Optional: Convert tool_usage to hypertable
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tool_usage') THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'tool_usage') THEN
            PERFORM create_hypertable('tool_usage', 'execution_time', 
                chunk_time_interval => INTERVAL '1 day',
                create_default_indexes => FALSE
            );
        END IF;
    END IF;
END $$;

-- ============================================================================
-- COMPRESSION POLICIES: Optimize storage costs
-- ============================================================================

-- Enable compression on outcomes hypertable
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'outcomes') THEN
        ALTER TABLE outcomes SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'simulation_id, scenario_id',
            timescaledb.compress_orderby = 'execution_time DESC'
        );
        -- Add compression policy if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression' AND hypertable_name = 'outcomes') THEN
            PERFORM add_compression_policy('outcomes', INTERVAL '7 days');
        END IF;
    END IF;
END $$;

-- Enable compression on failure_patterns
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'failure_patterns') THEN
        ALTER TABLE failure_patterns SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'failure_category',
            timescaledb.compress_orderby = 'execution_time DESC'
        );
        -- Add compression policy if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression' AND hypertable_name = 'failure_patterns') THEN
            PERFORM add_compression_policy('failure_patterns', INTERVAL '14 days');
        END IF;
    END IF;
END $$;

-- Enable compression on tool_usage
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'tool_usage') THEN
        ALTER TABLE tool_usage SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'tool_name',
            timescaledb.compress_orderby = 'execution_time DESC'
        );
        -- Add compression policy if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression' AND hypertable_name = 'tool_usage') THEN
            PERFORM add_compression_policy('tool_usage', INTERVAL '7 days');
        END IF;
    END IF;
END $$;

-- ============================================================================
-- RETENTION POLICIES: Automatic data lifecycle management
-- ============================================================================

-- Retain outcomes for 2 years
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'outcomes') THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention' AND hypertable_name = 'outcomes') THEN
            PERFORM add_retention_policy('outcomes', INTERVAL '2 years');
        END IF;
    END IF;
END $$;

-- Retain failure patterns for 1 year
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'failure_patterns') THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention' AND hypertable_name = 'failure_patterns') THEN
            PERFORM add_retention_policy('failure_patterns', INTERVAL '1 year');
        END IF;
    END IF;
END $$;

-- Retain tool usage for 6 months
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'tool_usage') THEN
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention' AND hypertable_name = 'tool_usage') THEN
            PERFORM add_retention_policy('tool_usage', INTERVAL '6 months');
        END IF;
    END IF;
END $$;

-- Note: audit_log is not a hypertable, so no retention policy needed

-- ============================================================================
-- PERFORMANCE INDEXES: Optimized for TimescaleDB queries
-- ============================================================================

-- Time-based indexes optimized for TimescaleDB (no CONCURRENTLY for hypertables)
CREATE INDEX IF NOT EXISTS idx_outcomes_time_status 
    ON outcomes(execution_time DESC, status) WHERE status != 'success';
CREATE INDEX IF NOT EXISTS idx_outcomes_simulation_time 
    ON outcomes(simulation_id, execution_time DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_scenario_time 
    ON outcomes(scenario_id, execution_time DESC);

-- Modal-specific indexes
CREATE INDEX IF NOT EXISTS idx_outcomes_modal_call 
    ON outcomes(modal_call_id) WHERE modal_call_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_simulations_modal_app 
    ON simulations(modal_app_id, created_at DESC);

-- Configuration and versioning indexes
CREATE INDEX IF NOT EXISTS idx_config_versions_config_version 
    ON config_versions(config_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_config_diffs_llm_generated 
    ON config_diffs(llm_generated, created_at DESC) WHERE llm_generated = true;

-- Vector similarity indexes for recommendations (pgvector)
CREATE INDEX IF NOT EXISTS idx_config_embedding_cosine 
    ON config_versions USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trajectory_embedding_cosine 
    ON outcomes USING ivfflat (trajectory_embedding vector_cosine_ops) 
    WITH (lists = 500) WHERE trajectory_embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_scenario_embedding_cosine 
    ON scenarios USING ivfflat (scenario_embedding vector_cosine_ops) 
    WITH (lists = 50) WHERE scenario_embedding IS NOT NULL;

-- JSONB indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_outcomes_trajectory_gin 
    ON outcomes USING GIN (trajectory) WHERE trajectory IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_config_versions_parsed_gin 
    ON config_versions USING GIN (parsed_config);
CREATE INDEX IF NOT EXISTS idx_recommendations_impact_gin 
    ON recommendations USING GIN (impact_estimate) WHERE impact_estimate IS NOT NULL;

-- Dashboard and analytics indexes
CREATE INDEX IF NOT EXISTS idx_recommendations_pending_confidence 
    ON recommendations(applied, confidence DESC, created_at DESC) 
    WHERE applied = false;
CREATE INDEX IF NOT EXISTS idx_failure_patterns_cluster_time 
    ON failure_patterns(pattern_cluster_id, execution_time DESC) 
    WHERE pattern_cluster_id IS NOT NULL;

-- Simulation status tracking indexes
CREATE INDEX IF NOT EXISTS idx_simulations_status_created 
    ON simulations(status, created_at DESC) WHERE status IN ('pending', 'running');
CREATE INDEX IF NOT EXISTS idx_simulations_scenarios_status 
    ON simulations_scenarios(status, started_at DESC) WHERE status IN ('pending', 'running');

-- User and security indexes
CREATE INDEX IF NOT EXISTS idx_configurations_user_active 
    ON configurations(user_id, is_active, created_at DESC) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_audit_log_table_timestamp 
    ON audit_log(table_name, timestamp DESC);

-- ============================================================================
-- CONTINUOUS AGGREGATES: Real-time dashboard metrics
-- ============================================================================

-- Note: Continuous aggregates will be created separately after initial deployment
-- They require special handling outside of transactions

-- ============================================================================
-- STATISTICS: Optimize query planning
-- ============================================================================

-- Update table statistics for better query planning
ANALYZE configurations;
ANALYZE config_versions;
ANALYZE scenarios;
ANALYZE simulations;
ANALYZE outcomes;
ANALYZE failure_patterns;
ANALYZE tool_usage;
ANALYZE recommendations;
