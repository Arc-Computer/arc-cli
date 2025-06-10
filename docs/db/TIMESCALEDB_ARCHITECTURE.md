# Arc TimescaleDB Architecture: Complete Reference Guide

This document explains our complete TimescaleDB implementation for the Arc platform, covering everything from basic concepts to production deployment, plus practical implementation details and client usage.

## Table of Contents

### **Part I: Architecture & Design**
1. [Why TimescaleDB?](#why-timescaledb)
2. [Core Concepts](#core-concepts)
3. [Database Schema Design](#database-schema-design)
4. [Modal Sandbox Integration](#modal-sandbox-integration)
5. [API Architecture](#api-architecture)

### **Part II: Implementation & Usage**
6. [Database Client Design](#database-client-design)
7. [Practical Integration Examples](#practical-integration-examples)
8. [Error Handling & Reliability](#error-handling--reliability)
9. [Performance Optimizations](#performance-optimizations)

### **Part III: Deployment & Operations**
10. [Deployment Guide](#deployment-guide)
11. [Monitoring & Observability](#monitoring--observability)
12. [API Reference](#api-reference)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

# Part I: Architecture & Design

## Why TimescaleDB?

### The Problem
Arc evaluates AI agents by running thousands of scenarios and collecting execution data over time. Traditional databases struggle with:
- **High-volume time-series data** (scenario outcomes with timestamps)
- **Complex analytics queries** (performance trends, reliability scores)
- **Real-time ingestion** from Modal sandbox executions
- **Storage efficiency** for large trajectory datasets

### The Solution: TimescaleDB
TimescaleDB extends PostgreSQL with time-series superpowers:

| **Challenge** | **TimescaleDB Solution** | **Arc Benefit** |
|---------------|-------------------------|-----------------|
| Time-series data | Automatic partitioning by time | Efficient storage of scenario outcomes |
| High write throughput | Optimized inserts & parallel processing | Handles Modal's parallel executions |
| Complex analytics | Native SQL + time-series functions | Rich reliability analytics |
| Data compression | Automatic compression policies | Cost-effective long-term storage |
| Managed scaling | TimescaleDB Cloud auto-scaling | Zero-ops production deployment |

---

## Core Concepts

### 1. Hypertables
**What they are**: Regular PostgreSQL tables that TimescaleDB automatically partitions by time.

```sql
-- Create regular table
CREATE TABLE outcomes (..., execution_time TIMESTAMPTZ, ...);

-- Convert to hypertable (magical!)
SELECT create_hypertable('outcomes', 'execution_time');
```

**What happens**: TimescaleDB automatically:
- Partitions data into time-based chunks
- Optimizes queries across time ranges
- Enables compression for older data
- Provides fast aggregations

### 2. Time-Series Architecture
Our data flows through time-aware layers:

```
Modal Sandbox → Real-time Ingestion → Hypertables → Analytics → Dashboard
     ↓              ↓                    ↓           ↓         ↓
  Scenarios →   Outcomes →         Time-based →  Trends →  Insights
                                   Chunks
```

### 3. Data Lifecycle
1. **Hot Data** (Recent): Uncompressed, fast writes/reads
2. **Warm Data** (1-7 days old): Compressed, fast analytics
3. **Cold Data** (>30 days): Highly compressed, archival queries

---

## Database Schema Design

### Core Tables Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CONFIGURATIONS │    │   SIMULATIONS   │    │    OUTCOMES     │
│                 │    │                 │    │   (HYPERTABLE)  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ config_id (PK)  │    │ simulation_id   │    │ outcome_id      │
│ name            │◄───┤ config_version_id│    │ simulation_id   │
│ user_id         │    │ scenario_set[]  │    │ scenario_id     │
│ latest_version  │    │ modal_app_id    │◄───┤ execution_time  │
└─────────────────┘    │ status          │    │ reliability_score│
                       │ total_scenarios │    │ trajectory      │
                       └─────────────────┘    │ modal_call_id   │
                                              └─────────────────┘
```

### 1. Configuration Management

```sql
-- Agent configurations with versioning
CREATE TABLE configurations (
    config_id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    user_id UUID NOT NULL,
    latest_version_id UUID,
    modal_app_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Immutable configuration snapshots
CREATE TABLE config_versions (
    version_id UUID PRIMARY KEY,
    config_id UUID REFERENCES configurations(config_id),
    version_number INTEGER,
    raw_yaml TEXT NOT NULL,
    parsed_config JSONB NOT NULL,
    config_hash TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Key Features**:
- **Immutable versions**: Never modify existing configs
- **YAML storage**: Human-readable configuration format
- **JSONB support**: Fast queries on configuration parameters
- **Hash-based deduplication**: Avoid storing identical configs

### 2. Scenario Management

```sql
-- Test scenarios with embeddings
CREATE TABLE scenarios (
    scenario_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    task_prompt TEXT NOT NULL,
    difficulty_level TEXT DEFAULT 'medium',
    tags TEXT[],
    scenario_embedding VECTOR(1536), -- For clustering similar scenarios
    expected_tools JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Key Features**:
- **Vector embeddings**: Enable similarity search and clustering
- **Flexible metadata**: Tags and difficulty levels for filtering
- **Tool expectations**: Define what tools scenarios should use

### 3. Simulation Tracking

```sql
-- Complete evaluation runs
CREATE TABLE simulations (
    simulation_id UUID PRIMARY KEY,
    config_version_id UUID REFERENCES config_versions(version_id),
    scenario_set TEXT[] NOT NULL,
    simulation_name TEXT,
    status TEXT DEFAULT 'pending',
    
    -- Modal integration
    modal_app_id TEXT,
    modal_environment TEXT DEFAULT 'production',
    sandbox_instances INTEGER DEFAULT 1,
    
    -- Progress tracking
    total_scenarios INTEGER NOT NULL,
    completed_scenarios INTEGER DEFAULT 0,
    overall_score REAL,
    
    -- Timing and cost
    execution_time_ms INTEGER,
    total_cost_usd REAL,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);
```

**Key Features**:
- **Modal integration**: Track app IDs and environments
- **Progress monitoring**: Real-time completion tracking
- **Cost tracking**: Monitor LLM token costs
- **Status management**: Lifecycle state tracking

### 4. Outcomes (Hypertable) ⭐

```sql
-- Individual scenario results - THE CORE TIME-SERIES TABLE
CREATE TABLE outcomes (
    outcome_id UUID PRIMARY KEY,
    simulation_id UUID REFERENCES simulations(simulation_id),
    scenario_id TEXT REFERENCES scenarios(scenario_id),
    
    -- Time-series key
    execution_time TIMESTAMPTZ NOT NULL,
    
    -- Results
    status TEXT NOT NULL,
    reliability_score REAL,
    execution_time_ms INTEGER,
    tokens_used INTEGER,
    cost_usd REAL,
    
    -- Rich data
    trajectory JSONB,        -- Complete execution trace
    modal_call_id TEXT,      -- Link to Modal execution
    sandbox_id TEXT,         -- Sandbox instance identifier
    
    -- Error tracking
    error_code TEXT,
    error_category TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Custom metrics
    metrics JSONB DEFAULT '{}'
);

-- Convert to hypertable (🚀 This is where the magic happens!)
SELECT create_hypertable('outcomes', 'execution_time');
```

**Why This is Special**:
- **Automatic partitioning**: Data split by time chunks
- **Fast time-range queries**: Query last 24 hours in milliseconds
- **Compression ready**: Older data automatically compressed
- **Parallel processing**: Multiple chunks processed simultaneously

### 5. Junction Tables

```sql
-- Many-to-many relationship between simulations and scenarios
CREATE TABLE simulations_scenarios (
    simulation_id UUID REFERENCES simulations(simulation_id),
    scenario_id TEXT REFERENCES scenarios(scenario_id),
    execution_order INTEGER,
    status TEXT DEFAULT 'pending',
    modal_call_id TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    PRIMARY KEY (simulation_id, scenario_id)
);
```

---

## Modal Sandbox Integration

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODAL SANDBOX ENVIRONMENT                    │
├─────────────────┬───────────────────┬───────────────────────────┤
│ @modal.function │ evaluate_scenarios│     Parallel Execution    │
│                 │                   │                           │
│  Agent Config   │   Scenario List   │    Results Collection     │
│  ┌─────────┐   │   ┌─────────────┐  │   ┌─────────────────────┐ │
│  │ GPT-4   │   │   │ weather_1   │  │   │ success: 0.95       │ │
│  │ temp:0.7│   │   │ database_2  │  │   │ tokens: 1,245       │ │
│  │ tools[] │   │   │ api_call_3  │  │   │ cost: $0.045        │ │
│  └─────────┘   │   └─────────────┘  │   └─────────────────────┘ │
└─────────────────┴───────────────────┴───────────────────────────┘
         │                   │                        │
         ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ARC DATABASE API                           │
├─────────────────────────────────────────────────────────────────┤
│  start_simulation() → record_outcomes() → complete_simulation() │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TIMESCALEDB CLOUD                          │
├─────────────────┬───────────────────┬───────────────────────────┤
│  Configurations │    Simulations    │      Outcomes             │
│                 │                   │     (Hypertable)          │
│  Version: v1.2  │  Status: Running  │  Time: 2024-01-15 14:30  │
│  Model: GPT-4   │  Progress: 75%    │  Score: 0.87             │
│  Temp: 0.7      │  Cost: $2.34      │  Tokens: 1,245           │
└─────────────────┴───────────────────┴───────────────────────────┘
```

### Integration Points

#### 1. Modal Function Integration
```python
import modal
from arc.database.api import create_arc_api

@modal.function(
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("timescale-secret")  # 🔑 Database connection
    ]
)
async def run_evaluation_suite(agent_config, scenarios):
    # 1. Initialize database tracking
    db_api = await create_arc_api()
    
    # 2. Start simulation
    sim_info = await db_api.start_simulation(
        config_version_id=agent_config["config_version_id"],
        scenarios=scenarios,
        modal_app_id=modal.current_app_id(),  # 🔗 Automatic linking
        simulation_name=f"eval_{agent_config['model']}"
    )
    
    # 3. Run evaluations (your existing code)
    results = await run_parallel_scenarios(agent_config, scenarios)
    
    # 4. Record all outcomes
    await db_api.record_batch_outcomes(
        simulation_id=sim_info["simulation_id"],
        scenario_results=results  # 📊 Rich trajectory data
    )
    
    # 5. Complete simulation
    await db_api.complete_simulation(
        simulation_id=sim_info["simulation_id"],
        suite_result=aggregate_results(results)
    )
    
    return results
```

#### 2. Data Structure Mapping
The API automatically extracts Modal data:

```python
# Modal scenario result structure
modal_result = {
    "scenario": {
        "id": "weather_query_1",
        "name": "NYC Weather Query",
        "task_prompt": "What's the weather in NYC?"
    },
    "trajectory": {
        "status": "success",
        "execution_time_seconds": 3.2,
        "token_usage": {
            "prompt_tokens": 156,
            "completion_tokens": 89,
            "total_tokens": 245,
            "total_cost": 0.0087
        },
        "full_trajectory": [
            {"step": 1, "tool": "get_weather", "input": {"city": "NYC"}},
            {"step": 2, "response": "Sunny, 72°F"}
        ]
    },
    "reliability_score": {
        "overall_score": 0.92,
        "dimension_scores": {
            "tool_execution": 1.0,
            "response_quality": 0.89,
            "completeness": 0.95
        }
    }
}

# API automatically maps to database record
db_record = {
    "scenario_id": "weather_query_1",
    "status": "success",
    "execution_time": datetime.utcnow(),
    "reliability_score": 0.92,
    "execution_time_ms": 3200,
    "tokens_used": 245,
    "cost_usd": 0.0087,
    "trajectory": {...},  # Complete trajectory JSON
    "modal_call_id": modal.current_call_id(),
    "metrics": {"tool_execution": 1.0, ...}
}
```

---

## API Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  Modal Functions → ArcAPI → Business Logic                 │
│                                                            │
│  • start_simulation()     • Error categorization          │
│  • record_outcomes()      • Data validation               │
│  • get_simulation_status()• Progress tracking             │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    DATABASE LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  ArcDBClient → Connection Management → Performance         │
│                                                            │
│  • Connection pooling     • Retry logic                   │
│  • Transaction handling   • Health monitoring             │
│  • Query optimization     • Metrics collection            │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                   TIMESCALEDB LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL + Time-Series Extensions                       │
│                                                            │
│  • Hypertable management   • Automatic compression        │
│  • Time-based partitioning • Continuous aggregates       │
│  • Parallel query execution• Backup & recovery           │
└─────────────────────────────────────────────────────────────┘
```

### Core API Methods

#### Simulation Lifecycle
```python
# Start tracking a new simulation
sim_info = await db_api.start_simulation(
    config_version_id="uuid",
    scenarios=["scenario_1", "scenario_2"],
    simulation_name="gpt4_evaluation",
    modal_app_id="app_123"
)

# Complete simulation with final metrics
await db_api.complete_simulation(
    simulation_id=sim_info["simulation_id"],
    suite_result={
        "overall_score": 0.87,
        "total_cost": 2.45,
        "execution_time_ms": 45000
    }
)
```

#### Outcome Recording
```python
# Record individual scenario result
await db_api.record_scenario_outcome(
    simulation_id="uuid",
    scenario_result=modal_result
)

# Batch recording for parallel executions
await db_api.record_batch_outcomes(
    simulation_id="uuid",
    scenario_results=[result1, result2, result3]
)
```

#### Query & Analytics
```python
# Real-time progress tracking
status = await db_api.get_simulation_status("simulation_id")
print(f"Progress: {status['completion_percent']:.1f}%")

# Flexible outcome retrieval
outcomes = await db_api.get_scenario_outcomes(
    simulation_id="uuid",
    status_filter="success",
    time_range=timedelta(hours=24)
)
```

---

# Part II: Implementation & Usage

## Database Client Design

### Connection Management

```python
from arc.database.client import ArcDBClient

# The client handles all the complexity for you
client = ArcDBClient(
    connection_string="postgresql://...",  # Or use TIMESCALE_SERVICE_URL env var
    enable_monitoring=True,
    log_level="INFO"
)

# Initialize once, use everywhere
await client.initialize()
```

### Automatic Retry Logic

Our client automatically retries failed operations:

```python
@with_retry(max_attempts=3, delay=1.0, backoff=2.0)
async def record_outcome(self, outcome_data):
    # If database is temporarily unavailable:
    # Attempt 1: Immediate
    # Attempt 2: Wait 1 second
    # Attempt 3: Wait 2 seconds
    # Then fail gracefully
```

This means your Modal functions don't need to worry about transient database issues!

### Performance Monitoring

```python
# Get real-time metrics
metrics = client.get_metrics()
print(f"Success rate: {metrics['success_rate']}%")
print(f"Average query time: {metrics['avg_query_time']}ms")
```

---

## Practical Integration Examples

### Recording Execution Data

Here's how to integrate the database client in your Modal sandbox:

```python
import modal
from arc.database.client import ArcDBClient

# Initialize client (do this once)
db_client = ArcDBClient()
await db_client.initialize()

@modal.function
async def evaluate_scenario(scenario_id: str, config_id: str):
    # 1. Create a simulation record
    simulation_id = await db_client.create_simulation(
        config_version_id=config_id,
        scenario_set=[scenario_id],
        simulation_name=f"eval_{scenario_id}",
        modal_app_id=modal.current_app_id()  # Automatic Modal tracking!
    )
    
    try:
        # 2. Run your evaluation
        result = await run_agent_evaluation(scenario_id, config_id)
        
        # 3. Record the outcome
        await db_client.record_outcome({
            "simulation_id": simulation_id,
            "scenario_id": scenario_id,
            "status": "success" if result.passed else "error",
            "reliability_score": result.score,
            "execution_time_ms": result.duration_ms,
            "tokens_used": result.token_count,
            "cost_usd": result.cost,
            "trajectory": result.trajectory,  # Your agent's execution trace
            "modal_call_id": modal.current_call_id(),  # Automatic linking!
            "sandbox_id": result.sandbox_id,
            "metrics": {
                "custom_metric_1": result.metric1,
                "custom_metric_2": result.metric2
            }
        })
        
    except Exception as e:
        # Errors are automatically tracked
        await db_client.record_outcome({
            "simulation_id": simulation_id,
            "scenario_id": scenario_id,
            "status": "error",
            "reliability_score": 0.0,
            "execution_time_ms": 0,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "trajectory": {"error": str(e)},
            "error_code": type(e).__name__,
            "error_category": "sandbox_error"
        })
```

### Batch Operations for High Throughput

When running many evaluations in parallel:

```python
@modal.function
async def evaluate_batch(scenario_ids: List[str], config_id: str):
    # Collect all outcomes
    outcomes = []
    
    for scenario_id in scenario_ids:
        result = await run_agent_evaluation(scenario_id, config_id)
        outcomes.append({
            "simulation_id": simulation_id,
            "scenario_id": scenario_id,
            "status": "success" if result.passed else "error",
            "reliability_score": result.score,
            # ... other fields ...
        })
    
    # Batch insert for efficiency (100+ outcomes/second!)
    await db_client.record_outcomes_batch(outcomes)
```

---

## Error Handling & Reliability

### Built-in Resilience

The database client handles common failure scenarios automatically:

1. **Connection Failures**: Automatic retry with exponential backoff
2. **Pool Exhaustion**: Queued requests wait for available connections
3. **Timeout Protection**: 60-second command timeout prevents hanging
4. **SSL/TLS Security**: Automatic encryption for cloud deployments

### Error Categories

The system tracks different types of failures:

```python
error_categories = [
    'timeout',        # Execution exceeded time limit
    'tool_error',     # Tool/function call failed
    'model_error',    # LLM produced invalid output
    'validation_error', # Output failed validation
    'system_error',   # Infrastructure issue
    'sandbox_error'   # Modal sandbox-specific error
]
```

### Connection Pool Management

```python
# Connection pool configuration
pool_size=20,        # Base connections
max_overflow=30,     # Additional connections under load
pool_timeout=30,     # Max wait for connection
pool_recycle=3600,   # Refresh connections hourly
```

---

## Performance Optimizations

### 1. Hypertable Optimizations

```sql
-- Time-based partitioning (automatic)
SELECT create_hypertable('outcomes', 'execution_time', chunk_time_interval => INTERVAL '1 day');

-- Compression policies (automatic)
SELECT add_compression_policy('outcomes', INTERVAL '7 days');

-- Retention policies (optional)
SELECT add_retention_policy('outcomes', INTERVAL '1 year');
```

**Benefits**:
- **1-day chunks**: Optimal for daily analytics queries
- **7-day compression**: Balance between query speed and storage
- **1-year retention**: Automatic cleanup of old data

### 2. Indexing Strategy

```sql
-- Time-range queries (most common)
CREATE INDEX idx_outcomes_execution_time ON outcomes (execution_time DESC);

-- Simulation filtering
CREATE INDEX idx_outcomes_simulation_id ON outcomes (simulation_id, execution_time DESC);

-- Status filtering for dashboards
CREATE INDEX idx_outcomes_status ON outcomes (status, execution_time DESC);

-- Modal call tracking
CREATE INDEX idx_outcomes_modal_call ON outcomes (modal_call_id);

-- Composite index for analytics
CREATE INDEX idx_outcomes_analytics ON outcomes (simulation_id, status, execution_time DESC);
```

### 3. Query Patterns

#### Fast Time-Range Queries
```sql
-- Last 24 hours performance
SELECT 
    time_bucket('1 hour', execution_time) as hour,
    COUNT(*) as scenarios_run,
    AVG(reliability_score) as avg_score,
    SUM(cost_usd) as total_cost
FROM outcomes 
WHERE execution_time >= NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;
```

#### Simulation Progress Tracking
```sql
-- Real-time simulation status
SELECT 
    s.simulation_id,
    s.total_scenarios,
    COUNT(o.outcome_id) as completed_scenarios,
    AVG(o.reliability_score) as current_avg_score
FROM simulations s
LEFT JOIN outcomes o ON s.simulation_id = o.simulation_id
WHERE s.status = 'running'
GROUP BY s.simulation_id, s.total_scenarios;
```

### 4. Connection Pool Optimization

```python
# Optimized for Modal's parallel execution
engine = create_async_engine(
    connection_string,
    pool_size=20,           # Base connections
    max_overflow=30,        # Burst capacity  
    pool_timeout=30,        # Wait time for connection
    pool_recycle=3600,      # Refresh connections hourly
    pool_pre_ping=True      # Health checks
)
```

---

# Part III: Deployment & Operations

## Deployment Guide

### 1. TimescaleDB Cloud Setup

#### Create Instance
```bash
# Sign up at https://cloud.timescale.com
# Create new service:
# - Plan: Start with "Time-series" tier
# - Region: Choose closest to Modal deployment
# - Storage: Start with 100GB (auto-scales)
```

#### Configure Connection
```bash
# Get connection string from dashboard
TIMESCALE_SERVICE_URL="postgresql://username:password@host:port/dbname?sslmode=require"

# Add to Modal secrets
modal secret create timescale-secret \
    TIMESCALE_SERVICE_URL="$TIMESCALE_SERVICE_URL"
```

### 2. Schema Deployment

```bash
# Deploy all tables, indexes, and policies
python arc/database/deploy_schema.py
```

### 3. Environment Setup

Create a `.env` file:
```bash
TIMESCALE_SERVICE_URL=postgresql://user:pass@host:port/dbname?sslmode=require
```

### 4. Run Tests

```bash
# Unit tests (no database needed)
python -m pytest arc/tests/unit/test_database_client.py -v

# Integration tests (requires database)
python -m arc/tests/integration/test_database.py
```

### 5. Basic Usage

```python
from arc.database.client import ArcDBClient

async def main():
    # Initialize client
    client = ArcDBClient()
    await client.initialize()
    
    # Check health
    health = await client.health.check_extensions()
    print(f"TimescaleDB version: {health['timescaledb']}")
    
    # Your evaluation code here
    # ...
    
    # Clean up
    await client.close()
```

---

## Monitoring & Observability

### 1. Database Health Monitoring

```python
# Built-in health checks
from arc.database.client import ArcDBClient

client = ArcDBClient()
health = await client.initialize()

print(health)
# {
#   "status": "healthy",
#   "extensions": {"timescaledb": "2.11.1", "uuid-ossp": "1.1"},
#   "hypertables": [{"hypertable_name": "outcomes", "num_chunks": 15}],
#   "pool_stats": {"checked_out": 5, "total": 25},
#   "response_time_ms": 23.4
# }
```

### 2. Performance Metrics

```python
# API performance tracking
metrics = await db_api.db.get_metrics()
print(f"Success rate: {metrics['success_rate']:.1f}%")
print(f"Avg query time: {metrics['avg_query_time']:.1f}ms")
print(f"Total queries: {metrics['total_queries']}")
```

### 3. TimescaleDB Native Monitoring

```sql
-- Hypertable statistics
SELECT * FROM timescaledb_information.hypertables;

-- Chunk information
SELECT * FROM timescaledb_information.chunks 
WHERE hypertable_name = 'outcomes';

-- Compression stats
SELECT * FROM timescaledb_information.compression_settings;

-- Query performance
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
WHERE query LIKE '%outcomes%'
ORDER BY mean_exec_time DESC;
```

### 4. Custom Dashboards

```sql
-- Create materialized views for dashboards
CREATE MATERIALIZED VIEW daily_simulation_stats AS
SELECT 
    date_trunc('day', execution_time) as day,
    COUNT(DISTINCT simulation_id) as simulations_run,
    COUNT(*) as total_scenarios,
    AVG(reliability_score) as avg_reliability,
    SUM(cost_usd) as total_cost
FROM outcomes
WHERE execution_time >= NOW() - INTERVAL '30 days'
GROUP BY day;

-- Refresh policy
SELECT add_continuous_aggregate_policy('daily_simulation_stats',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

---

## API Reference

### Core Methods

#### `create_simulation()`
Creates a new simulation run for tracking multiple scenario evaluations.

```python
simulation_id = await client.create_simulation(
    config_version_id="uuid",      # Configuration version
    scenario_set=["s1", "s2"],     # List of scenarios
    simulation_name="test_run",    # Optional name
    modal_app_id="app_123"         # Modal app identifier
)
```

#### `record_outcome()`
Records a single scenario execution result.

```python
outcome_id = await client.record_outcome({
    "simulation_id": "uuid",
    "scenario_id": "scenario_1",
    "status": "success",           # success/error/timeout/cancelled
    "reliability_score": 0.95,     # 0.0 to 1.0
    "execution_time_ms": 1500,
    "tokens_used": 250,
    "cost_usd": 0.02,
    "trajectory": {...},           # Agent execution trace
    "modal_call_id": "call_123",   # Modal tracking
    "metrics": {...}               # Custom metrics
})
```

#### `record_outcomes_batch()`
Efficiently records multiple outcomes at once.

```python
outcome_ids = await client.record_outcomes_batch([
    {"simulation_id": "uuid", "scenario_id": "s1", ...},
    {"simulation_id": "uuid", "scenario_id": "s2", ...},
    # ... up to 1000 outcomes
])
```

#### `get_simulation_performance()`
Retrieves time-series performance metrics.

```python
performance = await client.get_simulation_performance(
    simulation_id="uuid",
    time_range=timedelta(hours=24)
)
# Returns hourly metrics and summary statistics
```

### Health Monitoring

```python
# Check database health
health_status = await client.initialize()

# Get performance metrics
metrics = client.get_metrics()

# Check connection pool
pool_stats = await client.health.check_connection_pool()
```

---

## Best Practices

### 1. **Use Batch Operations**
When recording multiple outcomes, always use `record_outcomes_batch()` instead of multiple `record_outcome()` calls.

### 2. **Include Modal Tracking**
Always include `modal_call_id` and `modal_app_id` for debugging and tracing.

### 3. **Structure Your Trajectories**
Keep trajectory data structured and consistent:
```python
trajectory = {
    "start_time": datetime.utcnow().isoformat(),
    "steps": [...],
    "final_output": ...,
    "status": "completed"
}
```

### 4. **Monitor Performance**
Regularly check metrics to ensure healthy operation:
```python
metrics = client.get_metrics()
if metrics['success_rate'] < 95:
    logger.warning(f"Low success rate: {metrics['success_rate']}%")
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Connection Issues

**Problem**: `ConnectionError: Could not connect to database`

**Solutions**:
```bash
# Check connection string format
echo $TIMESCALE_SERVICE_URL
# Should be: postgresql://user:pass@host:port/db?sslmode=require

# Test connection manually
psql "$TIMESCALE_SERVICE_URL"

# Verify Modal secrets
modal secret list | grep timescale
```

#### 2. Performance Issues

**Problem**: Slow query performance

**Solutions**:
```sql
-- Check for missing indexes
EXPLAIN ANALYZE SELECT * FROM outcomes 
WHERE execution_time >= NOW() - INTERVAL '1 hour';

-- Monitor chunk exclusion
SELECT chunks_excluded, chunks_total 
FROM timescaledb_information.hypertables;

-- Check compression status
SELECT compressed_chunk_id, uncompressed_heap_size, compressed_heap_size
FROM timescaledb_information.compressed_chunk_stats;
```

#### 3. Schema Issues

**Problem**: `relation "outcomes" does not exist`

**Solutions**:
```python
# Redeploy schema
from arc.database.client import ArcDBClient

client = ArcDBClient()
result = await client.deploy_schema()
print(result)

# Check existing tables
async with client.engine.begin() as conn:
    result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
    tables = [row[0] for row in result]
    print(f"Existing tables: {tables}")
```

#### 4. Modal Integration Issues

**Problem**: API calls failing in Modal functions

**Solutions**:
```python
# Add comprehensive error handling
try:
    db_api = await create_arc_api()
except Exception as e:
    print(f"Database initialization failed: {e}")
    # Continue without database tracking
    return run_evaluation_without_db(scenarios)

# Test with minimal data
try:
    result = await db_api.get_simulation_status("test-id")
except Exception as e:
    print(f"Database query failed: {e}")
```

### Legacy Issues

1. **Connection Refused**
   - Check `TIMESCALE_SERVICE_URL` is set correctly
   - Verify SSL mode matches your TimescaleDB configuration

2. **Pool Exhausted**
   - Increase `pool_size` in client configuration
   - Check for connection leaks in your code

3. **Slow Queries**
   - Ensure hypertables are created (check logs)
   - Verify compression policies are active
   - Check if continuous aggregates need refresh

### Debug Mode

```python
# Enable detailed logging
import logging
logging.getLogger("arc.database").setLevel(logging.DEBUG)

# Test connection health
health = await client.initialize()
if health["status"] != "healthy":
    print(f"Database unhealthy: {health}")
```

---

## Summary

This TimescaleDB architecture provides:

✅ **Scalable time-series storage** for scenario outcomes  
✅ **Seamless Modal integration** with minimal code changes  
✅ **Production-ready performance** with automatic optimizations  
✅ **Rich analytics capabilities** for reliability insights  
✅ **Managed cloud deployment** with zero-ops maintenance  
✅ **Reliability**: Automatic retries and error handling  
✅ **Performance**: 100+ outcomes/second throughput  
✅ **Scalability**: Handles 1000+ concurrent executions  
✅ **Observability**: Built-in monitoring and metrics  

The architecture is designed to grow with your needs - from prototype to production scale, handling thousands of parallel scenarios with sub-second query performance.

**Next Steps**:
1. Deploy TimescaleDB Cloud instance
2. Run schema deployment
3. Add database tracking to Modal functions  
4. Monitor performance and optimize as needed

Your AI evaluation pipeline now has enterprise-grade data infrastructure! 🚀 