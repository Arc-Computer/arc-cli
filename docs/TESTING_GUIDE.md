# Arc Testing Guide

This guide provides instructions to test the complete Arc functionality: **Database → API → Sandbox Integration**.

## 🎯 System Overview

The Arc system consists of:
1. **TimescaleDB Database** - Time-series data storage with hypertables
2. **ArcAPI** - High-level Python API for database operations  
3. **FastAPI Server** - REST API endpoints
4. **Modal Sandbox Integration** - Simulation execution environment

## 📊 Database Schema

**Total Tables: 12**
- **9 Regular Tables**: configurations, config_versions, simulations, scenarios, simulations_scenarios, audit_log, config_diffs, llm_recommendation_sessions, recommendations
- **3 Hypertables**: outcomes, failure_patterns, tool_usage

## 📋 Prerequisites

### 1. Environment Setup

**Create a `.env` file in your project root:**
```bash
# .env file
TIMESCALE_SERVICE_URL=your_timescaledb_url
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

**Install required dependencies:**
```bash
pip install httpx asyncpg sqlalchemy[asyncio] fastapi uvicorn python-dotenv pyyaml
```

### 2. Verify Database Access
```bash
# Test database connectivity
psql -h your_host -U tsdbadmin -d tsdb -c "SELECT COUNT(*) FROM timescaledb_information.hypertables;"
```

You should see **3 hypertables**: `outcomes`, `failure_patterns`, `tool_usage`.

## 🧪 Testing Strategy

### Test Structure
```
arc/tests/
├── e2e/
│   ├── test_complete_pipeline.py    # Main end-to-end test
│   └── database_inspection.py       # Schema and data inspection
├── integration/                     # Integration tests  
└── unit/                           # Unit tests
```

## 🚀 Main Test: Complete Pipeline

**Location**: `arc/tests/e2e/test_complete_pipeline.py`

**What it tests:**
- Complete pipeline with real Modal execution
- Agent configuration parsing and normalization
- Scenario generation using Arc's scenario generator
- Modal sandbox integration (real execution)
- Database operations (configuration → simulation → outcomes)
- TimescaleDB hypertable data storage
- End-to-end workflow validation

**Run the test:**
```bash
cd arc/tests/e2e
python test_complete_pipeline.py
```

**Expected output:**
```
🚀 Starting E2E Pipeline Test
📋 Setting up test environment...
✅ Database API created
✅ Modal authentication verified
📄 Loading config: examples/configs/minimal_agent.yaml
✅ Config normalized: gpt-4.1
📝 Generating scenarios...
✅ Generated 2 scenarios
🔥 Executing scenarios with Modal...
✅ Modal execution completed: 2 results
   - Execution time: 15.34s
   - Total cost: $0.0234
💾 Testing database operations...
✅ Configuration created: abc-123-def
✅ Simulation started: sim-789-abc
📊 Recording outcomes...
✅ Recorded outcome: outcome-1
✅ Recorded outcome: outcome-2
✅ Simulation completed: completed
🎉 E2E Pipeline Test PASSED
```

## 🔍 Database Inspection

**Location**: `arc/tests/e2e/database_inspection.py`

**Use this to inspect test results:**
```bash
cd arc/tests/e2e
python database_inspection.py
```

**What it shows:**
- Complete schema overview (9 tables + 3 hypertables)
- Recent data in all tables
- TimescaleDB hypertable details and chunks
- Time-series data distribution
- Summary statistics

## 📊 Verification Checklist

### ✅ Database Layer
- [ ] TimescaleDB connection established
- [ ] All 12 tables present (9 regular + 3 hypertables)
- [ ] Configuration creation works
- [ ] Simulation lifecycle works (create → execute → record → complete)
- [ ] Hypertable data storage works
- [ ] Time-series queries work

### ✅ Integration Layer  
- [ ] Agent config parsing works
- [ ] Scenario generation works
- [ ] Modal execution works
- [ ] Database API integration works
- [ ] End-to-end pipeline completes successfully

### ✅ Production Readiness
- [ ] All functions use production API calls (no custom test code)
- [ ] Real data flows through the system
- [ ] Error handling works properly
- [ ] Performance metrics are captured
- [ ] Cost tracking works

## 🛠️ Additional Testing

### API Server Testing
```bash
# Start the API server
arc-server

# Test endpoints (if needed)
python test_api_endpoints.py

# View API docs
open http://localhost:8000/docs
```

### Scenario Generation Testing
```bash
python test_scenario_generation.py
```

## 🎯 Success Criteria

A successful test run should show:
1. ✅ Modal authentication works
2. ✅ Agent config loads and normalizes
3. ✅ Scenarios generate successfully
4. ✅ Modal execution completes
5. ✅ Database operations succeed
6. ✅ Data appears in all relevant tables
7. ✅ Simulation completes with status "completed"
8. ✅ No errors in any step

## 🐛 Troubleshooting

### Common Issues

**Modal Authentication Failed**
```bash
# Ensure Modal is installed and authenticated
pip install modal
modal token new
```

**Database Connection Failed**
```bash
# Check your .env file has the correct TIMESCALE_SERVICE_URL
# Test connection manually with psql
```

**Config File Not Found**
```bash
# Ensure you have example configs
# Or create a minimal config in examples/configs/minimal_agent.yaml
```

**Scenario Generation Failed**  
```bash
# Check your OpenAI/OpenRouter API key is valid
# Ensure you have sufficient API credits
```

## 📈 Performance Expectations

- **Scenario Generation**: 2-5 scenarios in 5-10 seconds
- **Modal Execution**: 10-30 seconds for 2-5 scenarios  
- **Database Operations**: < 1 second per operation
- **Total E2E Time**: 30-60 seconds

## 🎉 Success!

If all tests pass, your Arc system is ready for production use with:
- ✅ Complete database integration
- ✅ Real scenario generation
- ✅ Modal sandbox execution
- ✅ Proper data storage and retrieval
- ✅ End-to-end workflow validation 