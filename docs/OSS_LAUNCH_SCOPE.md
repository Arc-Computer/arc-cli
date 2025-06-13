# Arc OSS Launch Scope

## Core Approach: Clean OSS Package with Abstraction Layer

### Implementation Plan
**Effort**: Medium-High
- Create execution abstraction (Modal → Local subprocess/Docker)
- Create storage abstraction (TimescaleDB → SQLite)
- Build extraction script to create clean OSS package
- Implement basic local execution engine

**Pros**: Clean separation, sustainable architecture
**Cons**: More upfront work, but worth it for clean OSS offering

## Implementation Details

### Core OSS Package v1.0
1. **All** analysis algorithms (`arc/analysis/`, `arc/recommendations/`)
2. **All** scenarios (`arc/scenarios/`, `arc/database/seed/failure_patterns/`)
3. **All** CLI commands working locally
4. Local execution (subprocess initially, Docker optional)
5. Local storage (SQLite)
6. Clean abstractions for execution and storage backends

### OSS Enhancements (v1.1+)
- Local web UI (localhost:8080)
- Dagger integration for container orchestration
- Parallel execution (10-20 containers)
- Advanced caching and performance optimization
- Model adapters for all providers
- CI/CD templates and integrations

### Cloud-Only Features (clear differentiation)
- Modal distributed execution (1000x speed)
- TimescaleDB analytics
- Team collaboration
- Compliance features
- Cross-org insights

## Technical Architecture

### Abstraction Layers to Build

1. **Execution Backend Interface**
   ```python
   # arc/core/interfaces/execution.py
   class ExecutionBackend(ABC):
       async def execute_scenario(self, scenario: Scenario) -> Result
   ```
   - LocalBackend (subprocess) - v1.0
   - DaggerBackend (containers) - v1.1
   - ModalBackend (cloud) - proprietary

2. **Storage Backend Interface**
   ```python
   # arc/core/interfaces/storage.py
   class StorageBackend(ABC):
       async def save_results(self, results: List[Result])
       async def query_results(self, filters: Dict) -> List[Result]
   ```
   - SQLiteBackend - v1.0
   - TimescaleBackend - proprietary

3. **Build Script**
   - Extract clean OSS package without proprietary deps
   - Automated process to maintain single codebase

## Proposed Timeline

**Week 1**: 
- Build abstraction interfaces
- Create extraction/build script
- Implement LocalBackend

**Week 2**: 
- Implement SQLiteBackend
- Test full OSS package
- Ensure all algorithms work locally

**Week 3**: 
- Documentation and examples
- Performance optimization
- Package and release

## Success Criteria
- Developer can run `pip install arc` and test an agent in <5 minutes
- Clean OSS package with no proprietary dependencies
- All core features work locally (analysis, recommendations, scenarios)
- Clear upgrade path to Arc Cloud for speed/scale