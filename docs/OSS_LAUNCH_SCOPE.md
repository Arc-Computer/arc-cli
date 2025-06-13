# Arc OSS Launch Scope

## Core Question: What's the minimum refactoring needed to launch OSS?

### Option 1: Full Abstraction (3 weeks)
**Effort**: High
- Abstract Modal → Docker execution 
- Abstract TimescaleDB → SQLite
- Create plugin architecture

**Pros**: Clean separation, future-proof
**Cons**: Delays launch, more testing needed

### Option 2: Feature Flags (1 week) 
**Effort**: Low
- Keep single codebase
- Add `--cloud` flag to CLI commands
- Disable Modal/DB imports when not authenticated
- Ship with "local mode" limitations

**Pros**: Fast to ship, maintain one codebase
**Cons**: Some features simply won't work locally


## Recommendation: Option 2 with these specifics

### Must Have for OSS v1
1. **All** analysis algorithms (`arc/analysis/`, `arc/recommendations/`)
2. **All** scenarios (`arc/scenarios/`, `arc/database/seed/failure_patterns/`)
3. **All** CLI commands working locally
4. Local execution (subprocess/Docker)
5. Local storage (SQLite)

### Nice to Have (can ship v1.1)
- Local web UI
- Parallel execution 
- Plugin system
- Advanced caching

### Cloud-Only Features (clear differentiation)
- Modal distributed execution (1000x speed)
- TimescaleDB analytics
- Team collaboration
- Compliance features
- Cross-org insights

## Key Technical Decisions Needed

1. **Execution approach for OSS?**
   - A) Subprocess (simple, no deps)
   - B) Docker required (better isolation)
   - C) Both with graceful degradation

2. **Storage for OSS?**
   - A) SQLite only
   - B) JSON files  
   - C) Pluggable (SQLite default)

3. **How to handle Modal imports?**
   - A) Try/except with fallback
   - B) Build script strips them
   - C) Feature flags

4. **Web UI priority?**
   - A) Ship CLI only first
   - B) Basic localhost:8080 required
   - C) Full UI from day 1

## Proposed Timeline

**Week 1**: Decisions + extraction script
**Week 2**: Implement + test
**Week 3**: Polish + launch

## Success Criteria
- Developer can run `pip install arc` and test an agent in <5 minutes
- No Modal/proprietary deps in OSS package