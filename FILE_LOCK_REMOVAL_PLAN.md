# File Lock System Removal Plan

## Overview
Remove the complex file locking mechanism in favor of atomic file operations and database-first architecture.

## Current Problems
1. File locks causing 15-30 second timeouts
2. Stale lock cleanup is error-prone
3. Lock acquisition blocks concurrent operations unnecessarily
4. Added complexity without clear benefit

## Implementation Plan

### Step 1: Create Atomic File Write Utility
Create a reusable utility for atomic file writes:
- Write to temporary file with `.tmp` suffix
- Use `Path.replace()` for atomic rename
- Handle errors gracefully

### Step 2: Remove Lock Infrastructure
From `arc/cli/utils/state.py`:
- Remove `_file_lock()` context manager
- Remove `_cleanup_stale_lock()` method
- Remove `self.lock_file` attribute
- Remove all lock acquisition/release code

### Step 3: Update File Operations
Update these methods to use atomic writes:
- `_save_config()` - Config file updates
- `save_run()` - Run result saves
- `save_analysis()` - Analysis data saves
- `save_recommendations()` - Recommendation saves
- `save_diff()` - Diff result saves

### Step 4: Simplify Error Handling
- Remove lock timeout exceptions
- Simplify file write error messages
- Keep database error handling separate

### Step 5: Testing
Verify:
- Concurrent `arc run` commands work without conflicts
- File writes complete successfully
- No data corruption with multiple processes
- Database remains primary source of truth

## Benefits
1. **Immediate**: No more timeout errors
2. **Performance**: Faster execution without lock waits
3. **Reliability**: Fewer failure modes
4. **Simplicity**: Cleaner, more maintainable code

## Risk Mitigation
- Atomic operations prevent partial writes
- Database handles true concurrent access
- File system is just local cache/backup
- Existing error handling remains in place