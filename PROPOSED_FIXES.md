# Proposed Fixes for Root Issues

## 1. Remove File Lock System

The file lock system is overly complex and causes more problems than it solves. Modern approaches:

### Option A: Atomic File Operations (Recommended)
```python
def save_run(self, result: RunResult) -> Path:
    """Save run results using atomic file operations."""
    run_dir = self.runs_dir / result.run_id
    run_dir.mkdir(exist_ok=True)
    
    # Write to temp files first
    for filename, data in [
        ("result.json", result.to_dict()),
        ("scenarios.json", result.scenarios),
        ("results.json", result.results),
    ]:
        temp_file = run_dir / f".{filename}.tmp"
        final_file = run_dir / filename
        
        # Write atomically
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename (POSIX guarantees atomicity)
        temp_file.replace(final_file)
    
    # Update config last (least critical)
    self.config["last_run_id"] = result.run_id
    self._save_config()
    
    return run_dir
```

### Option B: Database-Only for Critical State
- Use database for all shared state
- Keep file system for read-only cached data
- No locks needed

## 2. Fix Modal Auth Testing

The Modal auth check is too aggressive. The actual execution works, but the test fails.

### Current Problem:
```python
# This test creates unnecessary Modal app instances
modal.App("arc-test-connection")  # Fails with stale tokens
```

### Fix:
```python
async def _check_modal_available(max_retries: int = 3) -> bool:
    """Check if Modal is installed and configured."""
    try:
        import modal
        
        # Just check if we can import and authenticate
        auth_configured = setup_modal_auth()
        if not auth_configured:
            console.print(format_warning("Modal not authenticated"))
            console.print(get_modal_auth_instructions())
            return False
            
        # Don't test connection - let actual execution handle it
        return True
        
    except ImportError:
        console.print(
            format_warning("Modal not installed. Run 'pip install modal' to install")
        )
        return False
```

## 3. Fix Tool Parameter Mismatch

The real issue in the Modal logs is that our tool behavior engine needs to handle missing parameters:

### Current spreadsheet_analyzer error:
```
Field required [type=missing, input_value={'file_path': 'empty_spreadsheet.xlsx'}, input_type=dict]
```

### Fix in ToolBehaviorEngine:
```python
def _create_tool_from_definition(self, tool_def: Dict[str, Any]) -> Any:
    """Create a tool from its definition with proper error handling."""
    
    @tool
    def dynamic_tool(**kwargs):
        # Add default values for missing parameters
        for param in tool_def.get("parameters", []):
            if param not in kwargs:
                # Provide sensible defaults
                if param == "sheet_name":
                    kwargs[param] = "Sheet1"
                elif param == "timeout":
                    kwargs[param] = 30
                # etc...
        
        # Rest of implementation...
```

## Summary

1. **File Locks**: Remove entirely, use atomic file operations
2. **Modal Auth**: Simplify check, don't create test apps
3. **Tool Parameters**: Add smart defaults for missing required parameters

These changes will make Arc more robust and fix the root causes rather than patching symptoms.