"""Project-wide constant definitions."""

__all__: list[str] = [
    "DEFAULT_TIMEOUT_MS",
    "DEFAULT_QUALITY_THRESHOLD",
    "MAX_PARALLEL_CONTAINERS",
    "DEFAULT_BATCH_SIZE"
]

# Timeouts
DEFAULT_TIMEOUT_MS: int = 300_000  # 5 minutes

# Quality thresholds
DEFAULT_QUALITY_THRESHOLD: float = 3.0  # Minimum quality score for scenarios

# Modal configuration
MAX_PARALLEL_CONTAINERS: int = 50  # Maximum parallel Modal containers

# Batch processing
DEFAULT_BATCH_SIZE: int = 20  # Default batch size for scenario generation
