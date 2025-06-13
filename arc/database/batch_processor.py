"""
Batch Processing System for Arc CLI Database Operations

Implements high-throughput batch processing for Modal sandbox executions
to handle concurrent scenario runs efficiently.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .client import ArcDBClient
from .api import ArcAPI

logger = logging.getLogger(__name__)


class ResourceLimitError(Exception):
    """Raised when batch processing resource limits are exceeded."""
    pass


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    total_batches: int = 0
    total_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    total_processing_time: float = 0.0
    average_batch_size: float = 0.0
    records_per_second: float = 0.0
    last_flush_time: Optional[datetime] = None
    
    def update_metrics(self, batch_size: int, processing_time: float, success: bool = True):
        """Update metrics after processing a batch."""
        self.total_batches += 1
        self.total_records += batch_size
        if success:
            self.successful_records += batch_size
        else:
            self.failed_records += batch_size
        self.total_processing_time += processing_time
        self.average_batch_size = self.total_records / self.total_batches
        if self.total_processing_time > 0:
            self.records_per_second = self.successful_records / self.total_processing_time
        self.last_flush_time = datetime.now(timezone.utc)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 100
    flush_interval_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    max_memory_mb: int = 100  # Maximum memory usage for all batches combined
    max_total_pending_records: int = 1000  # Maximum total records across all batches
    
    @classmethod
    def for_production(cls) -> 'BatchConfig':
        """Production-optimized configuration."""
        return cls(
            max_batch_size=500,
            flush_interval_seconds=15.0,
            max_retries=5,
            retry_delay_seconds=2.0,
            enable_circuit_breaker=True,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=120.0,
            max_memory_mb=500,
            max_total_pending_records=5000
        )
    
    @classmethod
    def for_development(cls) -> 'BatchConfig':
        """Development-friendly configuration."""
        return cls(
            max_batch_size=10,
            flush_interval_seconds=5.0,
            max_retries=2,
            retry_delay_seconds=0.5,
            enable_circuit_breaker=False,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30.0,
            max_memory_mb=50,
            max_total_pending_records=100
        )
    
    @classmethod
    def for_testing(cls) -> 'BatchConfig':
        """Testing configuration with immediate flushing."""
        return cls(
            max_batch_size=1,
            flush_interval_seconds=0.1,
            max_retries=1,
            retry_delay_seconds=0.1,
            enable_circuit_breaker=False,
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=1.0,
            max_memory_mb=10,
            max_total_pending_records=10
        )
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.max_total_pending_records <= 0:
            raise ValueError("max_total_pending_records must be positive")


class CircuitBreaker:
    """Circuit breaker pattern for resilient batch processing."""
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"


class BatchExecutionRecorder:
    """
    High-throughput batch processor for Modal sandbox execution results.
    
    Features:
    - Configurable batch sizes (default 100 records)
    - Automatic flush on time intervals (default 30s)
    - Support for batching outcomes, tool usage, and failures
    - Performance monitoring and metrics
    - Graceful handling of partial batch failures
    - Circuit breaker pattern for resilience
    """
    
    def __init__(
        self,
        db_api: ArcAPI,
        config: Optional[BatchConfig] = None,
        on_batch_complete: Optional[Callable[[int, float], None]] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            db_api: Database API instance
            config: Batch processing configuration
            on_batch_complete: Optional callback for batch completion events
            
        Raises:
            ValueError: If database API doesn't support required batch operations
        """
        self.db_api = db_api
        self.config = config or BatchConfig()
        self.on_batch_complete = on_batch_complete
        
        # Validate database API compatibility
        self._validate_db_api_compatibility()
        
        # Batch storage
        self._outcome_batch: List[Dict[str, Any]] = []
        self._tool_usage_batch: List[Dict[str, Any]] = []
        self._failure_batch: List[Dict[str, Any]] = []
        
        # Timing and control
        self._last_flush_time = time.time()
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Metrics and monitoring
        self.metrics = BatchMetrics()
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout
        ) if self.config.enable_circuit_breaker else None
        
        logger.info(f"BatchExecutionRecorder initialized with batch_size={self.config.max_batch_size}, "
                   f"flush_interval={self.config.flush_interval_seconds}s")
    
    def _validate_db_api_compatibility(self):
        """Validate that the database API supports required batch operations."""
        required_methods = ['record_batch_outcomes', 'record_batch_tool_usage', 'record_batch_failures']
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(self.db_api, method):
                missing_methods.append(method)
        
        if missing_methods:
            raise ValueError(
                f"Database API missing required batch methods: {missing_methods}. "
                f"Please ensure your ArcAPI version supports batch processing."
            )
        
        logger.info("All required batch processing methods are available in database API")
    
    async def start(self):
        """Start the batch processor with automatic flushing."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._auto_flush_loop())
            logger.info("Batch processor started with automatic flushing")
    
    async def stop(self):
        """Stop the batch processor and flush remaining data."""
        self._shutdown_event.set()
        if self._flush_task:
            try:
                await asyncio.wait_for(self._flush_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Auto-flush task did not complete within timeout, cancelling")
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
        await self.flush_all()
        logger.info("Batch processor stopped")
    
    async def _check_resource_limits(self):
        """Check if adding another record would exceed resource limits."""
        total_pending = (
            len(self._outcome_batch) + 
            len(self._tool_usage_batch) + 
            len(self._failure_batch)
        )
        
        # Check total record limit
        if total_pending >= self.config.max_total_pending_records:
            logger.warning(f"Total pending records ({total_pending}) approaching limit "
                          f"({self.config.max_total_pending_records}), forcing flush")
            await self.flush_all()
        
        # If circuit breaker is open and we have pending records, flush them
        if (self.circuit_breaker and 
            not self.circuit_breaker.can_execute() and 
            total_pending > 0):
            logger.warning("Circuit breaker OPEN with pending records - forcing flush to prevent data loss")
            await self.flush_all()
    
    async def add_outcome(
        self,
        simulation_id: str,
        scenario_result: Dict[str, Any],
        modal_call_id: Optional[str] = None,
        sandbox_id: Optional[str] = None
    ):
        """
        Add a scenario outcome to the batch.
        
        Args:
            simulation_id: ID of the parent simulation
            scenario_result: Complete result dict from evaluate_single_scenario
            modal_call_id: Modal function call ID
            sandbox_id: Sandbox instance ID
            
        Raises:
            ResourceLimitError: If adding this record would exceed resource limits
        """
        # Check resource limits before adding
        await self._check_resource_limits()
        
        outcome_data = {
            "simulation_id": simulation_id,
            "scenario_result": scenario_result,
            "modal_call_id": modal_call_id,
            "sandbox_id": sandbox_id,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self._outcome_batch.append(outcome_data)
        
        # Check if we need to flush
        if len(self._outcome_batch) >= self.config.max_batch_size:
            await self._flush_outcomes()
    
    async def add_tool_usage(self, tool_usage_data: Dict[str, Any]):
        """Add tool usage data to the batch."""
        self._tool_usage_batch.append(tool_usage_data)
        
        if len(self._tool_usage_batch) >= self.config.max_batch_size:
            await self._flush_tool_usage()
    
    async def add_failure(self, failure_data: Dict[str, Any]):
        """Add failure data to the batch."""
        self._failure_batch.append(failure_data)
        
        if len(self._failure_batch) >= self.config.max_batch_size:
            await self._flush_failures()
    
    async def flush_all(self):
        """Flush all pending batches in parallel for optimal performance."""
        # Execute all flush operations concurrently
        flush_tasks = []
        
        if self._outcome_batch:
            flush_tasks.append(self._flush_outcomes())
        if self._tool_usage_batch:
            flush_tasks.append(self._flush_tool_usage())
        if self._failure_batch:
            flush_tasks.append(self._flush_failures())
        
        if flush_tasks:
            # Execute all flushes in parallel and collect results
            results = await asyncio.gather(*flush_tasks, return_exceptions=True)
            
            # Log any exceptions that occurred
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    batch_type = ["outcomes", "tool_usage", "failures"][i]
                    logger.error(f"Failed to flush {batch_type} batch: {result}")
                    # Re-raise the first exception encountered
                    if i == 0:  # Only re-raise the first exception to avoid multiple exceptions
                        raise result
    
    async def _flush_outcomes(self):
        """Flush the outcomes batch to database."""
        if not self._outcome_batch:
            return
        
        batch_size = len(self._outcome_batch)
        start_time = time.time()
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN - skipping batch of {batch_size} outcomes")
            return
        
        try:
            # Group by simulation_id for efficient batch processing
            simulation_groups = {}
            for outcome in self._outcome_batch:
                sim_id = outcome["simulation_id"]
                if sim_id not in simulation_groups:
                    simulation_groups[sim_id] = []
                simulation_groups[sim_id].append(outcome)
            
            # Process each simulation group
            for simulation_id, outcomes in simulation_groups.items():
                scenario_results = [o["scenario_result"] for o in outcomes]
                modal_call_ids = [o.get("modal_call_id") for o in outcomes]
                
                # Use existing batch API
                await self._retry_operation(
                    lambda: self.db_api.record_batch_outcomes(
                        simulation_id=simulation_id,
                        scenario_results=scenario_results,
                        modal_call_ids=modal_call_ids
                    )
                )
            
            processing_time = time.time() - start_time
            self.metrics.update_metrics(batch_size, processing_time, success=True)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            if self.on_batch_complete:
                self.on_batch_complete(batch_size, processing_time)
            
            logger.debug(f"Flushed {batch_size} outcomes in {processing_time:.3f}s "
                        f"({batch_size/processing_time:.1f} records/sec)")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_metrics(batch_size, processing_time, success=False)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            logger.error(f"Failed to flush outcomes batch: {e}")
            raise
        finally:
            self._outcome_batch.clear()
            self._last_flush_time = time.time()
    
    async def _flush_tool_usage(self):
        """Flush the tool usage batch to database."""
        if not self._tool_usage_batch:
            return
        
        batch_size = len(self._tool_usage_batch)
        start_time = time.time()
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN - skipping batch of {batch_size} tool usage records")
            return
        
        try:
            # Use batch API method directly
            await self._retry_operation(
                lambda: self.db_api.record_batch_tool_usage(self._tool_usage_batch)
            )
            
            processing_time = time.time() - start_time
            self.metrics.update_metrics(batch_size, processing_time, success=True)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            logger.debug(f"Flushed {batch_size} tool usage records in {processing_time:.3f}s")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_metrics(batch_size, processing_time, success=False)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            logger.error(f"Failed to flush tool usage batch: {e}")
            raise
        finally:
            self._tool_usage_batch.clear()
    
    async def _flush_failures(self):
        """Flush the failures batch to database."""
        if not self._failure_batch:
            return
        
        batch_size = len(self._failure_batch)
        start_time = time.time()
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN - skipping batch of {batch_size} failure records")
            return
        
        try:
            # Use batch API method directly
            await self._retry_operation(
                lambda: self.db_api.record_batch_failures(self._failure_batch)
            )
            
            processing_time = time.time() - start_time
            self.metrics.update_metrics(batch_size, processing_time, success=True)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            logger.debug(f"Flushed {batch_size} failure records in {processing_time:.3f}s")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_metrics(batch_size, processing_time, success=False)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            logger.error(f"Failed to flush failures batch: {e}")
            raise
        finally:
            self._failure_batch.clear()
    
    async def _retry_operation(self, operation: Callable):
        """Retry an operation with exponential backoff."""
        for attempt in range(self.config.max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                delay = self.config.retry_delay_seconds * (2 ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                              f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    
    async def _auto_flush_loop(self):
        """Automatic flush loop that runs in the background."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for flush interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.flush_interval_seconds
                )
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                # Flush interval reached
                current_time = time.time()
                if current_time - self._last_flush_time >= self.config.flush_interval_seconds:
                    try:
                        await self.flush_all()
                    except Exception as e:
                        logger.error(f"Auto-flush failed: {e}")
    
    def get_metrics(self) -> BatchMetrics:
        """Get current batch processing metrics."""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the batch processor."""
        return {
            "pending_outcomes": len(self._outcome_batch),
            "pending_tool_usage": len(self._tool_usage_batch),
            "pending_failures": len(self._failure_batch),
            "circuit_breaker_state": self.circuit_breaker.state if self.circuit_breaker else "DISABLED",
            "metrics": {
                "total_batches": self.metrics.total_batches,
                "total_records": self.metrics.total_records,
                "successful_records": self.metrics.successful_records,
                "failed_records": self.metrics.failed_records,
                "records_per_second": self.metrics.records_per_second,
                "last_flush_time": self.metrics.last_flush_time.isoformat() if self.metrics.last_flush_time else None
            }
        }


@asynccontextmanager
async def batch_processor_context(
    db_api: ArcAPI,
    config: Optional[BatchConfig] = None,
    on_batch_complete: Optional[Callable[[int, float], None]] = None
):
    """
    Context manager for batch processor lifecycle.
    
    Usage:
        async with batch_processor_context(db_api) as batch_processor:
            await batch_processor.add_outcome(simulation_id, result)
    """
    processor = BatchExecutionRecorder(db_api, config, on_batch_complete)
    await processor.start()
    try:
        yield processor
    finally:
        await processor.stop() 