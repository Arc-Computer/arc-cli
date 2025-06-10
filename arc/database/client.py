"""Enhanced async database client for TimescaleDB integration.

Provides high-level database operations optimized for TimescaleDB features
including hypertables, continuous aggregates, and time-series queries.
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import AsyncIterator, Dict, List, Optional, Any, TypeVar, Callable
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from functools import wraps

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select, insert, update, delete
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
import asyncpg

# Configure logging
logger = logging.getLogger(__name__)

__all__: list[str] = ["get_engine", "get_session", "ArcDBClient", "TimescaleDBHealth"]

_engine: AsyncEngine | None = None

# Type variable for retry decorator
T = TypeVar('T')


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class RetryableError(DatabaseError):
    """Raised for errors that can be retried."""
    pass


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying database operations with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, asyncpg.PostgresConnectionError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Retryable error in {func.__name__} (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            raise RetryableError(f"Operation failed after {max_attempts} attempts") from last_exception
        
        return wrapper
    return decorator


def get_engine(dsn: str | None = None) -> AsyncEngine:
    """Return a cached AsyncEngine optimized for TimescaleDB."""
    global _engine
    if _engine is None:
        # Use environment variables for TimescaleDB connection
        dsn = dsn or os.getenv(
            "TIMESCALE_SERVICE_URL",
            "postgresql+asyncpg://user:pass@localhost:5432/arc"
        )
        
        # Handle connection string format conversion
        if dsn.startswith("postgres://"):
            dsn = dsn.replace("postgres://", "postgresql+asyncpg://", 1)
        elif dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif not dsn.startswith("postgresql+asyncpg://"):
            # If no dialect specified, add it
            if "://" in dsn:
                dsn = dsn.replace("://", "+asyncpg://", 1)
        
        # Remove sslmode from connection string if present (handled in connect_args)
        if "?sslmode=" in dsn:
            dsn = dsn.split("?sslmode=")[0]
        elif "&sslmode=" in dsn:
            dsn = dsn.split("&sslmode=")[0]
        
        _engine = create_async_engine(
            dsn, 
            echo=False, 
            future=True,
            pool_size=20,
            max_overflow=30,
            pool_timeout=30,
            pool_recycle=3600,  # 1 hour
            pool_pre_ping=True,  # Enable connection health checks
            connect_args={
                "ssl": "require",  # SSL for TimescaleDB Cloud
                "server_settings": {
                    "jit": "off",  # Recommended for analytical workloads
                    "timezone": "UTC"
                }
            }
        )
        
        logger.info("Database engine created successfully")
    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession bound to the global engine."""
    engine = get_engine()
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session


class TimescaleDBHealth:
    """Health monitoring for TimescaleDB instance."""
    
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
    
    @with_retry(max_attempts=3, delay=0.5)
    async def check_extensions(self) -> Dict[str, str]:
        """Check that required extensions are installed."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT name, installed_version 
                    FROM pg_available_extensions 
                    WHERE name IN ('timescaledb', 'uuid-ossp', 'vector')
                    AND installed_version IS NOT NULL
                """))
                extensions = {row.name: row.installed_version for row in result}
                logger.debug(f"Extensions found: {extensions}")
                return extensions
        except Exception as e:
            logger.error(f"Failed to check extensions: {e}")
            raise DatabaseError(f"Extension check failed: {e}") from e
    
    @with_retry(max_attempts=3, delay=0.5)
    async def check_hypertables(self) -> List[Dict[str, Any]]:
        """Check status of hypertables."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT 
                        hypertable_name,
                        num_chunks,
                        compression_enabled
                    FROM timescaledb_information.hypertables
                """))
                hypertables = [dict(row._mapping) for row in result]
                logger.debug(f"Found {len(hypertables)} hypertables")
                return hypertables
        except Exception as e:
            logger.error(f"Failed to check hypertables: {e}")
            raise DatabaseError(f"Hypertable check failed: {e}") from e
    
    @with_retry(max_attempts=3, delay=0.5)
    async def check_compression_stats(self) -> List[Dict[str, Any]]:
        """Check compression statistics."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT 
                        hypertable_name,
                        total_chunks,
                        number_compressed_chunks,
                        uncompressed_heap_size,
                        compressed_heap_size,
                        compressed_index_size,
                        compressed_toast_size
                    FROM timescaledb_information.compression_settings cs
                    JOIN timescaledb_information.hypertables h 
                        ON cs.hypertable_name = h.hypertable_name
                """))
                stats = [dict(row._mapping) for row in result]
                logger.debug(f"Compression stats retrieved for {len(stats)} hypertables")
                return stats
        except Exception as e:
            logger.error(f"Failed to check compression stats: {e}")
            raise DatabaseError(f"Compression stats check failed: {e}") from e
    
    async def check_connection_pool(self) -> Dict[str, Any]:
        """Check connection pool statistics."""
        try:
            pool = self.engine.pool
            return {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total": pool.total()
            }
        except Exception as e:
            logger.error(f"Failed to check connection pool: {e}")
            return {"error": str(e)}


class ArcDBClient:
    """
    High-level database client for Arc platform.
    Optimized for TimescaleDB features and Modal sandbox integration.
    
    Features:
    - Async connection pooling with health checks
    - Automatic retry logic with exponential backoff
    - Comprehensive error handling and logging
    - Performance monitoring and metrics
    - Environment-based configuration
    """
    
    def __init__(self, connection_string: str | None = None, 
                 enable_monitoring: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize the database client.
        
        Args:
            connection_string: Optional database connection string. 
                             Falls back to TIMESCALE_SERVICE_URL env var.
            enable_monitoring: Enable performance monitoring and metrics.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        """
        self.connection_string = connection_string or os.getenv("TIMESCALE_SERVICE_URL")
        if not self.connection_string:
            raise ValueError(
                "No connection string provided. Set TIMESCALE_SERVICE_URL environment variable "
                "or pass connection_string parameter."
            )
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        
        self.engine = get_engine(self.connection_string)
        self.health = TimescaleDBHealth(self.engine)
        self.enable_monitoring = enable_monitoring
        
        # Performance metrics
        self._metrics = {
            "total_queries": 0,
            "failed_queries": 0,
            "retry_count": 0,
            "avg_query_time": 0.0,
            "last_health_check": None
        }
        
        logger.info(f"ArcDBClient initialized with monitoring={'enabled' if enable_monitoring else 'disabled'}")
    
    def _update_metrics(self, query_time: float, success: bool, retries: int = 0):
        """Update internal performance metrics."""
        if not self.enable_monitoring:
            return
        
        self._metrics["total_queries"] += 1
        if not success:
            self._metrics["failed_queries"] += 1
        self._metrics["retry_count"] += retries
        
        # Update average query time
        current_avg = self._metrics["avg_query_time"]
        total_queries = self._metrics["total_queries"]
        self._metrics["avg_query_time"] = (
            (current_avg * (total_queries - 1) + query_time) / total_queries
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        pool_stats = await self.health.check_connection_pool()
        return {
            **self._metrics,
            "success_rate": (
                (self._metrics["total_queries"] - self._metrics["failed_queries"]) 
                / self._metrics["total_queries"] * 100
                if self._metrics["total_queries"] > 0 else 0
            ),
            "pool_stats": pool_stats
        }
    
    @with_retry(max_attempts=3, delay=1.0)
    async def initialize(self) -> Dict[str, Any]:
        """Initialize database connection and verify TimescaleDB setup."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check extensions
            extensions = await self.health.check_extensions()
            
            # Check hypertables
            hypertables = await self.health.check_hypertables()
            
            # Check connection pool
            pool_stats = await self.health.check_connection_pool()
            
            query_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(query_time, True)
            self._metrics["last_health_check"] = datetime.utcnow()
            
            result = {
                "status": "healthy",
                "extensions": extensions,
                "hypertables": hypertables,
                "pool_stats": pool_stats,
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": query_time * 1000
            }
            
            logger.info(f"Database initialized successfully in {query_time*1000:.1f}ms")
            return result
            
        except Exception as e:
            query_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(query_time, False)
            
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "response_time_ms": query_time * 1000
            }
            
            logger.error(f"Database initialization failed: {e}")
            return error_result
    
    async def deploy_schema(self) -> Dict[str, Any]:
        """Deploy the complete schema to TimescaleDB."""
        try:
            schema_file = os.path.join(os.path.dirname(__file__), "schema", "tables.sql")
            indexes_file = os.path.join(os.path.dirname(__file__), "schema", "indexes.sql")
            
            # Read schema files
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            with open(indexes_file, 'r') as f:
                indexes_sql = f.read()
            
            # Split SQL into individual statements, handling DO blocks properly
            def split_sql_statements(sql_content):
                statements = []
                current_statement = ""
                in_do_block = False
                
                for line in sql_content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('--'):
                        continue
                    
                    current_statement += line + '\n'
                    
                    # Check for DO block start
                    if line.startswith('DO $$'):
                        in_do_block = True
                    
                    # Check for DO block end
                    if in_do_block and line == 'END $$;':
                        in_do_block = False
                        statements.append(current_statement.strip())
                        current_statement = ""
                    elif not in_do_block and line.endswith(';'):
                        statements.append(current_statement.strip())
                        current_statement = ""
                
                if current_statement.strip():
                    statements.append(current_statement.strip())
                
                return statements
            
            schema_statements = split_sql_statements(schema_sql)
            indexes_statements = split_sql_statements(indexes_sql)
            
            # Separate continuous aggregates and policies (can't run in transaction)
            continuous_agg_statements = []
            policy_statements = []
            other_statements = []
            
            for stmt in indexes_statements:
                if 'CREATE MATERIALIZED VIEW' in stmt and 'timescaledb.continuous' in stmt:
                    continuous_agg_statements.append(stmt)
                elif 'add_continuous_aggregate_policy' in stmt or 'add_compression_policy' in stmt or 'add_retention_policy' in stmt:
                    policy_statements.append(stmt)
                else:
                    other_statements.append(stmt)
            
            async with self.engine.begin() as conn:
                # Execute schema statements one by one
                for stmt in schema_statements:
                    if stmt:
                        await conn.execute(text(stmt))
                
                # Execute non-continuous aggregate statements
                for stmt in other_statements:
                    if stmt:
                        await conn.execute(text(stmt))
            
            # Execute continuous aggregates outside transaction (autocommit mode)
            for stmt in continuous_agg_statements:
                if stmt:
                    async with self.engine.connect() as conn:
                        await conn.execute(text(stmt))
            
            # Execute policies after continuous aggregates are created
            for stmt in policy_statements:
                if stmt:
                    async with self.engine.connect() as conn:
                        await conn.execute(text(stmt))
            
            return {
                "status": "success", 
                "message": "Schema deployed successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Configuration Management
    async def create_configuration(self, name: str, user_id: str, 
                                 initial_config: Dict[str, Any]) -> str:
        """Create a new configuration with initial version.
        
        Args:
            name: Configuration name
            user_id: User ID who created the configuration
            initial_config: Initial configuration dictionary
            
        Returns:
            str: The version_id of the created configuration version
        """
        config_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        
        # Generate config hash
        config_yaml = json.dumps(initial_config, sort_keys=True)
        config_hash = hashlib.sha256(config_yaml.encode()).hexdigest()
        
        async with self.engine.begin() as conn:
            # Create configuration
            await conn.execute(text("""
                INSERT INTO configurations (config_id, name, user_id, latest_version_id)
                VALUES (:config_id, :name, :user_id, :version_id)
            """), {
                "config_id": config_id,
                "name": name,
                "user_id": user_id,
                "version_id": version_id
            })
            
            # Create initial version
            await conn.execute(text("""
                INSERT INTO config_versions (
                    version_id, config_id, version_number, raw_yaml, 
                    parsed_config, config_hash
                ) VALUES (:version_id, :config_id, 1, :raw_yaml, :parsed_config, :config_hash)
            """), {
                "version_id": version_id,
                "config_id": config_id,
                "raw_yaml": config_yaml,
                "parsed_config": json.dumps(initial_config),
                "config_hash": config_hash
            })
        
        return version_id
    
    # Simulation Management
    async def create_simulation(self, config_version_id: str, scenario_set: List[str],
                              simulation_name: str = None, 
                              modal_app_id: str = None) -> str:
        """Create a new simulation run."""
        simulation_id = str(uuid.uuid4())
        
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO simulations (
                    simulation_id, config_version_id, scenario_set,
                    simulation_name, total_scenarios, modal_app_id, status
                ) VALUES (:simulation_id, :config_version_id, :scenario_set, 
                         :simulation_name, :total_scenarios, :modal_app_id, 'pending')
            """), {
                "simulation_id": simulation_id,
                "config_version_id": config_version_id,
                "scenario_set": scenario_set,
                "simulation_name": simulation_name,
                "total_scenarios": len(scenario_set),
                "modal_app_id": modal_app_id
            })
        
        return simulation_id

    @with_retry(max_attempts=3, delay=0.5)
    async def finalize_simulation(
        self,
        simulation_id: str,
        *,
        status: str = "completed",
        overall_score: float | None = None,
        total_cost_usd: float | None = None,
        execution_time_ms: int | None = None,
        metadata: Dict[str, Any] | None = None,
        completed_scenarios: int | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        """Update simulation record with final metadata."""
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    """
                UPDATE simulations
                SET
                    status = :status,
                    overall_score = :overall_score,
                    total_cost_usd = :total_cost_usd,
                    execution_time_ms = :execution_time_ms,
                    metadata = :metadata,
                    completed_scenarios = :completed_scenarios,
                    completed_at = :completed_at
                WHERE simulation_id = :simulation_id
                """
                ),
                {
                    "simulation_id": simulation_id,
                    "status": status,
                    "overall_score": overall_score,
                    "total_cost_usd": total_cost_usd,
                    "execution_time_ms": execution_time_ms,
                    "metadata": json.dumps(metadata or {}),
                    "completed_scenarios": completed_scenarios,
                    "completed_at": completed_at or datetime.utcnow(),
                },
            )

    # Scenario Management
    async def ensure_scenario_exists(self, scenario_id: str, scenario_data: Optional[Dict[str, Any]] = None) -> None:
        """Ensure a scenario exists in the database, creating it if necessary."""
        async with self.engine.begin() as conn:
            # Check if scenario exists
            result = await conn.execute(text(
                "SELECT 1 FROM scenarios WHERE scenario_id = :scenario_id"
            ), {"scenario_id": scenario_id})
            
            if result.fetchone() is None:
                # Create minimal scenario record
                scenario_name = scenario_data.get("name", f"Generated Scenario {scenario_id}") if scenario_data else f"Generated Scenario {scenario_id}"
                task_prompt = scenario_data.get("task_prompt", "Auto-generated test scenario") if scenario_data else "Auto-generated test scenario"
                
                await conn.execute(text("""
                    INSERT INTO scenarios (scenario_id, name, task_prompt, difficulty_level, is_active)
                    VALUES (:scenario_id, :name, :task_prompt, 'medium', true)
                    ON CONFLICT (scenario_id) DO NOTHING
                """), {
                    "scenario_id": scenario_id,
                    "name": scenario_name[:200],  # Respect length constraint
                    "task_prompt": task_prompt
                })

    # Outcome Recording (Optimized for TimescaleDB hypertable)
    def _validate_trajectory(self, trajectory: Dict[str, Any]) -> None:
        """Validate trajectory data meets database constraints."""
        if not isinstance(trajectory, dict):
            raise ValueError("Trajectory must be a dictionary")
        
        # Check required fields
        if "start_time" not in trajectory:
            raise ValueError(
                "Trajectory missing required field 'start_time'. "
                "Must be ISO 8601 timestamp (e.g., '2024-06-10T14:30:00Z')"
            )
        
        if "status" not in trajectory:
            raise ValueError(
                "Trajectory missing required field 'status'. "
                "Must be one of: 'success', 'error', 'timeout', 'cancelled'"
            )
        
        # Validate status value
        valid_statuses = {'success', 'error', 'timeout', 'cancelled'}
        if trajectory["status"] not in valid_statuses:
            raise ValueError(
                f"Invalid trajectory status '{trajectory['status']}'. "
                f"Must be one of: {', '.join(sorted(valid_statuses))}"
            )

    async def record_outcome(self, outcome_data: Dict[str, Any]) -> str:
        """Record individual scenario outcome to TimescaleDB hypertable."""
        outcome_id = str(uuid.uuid4())
        
        # Validate trajectory data
        if "trajectory" in outcome_data:
            self._validate_trajectory(outcome_data["trajectory"])
        
        # Ensure scenario exists before recording outcome
        await self.ensure_scenario_exists(
            outcome_data["scenario_id"], 
            outcome_data.get("scenario_data")
        )
        
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO outcomes (
                    outcome_id, simulation_id, scenario_id, execution_time,
                    status, reliability_score, execution_time_ms, tokens_used,
                    cost_usd, trajectory, modal_call_id, sandbox_id,
                    error_code, error_category, retry_count, metrics
                ) VALUES (
                    :outcome_id, :simulation_id, :scenario_id, :execution_time,
                    :status, :reliability_score, :execution_time_ms, :tokens_used,
                    :cost_usd, :trajectory, :modal_call_id, :sandbox_id,
                    :error_code, :error_category, :retry_count, :metrics
                )
            """), {
                "outcome_id": outcome_id,
                "simulation_id": outcome_data["simulation_id"],
                "scenario_id": outcome_data["scenario_id"],
                "execution_time": outcome_data.get("execution_time", datetime.utcnow()),
                "status": outcome_data["status"],
                "reliability_score": outcome_data["reliability_score"],
                "execution_time_ms": outcome_data["execution_time_ms"],
                "tokens_used": outcome_data["tokens_used"],
                "cost_usd": outcome_data["cost_usd"],
                "trajectory": json.dumps(outcome_data["trajectory"]),
                "modal_call_id": outcome_data.get("modal_call_id"),
                "sandbox_id": outcome_data.get("sandbox_id"),
                "error_code": outcome_data.get("error_code"),
                "error_category": outcome_data.get("error_category"),
                "retry_count": outcome_data.get("retry_count", 0),
                "metrics": json.dumps(outcome_data.get("metrics", {}))
            })
        
        return outcome_id
    
    # Batch Operations (Optimized for high-throughput Modal executions)
    async def record_outcomes_batch(self, outcomes: List[Dict[str, Any]]) -> List[str]:
        """Batch insert outcomes for high-throughput Modal executions."""
        outcome_ids = [str(uuid.uuid4()) for _ in outcomes]

        # Validate all trajectories and ensure scenarios exist
        for outcome in outcomes:
            if "trajectory" in outcome:
                self._validate_trajectory(outcome["trajectory"])
            await self.ensure_scenario_exists(
                outcome["scenario_id"],
                outcome.get("scenario_data")
            )
        
        batch_data = []
        for i, outcome in enumerate(outcomes):
            batch_data.append({
                "outcome_id": outcome_ids[i],
                "simulation_id": outcome["simulation_id"],
                "scenario_id": outcome["scenario_id"],
                "execution_time": outcome.get("execution_time", datetime.utcnow()),
                "status": outcome["status"],
                "reliability_score": outcome["reliability_score"],
                "execution_time_ms": outcome["execution_time_ms"],
                "tokens_used": outcome["tokens_used"],
                "cost_usd": outcome["cost_usd"],
                "trajectory": json.dumps(outcome["trajectory"]),
                "modal_call_id": outcome.get("modal_call_id"),
                "sandbox_id": outcome.get("sandbox_id"),
                "error_code": outcome.get("error_code"),
                "error_category": outcome.get("error_category"),
                "retry_count": outcome.get("retry_count", 0),
                "metrics": json.dumps(outcome.get("metrics", {}))
            })
        
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                INSERT INTO outcomes (
                    outcome_id, simulation_id, scenario_id, execution_time,
                    status, reliability_score, execution_time_ms, tokens_used,
                    cost_usd, trajectory, modal_call_id, sandbox_id,
                    error_code, error_category, retry_count, metrics
                ) VALUES (
                    :outcome_id, :simulation_id, :scenario_id, :execution_time,
                    :status, :reliability_score, :execution_time_ms, :tokens_used,
                    :cost_usd, :trajectory, :modal_call_id, :sandbox_id,
                    :error_code, :error_category, :retry_count, :metrics
                )
            """), batch_data)
        
        return outcome_ids
    
    # Analytics Queries (Direct queries since continuous aggregates are not deployed yet)
    async def get_simulation_performance(self, simulation_id: str, 
                                       time_range: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Get simulation performance metrics using direct TimescaleDB queries."""
        async with self.engine.begin() as conn:
            # Convert timedelta to PostgreSQL interval string
            total_seconds = int(time_range.total_seconds())
            interval_str = f"'{total_seconds} seconds'"
            
            # Query outcomes table directly with time bucketing
            # Note: Using string formatting for interval since PostgreSQL doesn't allow parameters after INTERVAL
            query = f"""
                SELECT 
                    time_bucket('1 hour', execution_time) AS hour,
                    COUNT(*) AS total_outcomes,
                    AVG(reliability_score) AS avg_reliability,
                    AVG(execution_time_ms) AS avg_execution_time,
                    SUM(cost_usd) AS total_cost,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count
                FROM outcomes
                WHERE simulation_id = :simulation_id
                AND execution_time >= NOW() - INTERVAL {interval_str}
                GROUP BY hour
                ORDER BY hour DESC
            """
            
            result = await conn.execute(text(query), {
                "simulation_id": simulation_id
            })
            
            rows = [dict(row._mapping) for row in result]
            
            return {
                "simulation_id": simulation_id,
                "time_range_hours": int(time_range.total_seconds() / 3600),
                "hourly_metrics": rows,
                "summary": {
                    "total_outcomes": sum(r["total_outcomes"] for r in rows),
                    "avg_reliability": sum(r["avg_reliability"] for r in rows) / len(rows) if rows else 0,
                    "total_cost": sum(r["total_cost"] for r in rows)
                }
            }
    
    async def get_recent_failures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent failures for analysis."""
        async with self.engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT 
                    o.outcome_id,
                    o.simulation_id,
                    o.scenario_id,
                    o.execution_time,
                    o.status,
                    o.error_code,
                    o.error_category,
                    o.modal_call_id,
                    o.trajectory->>'error_message' as error_message
                FROM outcomes o
                WHERE o.status IN ('error', 'timeout', 'cancelled')
                ORDER BY o.execution_time DESC
                LIMIT :limit
            """), {"limit": limit})
            
            return [dict(row._mapping) for row in result]
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
