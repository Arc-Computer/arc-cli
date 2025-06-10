"""
Unit tests for ArcEvalDBClient
==============================

Comprehensive unit tests for the database client with mocking
to ensure >90% code coverage without requiring actual database.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
import json

from sqlalchemy.exc import OperationalError
import asyncpg

# Import the client and related classes
from arc.database.client import (
    ArcEvalDBClient, 
    TimescaleDBHealth,
    get_engine,
    DatabaseError,
    ConnectionError,
    RetryableError,
    with_retry
)


class TestRetryDecorator:
    """Test the retry decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test function succeeds on first attempt."""
        call_count = 0
        
        @with_retry(max_attempts=3, delay=0.1)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test function succeeds after retryable failures."""
        call_count = 0
        
        @with_retry(max_attempts=3, delay=0.1)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OperationalError("Connection failed", None, None)
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self):
        """Test function fails after max attempts."""
        call_count = 0
        
        @with_retry(max_attempts=3, delay=0.1)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise OperationalError("Connection failed", None, None)
        
        with pytest.raises(RetryableError) as exc_info:
            await test_func()
        
        assert "Operation failed after 3 attempts" in str(exc_info.value)
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self):
        """Test non-retryable errors are raised immediately."""
        call_count = 0
        
        @with_retry(max_attempts=3, delay=0.1)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            await test_func()
        
        assert call_count == 1


class TestTimescaleDBHealth:
    """Test the TimescaleDBHealth class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        engine = Mock()  # Use Mock instead of AsyncMock for the engine itself
        engine.pool = Mock()
        engine.pool.size.return_value = 20
        engine.pool.checkedin.return_value = 18
        engine.pool.checkedout.return_value = 2
        engine.pool.overflow.return_value = 0
        engine.pool.total.return_value = 20
        
        # Configure begin() to return an async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock()
        mock_context.__aexit__ = AsyncMock()
        engine.begin = Mock(return_value=mock_context)
        
        return engine
    
    @pytest.fixture
    def health_checker(self, mock_engine):
        """Create a TimescaleDBHealth instance with mock engine."""
        return TimescaleDBHealth(mock_engine)
    
    @pytest.mark.asyncio
    async def test_check_extensions_success(self, health_checker, mock_engine):
        """Test successful extension check."""
        # Create mock rows that behave like database result rows
        class MockRow:
            def __init__(self, name, installed_version):
                self.name = name
                self.installed_version = installed_version
        
        mock_result = [
            MockRow("timescaledb", "2.11.0"),
            MockRow("uuid-ossp", "1.1"),
            MockRow("vector", "0.5.0")
        ]
        
        # Configure the connection mock
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        
        # Set up the async context manager
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        result = await health_checker.check_extensions()
        
        # The result should be a dict with extension names as keys
        assert result == {
            "timescaledb": "2.11.0",
            "uuid-ossp": "1.1",
            "vector": "0.5.0"
        }
    
    @pytest.mark.asyncio
    async def test_check_extensions_failure(self, health_checker, mock_engine):
        """Test extension check failure."""
        # The issue might be that we need to make the whole async context fail
        # Let's make the begin() method itself raise an exception
        mock_engine.begin.side_effect = Exception("Database connection error")
        
        # Now when check_extensions tries to use the engine, it will fail
        with pytest.raises(Exception) as exc_info:
            await health_checker.check_extensions()
        
        assert "Database connection error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_check_hypertables_success(self, health_checker, mock_engine):
        """Test successful hypertable check."""
        mock_result = [
            Mock(_mapping={"hypertable_name": "outcomes", "num_chunks": 10, "compression_enabled": True}),
            Mock(_mapping={"hypertable_name": "tool_usage", "num_chunks": 5, "compression_enabled": True})
        ]
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        
        # Set up the async context manager
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        result = await health_checker.check_hypertables()
        
        assert len(result) == 2
        assert result[0]["hypertable_name"] == "outcomes"
        assert result[0]["num_chunks"] == 10
    
    def test_check_connection_pool(self, health_checker, mock_engine):
        """Test connection pool statistics."""
        result = asyncio.run(health_checker.check_connection_pool())
        
        assert result == {
            "size": 20,
            "checked_in": 18,
            "checked_out": 2,
            "overflow": 0,
            "total": 20
        }


class TestArcEvalDBClient:
    """Test the main ArcEvalDBClient class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        with patch('arc.database.client.get_engine') as mock_get_engine:
            engine = Mock()  # Use Mock instead of AsyncMock
            engine.pool = Mock()
            engine.pool.size.return_value = 20
            engine.pool.checkedin.return_value = 20
            engine.pool.checkedout.return_value = 0
            engine.pool.overflow.return_value = 0
            engine.pool.total.return_value = 20
            
            # Configure begin() to return an async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock()
            mock_context.__aexit__ = AsyncMock()
            engine.begin = Mock(return_value=mock_context)
            
            # Configure dispose as an async method
            engine.dispose = AsyncMock()
            
            mock_get_engine.return_value = engine
            yield engine
    
    @pytest.fixture
    def client(self, mock_engine):
        """Create an ArcEvalDBClient instance."""
        with patch.dict('os.environ', {'TIMESCALE_SERVICE_URL': 'postgresql://test:test@localhost/test'}):
            return ArcEvalDBClient(enable_monitoring=True)
    
    def test_client_initialization_with_connection_string(self, mock_engine):
        """Test client initialization with explicit connection string."""
        client = ArcEvalDBClient(connection_string="postgresql://test:test@localhost/test")
        assert client.connection_string == "postgresql://test:test@localhost/test"
        assert client.enable_monitoring is True
    
    def test_client_initialization_with_env_var(self, mock_engine):
        """Test client initialization with environment variable."""
        with patch.dict('os.environ', {'TIMESCALE_SERVICE_URL': 'postgresql://env:env@localhost/env'}):
            client = ArcEvalDBClient()
            assert client.connection_string == "postgresql://env:env@localhost/env"
    
    def test_client_initialization_no_connection_string(self, mock_engine):
        """Test client initialization fails without connection string."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ArcEvalDBClient()
            assert "No connection string provided" in str(exc_info.value)
    
    def test_update_metrics(self, client):
        """Test metrics update functionality."""
        # Test successful query
        client._update_metrics(0.1, True, 0)
        assert client._metrics["total_queries"] == 1
        assert client._metrics["failed_queries"] == 0
        assert client._metrics["avg_query_time"] == 0.1
        
        # Test failed query with retries
        client._update_metrics(0.2, False, 2)
        assert client._metrics["total_queries"] == 2
        assert client._metrics["failed_queries"] == 1
        assert client._metrics["retry_count"] == 2
        assert abs(client._metrics["avg_query_time"] - 0.15) < 0.0001  # Fix floating point comparison
    
    def test_get_metrics(self, client):
        """Test getting metrics."""
        client._update_metrics(0.1, True, 0)
        client._update_metrics(0.2, False, 1)
        
        metrics = client.get_metrics()
        assert metrics["total_queries"] == 2
        assert metrics["failed_queries"] == 1
        assert metrics["success_rate"] == 50.0
        assert "pool_stats" in metrics
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, client, mock_engine):
        """Test successful initialization."""
        # Mock health check methods
        with patch.object(client.health, 'check_extensions') as mock_extensions, \
             patch.object(client.health, 'check_hypertables') as mock_hypertables, \
             patch.object(client.health, 'check_connection_pool') as mock_pool:
            
            mock_extensions.return_value = {"timescaledb": "2.11.0"}
            mock_hypertables.return_value = [{"hypertable_name": "outcomes"}]
            mock_pool.return_value = {"size": 20}
            
            result = await client.initialize()
            
            assert result["status"] == "healthy"
            assert "extensions" in result
            assert "hypertables" in result
            assert "pool_stats" in result
            assert "response_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, client, mock_engine):
        """Test initialization failure."""
        with patch.object(client.health, 'check_extensions') as mock_extensions:
            mock_extensions.side_effect = Exception("Connection failed")
            
            result = await client.initialize()
            
            assert result["status"] == "error"
            assert "Connection failed" in result["error"]
            assert client._metrics["failed_queries"] == 1
    
    @pytest.mark.asyncio
    async def test_create_configuration(self, client, mock_engine):
        """Test configuration creation."""
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        config_id = await client.create_configuration(
            name="test_config",
            user_id="test_user",
            initial_config={"model": "gpt-4", "temperature": 0.7}
        )
        
        # Verify UUID format
        uuid.UUID(config_id)  # Will raise if invalid
        
        # Verify execute was called twice (config and version)
        assert mock_conn.execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_record_outcome(self, client, mock_engine):
        """Test outcome recording."""
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        outcome_data = {
            "simulation_id": str(uuid.uuid4()),
            "scenario_id": "test_scenario",
            "status": "success",
            "reliability_score": 0.95,
            "execution_time_ms": 1500,
            "tokens_used": 250,
            "cost_usd": 0.01,
            "trajectory": {"steps": ["step1", "step2"]}
        }
        
        outcome_id = await client.record_outcome(outcome_data)
        
        # Verify UUID format
        uuid.UUID(outcome_id)
        
        # Verify execute was called
        assert mock_conn.execute.called
    
    @pytest.mark.asyncio
    async def test_record_outcomes_batch(self, client, mock_engine):
        """Test batch outcome recording."""
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        outcomes = [
            {
                "simulation_id": str(uuid.uuid4()),
                "scenario_id": f"scenario_{i}",
                "status": "success",
                "reliability_score": 0.9,
                "execution_time_ms": 1000,
                "tokens_used": 200,
                "cost_usd": 0.01,
                "trajectory": {"batch": i}
            }
            for i in range(5)
        ]
        
        outcome_ids = await client.record_outcomes_batch(outcomes)
        
        assert len(outcome_ids) == 5
        for outcome_id in outcome_ids:
            uuid.UUID(outcome_id)  # Verify valid UUID
    
    @pytest.mark.asyncio
    async def test_get_simulation_performance(self, client, mock_engine):
        """Test simulation performance query."""
        mock_conn = AsyncMock()
        mock_result = [
            Mock(_mapping={
                "hour": datetime.now(),
                "total_outcomes": 100,
                "avg_reliability": 0.92,
                "avg_execution_time": 1200,
                "total_cost": 1.5,
                "error_count": 5
            })
        ]
        mock_conn.execute.return_value = mock_result
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        
        result = await client.get_simulation_performance(
            simulation_id=str(uuid.uuid4()),
            time_range=timedelta(hours=24)
        )
        
        assert "hourly_metrics" in result
        assert "summary" in result
        assert result["summary"]["total_outcomes"] == 100
    
    @pytest.mark.asyncio
    async def test_close(self, client, mock_engine):
        """Test client close method."""
        await client.close()
        mock_engine.dispose.assert_called_once()


class TestEngineCreation:
    """Test the engine creation function."""
    
    @patch('arc.database.client.create_async_engine')
    def test_get_engine_postgres_url_conversion(self, mock_create_engine):
        """Test postgres:// URL conversion."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Reset global engine
        import arc.database.client
        arc.database.client._engine = None
        
        engine = get_engine("postgres://user:pass@host:5432/db")
        
        # Check that URL was converted
        call_args = mock_create_engine.call_args[0][0]
        assert call_args.startswith("postgresql+asyncpg://")
    
    @patch('arc.database.client.create_async_engine')
    def test_get_engine_ssl_removal(self, mock_create_engine):
        """Test SSL parameter removal from URL."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Reset global engine
        import arc.database.client
        arc.database.client._engine = None
        
        engine = get_engine("postgresql://user:pass@host:5432/db?sslmode=require")
        
        # Check that sslmode was removed
        call_args = mock_create_engine.call_args[0][0]
        assert "sslmode" not in call_args
    
    @patch('arc.database.client.create_async_engine')
    def test_get_engine_caching(self, mock_create_engine):
        """Test engine caching."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Reset global engine
        import arc.database.client
        arc.database.client._engine = None
        
        engine1 = get_engine()
        engine2 = get_engine()
        
        # Should only create engine once
        assert mock_create_engine.call_count == 1
        assert engine1 is engine2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=arc.database.client", "--cov-report=term-missing"]) 