"""
Unit tests for Arc Database API
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from arc.database.api import ArcAPI, SimulationStatus, OutcomeStatus, create_arc_api
from arc.database.client import ArcDBClient, DatabaseError


@pytest.fixture
def mock_db_client():
    """Create a mock database client."""
    client = AsyncMock(spec=ArcDBClient)
    
    # Create a proper mock for the engine
    client.engine = MagicMock()
    
    # Mock the async context manager for engine.begin()
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    
    # Create async context manager
    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_conn
    async_cm.__aexit__.return_value = None
    
    client.engine.begin.return_value = async_cm
    client.engine.text = MagicMock(side_effect=lambda x: x)
    
    return client


@pytest.fixture
def api(mock_db_client):
    """Create API instance with mock client."""
    return ArcAPI(mock_db_client)


@pytest.fixture
def sample_scenarios():
    """Sample scenario data."""
    return [
        {
            "id": "test_scenario_1",
            "name": "Test Scenario 1",
            "task_prompt": "Test prompt 1",
            "expected_tools": ["tool1"],
            "complexity_level": "simple"
        },
        {
            "id": "test_scenario_2",
            "name": "Test Scenario 2", 
            "task_prompt": "Test prompt 2",
            "expected_tools": ["tool2"],
            "complexity_level": "medium"
        }
    ]


@pytest.fixture
def sample_scenario_result():
    """Sample scenario execution result."""
    return {
        "scenario": {
            "id": "test_scenario_1",
            "name": "Test Scenario 1",
            "task_prompt": "Test prompt 1"
        },
        "trajectory": {
            "status": "success",
            "task_prompt": "Test prompt 1",
            "final_response": "Test response",
            "execution_time_seconds": 2.5,
            "full_trajectory": [
                {
                    "step": 1,
                    "type": "tool_call",
                    "tool": "tool1",
                    "tool_input": {"param": "value"},
                    "tool_output": "result"
                }
            ],
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "total_cost": 0.005
            }
        },
        "reliability_score": {
            "overall_score": 0.85,
            "grade": "B",
            "dimension_scores": {
                "tool_execution": 0.9,
                "response_quality": 0.8,
                "error_handling": 0.85,
                "performance": 0.9,
                "completeness": 0.8
            }
        },
        "detailed_trajectory": {
            "scenario_id": "test_scenario_1",
            "spans": [],
            "tool_calls": 1
        }
    }


@pytest.fixture
def sample_suite_result():
    """Sample suite execution result."""
    return {
        "average_reliability_score": 0.85,
        "total_cost_usd": 0.01,
        "parallel_execution_time": 5.0,
        "successful_runs": 2,
        "total_tokens_used": 300,
        "average_execution_time": 2.5,
        "average_dimension_scores": {
            "tool_execution": 0.9,
            "response_quality": 0.8,
            "error_handling": 0.85,
            "performance": 0.9,
            "completeness": 0.8
        },
        "speedup_factor": 2.0
    }


class TestArcAPI:
    """Test suite for ArcAPI."""
    
    @pytest.mark.asyncio
    async def test_start_simulation(self, api, mock_db_client, sample_scenarios):
        """Test starting a new simulation."""
        # Setup mock
        mock_db_client.create_simulation.return_value = "test-sim-id"
        
        # Get the mocked connection from fixture
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        
        # Call method
        result = await api.start_simulation(
            config_version_id="test-config-v1",
            scenarios=sample_scenarios,
            simulation_name="Test Simulation",
            modal_app_id="test-app",
            modal_environment="test",
            sandbox_instances=2,
            metadata={"test": "data"}
        )
        
        # Verify
        assert result["simulation_id"] == "test-sim-id"
        assert result["scenario_count"] == 2
        assert result["status"] == SimulationStatus.RUNNING.value
        assert "created_at" in result
        
        # Check DB calls
        mock_db_client.create_simulation.assert_called_once()
        assert mock_conn.execute.call_count >= 3  # UPDATE + 2 INSERTs
    
    @pytest.mark.asyncio
    async def test_record_scenario_outcome(self, api, mock_db_client, sample_scenario_result):
        """Test recording a single scenario outcome."""
        # Setup mock
        mock_db_client.record_outcome.return_value = "test-outcome-id"
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        
        # Call method
        outcome_id = await api.record_scenario_outcome(
            simulation_id="test-sim-id",
            scenario_result=sample_scenario_result,
            modal_call_id="test-modal-call",
            sandbox_id="test-sandbox"
        )
        
        # Verify
        assert outcome_id == "test-outcome-id"
        
        # Check outcome data passed to DB
        call_args = mock_db_client.record_outcome.call_args[0][0]
        assert call_args["simulation_id"] == "test-sim-id"
        assert call_args["scenario_id"] == "test_scenario_1"
        assert call_args["status"] == "success"
        assert call_args["reliability_score"] == 0.85
        assert call_args["tokens_used"] == 150
        assert call_args["cost_usd"] == 0.005
        assert call_args["modal_call_id"] == "test-modal-call"
        assert call_args["sandbox_id"] == "test-sandbox"
        
        # Check status updates
        assert mock_conn.execute.call_count == 2  # UPDATE simulations_scenarios + UPDATE simulations
    
    @pytest.mark.asyncio
    async def test_record_scenario_outcome_with_error(self, api, mock_db_client):
        """Test recording a failed scenario outcome."""
        # Setup error result
        error_result = {
            "scenario": {"id": "test_error", "name": "Error Test"},
            "trajectory": {
                "status": "error",
                "task_prompt": "Test",
                "final_response": "Error occurred",
                "execution_time_seconds": 0,
                "error": "TimeoutError: Request timed out",
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost": 0
                }
            },
            "reliability_score": {
                "overall_score": 0.0,
                "grade": "F",
                "dimension_scores": {}
            },
            "detailed_trajectory": {}
        }
        
        mock_db_client.record_outcome.return_value = "error-outcome-id"
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        
        # Call method
        outcome_id = await api.record_scenario_outcome(
            simulation_id="test-sim-id",
            scenario_result=error_result
        )
        
        # Verify error handling
        call_args = mock_db_client.record_outcome.call_args[0][0]
        assert call_args["status"] == "error"
        assert call_args["error_code"] == "TimeoutError: Request timed out"
        assert call_args["error_category"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_record_batch_outcomes(self, api, mock_db_client, sample_scenario_result):
        """Test batch recording of outcomes."""
        # Setup mock
        mock_db_client.record_outcomes_batch.return_value = ["outcome-1", "outcome-2"]
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        
        # Create two results
        results = [sample_scenario_result, sample_scenario_result]
        
        # Call method
        outcome_ids = await api.record_batch_outcomes(
            simulation_id="test-sim-id",
            scenario_results=results,
            modal_call_ids=["call-1", "call-2"]
        )
        
        # Verify
        assert outcome_ids == ["outcome-1", "outcome-2"]
        assert mock_db_client.record_outcomes_batch.called
        
        # Check batch data
        batch_data = mock_db_client.record_outcomes_batch.call_args[0][0]
        assert len(batch_data) == 2
        assert batch_data[0]["modal_call_id"] == "call-1"
        assert batch_data[1]["modal_call_id"] == "call-2"
    
    @pytest.mark.asyncio
    async def test_complete_simulation(self, api, mock_db_client, sample_suite_result):
        """Test completing a simulation."""
        # Get mock connection from fixture
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        
        # Call method
        result = await api.complete_simulation(
            simulation_id="test-sim-id",
            suite_result=sample_suite_result
        )
        
        # Verify
        assert result["simulation_id"] == "test-sim-id"
        assert result["status"] == SimulationStatus.COMPLETED.value
        assert result["overall_score"] == 0.85
        assert result["total_cost_usd"] == 0.01
        assert result["execution_time_seconds"] == 5.0
        assert "completed_at" in result
        
        # Check UPDATE call
        mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_simulation_status(self, api, mock_db_client):
        """Test getting simulation status."""
        # Setup mock query result
        mock_row = MagicMock()
        mock_row._mapping = {
            "simulation_id": "test-sim-id",
            "status": "running",
            "total_scenarios": 10,
            "completed_scenarios": 5,
            "successful_outcomes": 4,
            "failed_outcomes": 1,
            "overall_score": None,
            "total_cost_usd": None,
            "execution_time_ms": None,
            "modal_app_id": "test-app",
            "metadata": {"test": "data"},
            "created_at": datetime.now(timezone.utc),
            "started_at": datetime.now(timezone.utc),
            "completed_at": None
        }
        
        mock_result = MagicMock()
        mock_result.first.return_value = mock_row
        
        # Get mock connection and set up the result
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        mock_conn.execute.return_value = mock_result
        
        # Call method
        status = await api.get_simulation_status("test-sim-id")
        
        # Verify
        assert status["simulation_id"] == "test-sim-id"
        assert status["status"] == "running"
        assert status["progress_percentage"] == 50.0
        assert status["total_scenarios"] == 10
        assert status["completed_scenarios"] == 5
        assert status["successful_outcomes"] == 4
        assert status["failed_outcomes"] == 1
    
    @pytest.mark.asyncio
    async def test_get_scenario_outcomes(self, api, mock_db_client):
        """Test getting scenario outcomes."""
        # Setup mock query result
        mock_rows = [
            MagicMock(_mapping={
                "outcome_id": "outcome-1",
                "scenario_id": "scenario-1",
                "execution_time": datetime.now(timezone.utc),
                "status": "success",
                "reliability_score": 0.9,
                "execution_time_ms": 2500,
                "tokens_used": 150,
                "cost_usd": 0.005,
                "trajectory": '{"test": "data"}',
                "modal_call_id": "call-1",
                "error_code": None,
                "error_category": None,
                "metrics": '{"score": 0.9}',
                "scenario_name": "Test Scenario 1"
            }),
            MagicMock(_mapping={
                "outcome_id": "outcome-2",
                "scenario_id": "scenario-2",
                "execution_time": datetime.now(timezone.utc),
                "status": "error",
                "reliability_score": 0.0,
                "execution_time_ms": 1000,
                "tokens_used": 50,
                "cost_usd": 0.002,
                "trajectory": '{"error": "timeout"}',
                "modal_call_id": "call-2",
                "error_code": "TimeoutError",
                "error_category": "timeout",
                "metrics": '{"score": 0.0}',
                "scenario_name": "Test Scenario 2"
            })
        ]
        
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter(mock_rows)
        
        # Get mock connection and set up the result
        mock_conn = mock_db_client.engine.begin.return_value.__aenter__.return_value
        mock_conn.execute.return_value = mock_result
        
        # Call method - all outcomes
        outcomes = await api.get_scenario_outcomes("test-sim-id")
        assert len(outcomes) == 2
        assert outcomes[0]["status"] == "success"
        assert outcomes[1]["status"] == "error"
        
        # Call method - filtered by status
        mock_result.__iter__ = lambda self: iter([mock_rows[1]])  # Only error
        failures = await api.get_scenario_outcomes("test-sim-id", status_filter="error")
        assert len(failures) == 1
        assert failures[0]["status"] == "error"
    
    def test_categorize_error(self, api):
        """Test error categorization."""
        assert api._categorize_error("TimeoutError: Request timed out") == "timeout"
        assert api._categorize_error("Rate limit exceeded") == "rate_limit"
        assert api._categorize_error("Connection refused") == "network"
        assert api._categorize_error("JSON parsing error") == "parsing"
        assert api._categorize_error("Tool execution failed") == "tool_error"
        assert api._categorize_error("Out of memory") == "resource"
        assert api._categorize_error("Unknown error") == "other"
        assert api._categorize_error(None) is None


@pytest.mark.asyncio
async def test_create_arc_api():
    """Test API factory function."""
    with patch('arc.database.api.ArcDBClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.initialize.return_value = {"status": "healthy"}
        mock_client_class.return_value = mock_client
        
        api = await create_arc_api("test-connection-string")
        
        assert isinstance(api, ArcAPI)
        mock_client_class.assert_called_once_with("test-connection-string")
        mock_client.initialize.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 