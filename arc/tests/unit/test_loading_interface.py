"""Unit tests for Arc loading interface components."""

import pytest
from unittest.mock import MagicMock
from rich.console import Console

from arc.cli.loading_interface import (
    ConfigAnalysisLoader,
    ExecutionProgressLoader,
    StreamingResultsDisplay
)


class TestConfigAnalysisLoader:
    """Test ConfigAnalysisLoader functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console."""
        console = MagicMock(spec=Console)
        console.get_time = MagicMock(return_value=0.0)
        console.is_jupyter = False
        console.is_terminal = True
        console.is_dumb_terminal = False
        console.is_interactive = True
        return console
    
    @pytest.fixture
    def loader(self, mock_console):
        """Create ConfigAnalysisLoader instance."""
        return ConfigAnalysisLoader(mock_console)
    
    @pytest.mark.asyncio
    async def test_analyze_config_with_progress_success(self, loader):
        """Test successful config analysis with progress."""
        # Mock parser and normalizer
        mock_parser = MagicMock()
        mock_parser.parse.return_value = {"test": "config"}
        mock_parser.extract_capabilities.return_value = {
            "domains": ["finance"],
            "complexity_level": "medium"
        }
        
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.return_value = {"normalized": "config"}
        
        # Execute analysis
        result = await loader.analyze_config_with_progress(
            "test.yaml", mock_parser, mock_normalizer
        )
        
        # Verify calls
        mock_parser.parse.assert_called_once_with("test.yaml")
        mock_parser.extract_capabilities.assert_called_once()
        mock_normalizer.normalize.assert_called_once()
        
        # Verify result structure
        assert "configuration" in result
        assert "capabilities" in result
        assert result["config_path"] == "test.yaml"
    
    @pytest.mark.asyncio
    async def test_analyze_config_with_progress_parse_error(self, loader):
        """Test config analysis with parse error."""
        # Mock parser that raises exception
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = ValueError("Parse error")
        
        mock_normalizer = MagicMock()
        
        # Execute and expect exception
        with pytest.raises(ValueError, match="Parse error"):
            await loader.analyze_config_with_progress(
                "test.yaml", mock_parser, mock_normalizer
            )
    
    def test_display_config_summary(self, loader, mock_console):
        """Test config summary display."""
        agent_profile = {
            "configuration": {
                "model": "gpt-4",
                "temperature": 0.7,
                "tools": ["tool1", "tool2", "tool3"]
            },
            "capabilities": {
                "domains": ["finance", "healthcare"],
                "complexity_level": "advanced",
                "tool_categories": {
                    "data": ["tool1"],
                    "api": ["tool2", "tool3"]
                }
            },
            "normalizer_enhancements": [
                "Enhancement 1",
                "Enhancement 2"
            ]
        }
        
        loader.display_config_summary(agent_profile)
        
        # Verify console was called
        assert mock_console.print.called
        # Check that multiple print calls were made
        assert len(mock_console.print.call_args_list) > 0


class TestExecutionProgressLoader:
    """Test ExecutionProgressLoader functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console."""
        console = MagicMock(spec=Console)
        console.get_time = MagicMock(return_value=0.0)
        console.is_jupyter = False
        console.is_terminal = True
        console.is_dumb_terminal = False
        console.is_interactive = True
        return console
    
    @pytest.fixture
    def loader(self, mock_console):
        """Create ExecutionProgressLoader instance."""
        return ExecutionProgressLoader(mock_console)
    
    def test_create_execution_progress(self, loader):
        """Test progress tracker creation."""
        progress = loader.create_execution_progress(50)
        
        # Verify progress object
        assert progress is not None
        assert hasattr(progress, 'add_task')
    
    def test_display_live_metrics(self, loader):
        """Test live metrics display."""
        metrics = {
            "completed": 25,
            "total": 50,
            "cost": 0.0156,
            "estimated_cost": 0.0312,
            "elapsed_time": 32.5,
            "failures": 3,
            "active_containers": 10,
            "max_containers": 20
        }
        
        panel = loader.display_live_metrics(metrics)
        
        # Verify panel structure
        assert panel is not None
        assert hasattr(panel, 'renderable')
        assert "25/50" in str(panel.renderable)
        assert "$0.02" in str(panel.renderable)  # format_currency rounds to 2 decimals
    
    def test_display_reliability_breakdown(self, loader):
        """Test reliability breakdown display."""
        reliability_data = {
            "overall_score": 73,
            "completed": 36,
            "total": 50,
            "dimension_scores": {
                "Tool Execution": 80,
                "Response Quality": 60,
                "Error Handling": 40
            },
            "dimension_weights": {
                "Tool Execution": 30,
                "Response Quality": 25,
                "Error Handling": 20
            }
        }
        
        panel = loader.display_reliability_breakdown(reliability_data)
        
        # Verify panel structure
        assert panel is not None
        assert "73%" in str(panel.renderable)
        assert "Tool Execution" in str(panel.renderable)
    
    def test_display_assumption_alerts(self, loader):
        """Test assumption alerts display."""
        violations = [
            {
                "type": "currency",
                "severity": "high",
                "description": "Agent assumes USD for all transactions"
            },
            {
                "type": "timezone",
                "severity": "medium",
                "description": "Agent uses system timezone without validation"
            }
        ]
        
        panel = loader.display_assumption_alerts(violations)
        
        # Verify panel structure
        assert panel is not None
        assert "CURRENCY" in str(panel.renderable)
        assert "USD for all transactions" in str(panel.renderable)
    
    def test_display_assumption_alerts_empty(self, loader):
        """Test assumption alerts with no violations."""
        panel = loader.display_assumption_alerts([])
        assert panel is None
    
    def test_format_duration(self, loader):
        """Test duration formatting."""
        assert loader._format_duration(65) == "01:05"
        assert loader._format_duration(3661) == "61:01"
        assert loader._format_duration(0) == "00:00"


class TestStreamingResultsDisplay:
    """Test StreamingResultsDisplay functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console."""
        console = MagicMock(spec=Console)
        console.get_time = MagicMock(return_value=0.0)
        console.is_jupyter = False
        console.is_terminal = True
        console.is_dumb_terminal = False
        console.is_interactive = True
        return console
    
    @pytest.fixture
    def display(self, mock_console):
        """Create StreamingResultsDisplay instance."""
        return StreamingResultsDisplay(mock_console)
    
    def test_add_result(self, display):
        """Test adding results to buffer."""
        result = {
            "scenario_id": "test_001",
            "success": True,
            "reliability_score": 85
        }
        
        display.add_result(result)
        assert len(display.results_buffer) == 1
        assert display.results_buffer[0] == result
    
    def test_add_result_buffer_limit(self, display):
        """Test buffer limit of 10 results."""
        # Add 15 results
        for i in range(15):
            display.add_result({
                "scenario_id": f"test_{i:03d}",
                "success": True
            })
        
        # Verify only last 10 are kept
        assert len(display.results_buffer) == 10
        assert display.results_buffer[0]["scenario_id"] == "test_005"
        assert display.results_buffer[9]["scenario_id"] == "test_014"
    
    def test_create_results_table(self, display):
        """Test results table creation."""
        # Add some results
        display.add_result({
            "scenario_id": "test_success",
            "success": True,
            "reliability_score": 90,
            "primary_issue": None
        })
        display.add_result({
            "scenario_id": "test_failure_long_name",
            "success": False,
            "reliability_score": 30,
            "primary_issue": "Currency assumption violation detected"
        })
        
        table = display.create_results_table()
        
        # Verify table structure
        assert table is not None
        assert hasattr(table, 'add_column')
        assert hasattr(table, 'add_row')