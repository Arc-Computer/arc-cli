"""Unit tests for agent configuration ingestion."""

import pytest
from pathlib import Path
import tempfile
import yaml

from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer


class TestAgentConfigParser:
    """Test the flexible agent configuration parser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = AgentConfigParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def create_config_file(self, content: dict, filename: str = "test_config.yaml") -> Path:
        """Helper to create a temporary config file."""
        config_path = Path(self.temp_dir) / filename
        with open(config_path, 'w') as f:
            yaml.dump(content, f)
        return config_path
    
    def test_parse_minimal_config(self):
        """Test parsing minimal valid configuration."""
        config = {
            "model": "gpt-4.1",
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant",
            "tools": ["web_search", "calculator"]
        }
        config_path = self.create_config_file(config)
        
        result = self.parser.parse(config_path)
        
        assert result["model"] == "gpt-4.1"
        assert result["temperature"] == 0.7
        assert result["system_prompt"] == "You are a helpful assistant"
        assert result["tools"] == ["web_search", "calculator"]
    
    def test_parse_with_field_variations(self):
        """Test parsing with common field name variations."""
        config = {
            "model_name": "claude-3-opus",  # variation of "model"
            "temp": 0.5,  # variation of "temperature"
            "instructions": "Be concise and accurate",  # variation of "system_prompt"
            "functions": ["search", "compute"]  # variation of "tools"
        }
        config_path = self.create_config_file(config)
        
        result = self.parser.parse(config_path)
        
        assert result["model"] == "claude-3-opus"
        assert result["temperature"] == 0.5
        assert result["system_prompt"] == "Be concise and accurate"
        assert result["tools"] == ["search", "compute"]
    
    def test_parse_tools_as_dicts(self):
        """Test parsing tools defined as dictionaries."""
        config = {
            "model": "gpt-4.1",
            "temperature": 0.7,
            "system_prompt": "Assistant",
            "tools": [
                {"name": "web_search", "enabled": True},
                {"tool": "calculator", "description": "Math operations"},
                {"custom_tool": {"config": "data"}}  # Tool name from key
            ]
        }
        config_path = self.create_config_file(config)
        
        result = self.parser.parse(config_path)
        
        assert result["tools"] == ["web_search", "calculator", "custom_tool"]
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise appropriate errors."""
        config = {
            "model": "gpt-4.1",
            "temperature": 0.7
            # Missing system_prompt and tools
        }
        config_path = self.create_config_file(config)
        
        with pytest.raises(ValueError) as exc_info:
            self.parser.parse(config_path)
        
        assert "Missing required fields" in str(exc_info.value)
        assert "system_prompt" in str(exc_info.value)
        assert "tools" in str(exc_info.value)
    
    def test_extract_capabilities(self):
        """Test capability extraction from configuration."""
        config = {
            "model": "gpt-4.1",
            "temperature": 0.2,
            "system_prompt": "You are a financial analyst specializing in currency trading",
            "tools": ["currency_converter", "market_data_api", "calculate_roi"],
            "assumptions": ["All amounts in USD unless specified"]
        }
        config_path = self.create_config_file(config)
        
        parsed = self.parser.parse(config_path)
        capabilities = self.parser.extract_capabilities(parsed)
        
        assert "finance" in capabilities["domains"]
        assert "calculation" in capabilities["tool_categories"]
        assert "external_api" in capabilities["tool_categories"]
        assert "highly_deterministic" in capabilities["behavioral_traits"]
        assert "assumption_driven" in capabilities["behavioral_traits"]
        assert capabilities["complexity_level"] in ["simple", "moderate", "complex"]
    
    def test_invalid_yaml(self):
        """Test handling of invalid YAML files."""
        config_path = Path(self.temp_dir) / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ValueError) as exc_info:
            self.parser.parse(config_path)
        
        assert "Invalid YAML format" in str(exc_info.value)
    
    def test_temperature_validation_warning(self):
        """Test that unusual temperature values generate warnings."""
        config = {
            "model": "gpt-4.1",
            "temperature": 2.5,  # Outside typical range
            "system_prompt": "Assistant",
            "tools": ["search"]
        }
        config_path = self.create_config_file(config)
        
        result = self.parser.parse(config_path)
        warnings = self.parser.get_warnings()
        
        assert result["temperature"] == 2.5
        assert any("outside typical range" in w for w in warnings)


class TestConfigNormalizer:
    """Test the configuration normalizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = ConfigNormalizer()
    
    def test_normalize_model_names(self):
        """Test model name normalization."""
        config = {"model": "gpt-4.1"}
        result = self.normalizer.normalize(config)
        assert result["model"] == "openai/gpt-4.1"
        
        config = {"model": "claude-3.5-haiku"}
        result = self.normalizer.normalize(config)
        assert result["model"] == "anthropic/claude-sonnet-4"
    
    def test_normalize_tool_definitions(self):
        """Test tool definition normalization."""
        config = {
            "tools": ["search", "calculate"]
        }
        result = self.normalizer.normalize(config)
        
        assert "tools_normalized" in result
        assert len(result["tools_normalized"]) == 2
        assert result["tools_normalized"][0]["name"] == "search"
        assert result["tools_normalized"][0]["type"] == "function"
        assert result["tools_normalized"][0]["enabled"] is True
    
    def test_normalize_empty_tools_list(self):
        """Test normalization with empty tools list doesn't cause IndexError."""
        config = {
            "tools": []
        }
        # This should not raise an IndexError
        result = self.normalizer.normalize(config)
        
        assert "tools" in result
        assert result["tools"] == []
        # tools_normalized is not added for empty tools list
        assert "tools_normalized" not in result
    
    def test_add_arc_metadata(self):
        """Test Arc metadata addition."""
        config = {"model": "gpt-4.1"}
        result = self.normalizer.normalize(config)
        
        assert "arc_metadata" in result
        assert result["arc_metadata"]["version"] == "1.0"
        assert result["arc_metadata"]["profile_type"] == "agent"
        assert result["arc_metadata"]["normalization_applied"] is True
    
    def test_ensure_consistent_types(self):
        """Test type consistency enforcement."""
        config = {
            "temperature": "0.7",  # String instead of float
            "max_tokens": "1000",  # String instead of int
            "top_p": "invalid",    # Invalid value
            "tools": "single_tool" # String instead of list
        }
        result = self.normalizer.normalize(config)
        
        assert isinstance(result["temperature"], float)
        assert result["temperature"] == 0.7
        assert isinstance(result["max_tokens"], int)
        assert result["max_tokens"] == 1000
        assert result["top_p"] == 1.0  # Default due to invalid value
        assert isinstance(result["tools"], list)
        assert result["tools"] == ["single_tool"]
    
    def test_create_arc_profile(self):
        """Test complete Arc profile creation."""
        config = {
            "model": "gpt-4.1",
            "temperature": 0.7,
            "system_prompt": "Financial assistant",
            "tools": ["calculator", "currency_api"]
        }
        capabilities = {
            "domains": ["finance"],
            "tool_categories": {"calculation": ["calculator"], "external_api": ["currency_api"]},
            "behavioral_traits": ["assumption_driven"],
            "complexity_level": "moderate"
        }
        
        profile = self.normalizer.create_arc_profile(config, capabilities)
        
        assert "configuration" in profile
        assert "capabilities" in profile
        assert "test_parameters" in profile
        assert "optimization_targets" in profile
        
        # Check test parameters
        assert profile["test_parameters"]["scenario_count"] == 50
        assert profile["test_parameters"]["require_deterministic"] is True
        assert profile["test_parameters"]["mock_external_calls"] is True
    
    def test_identify_optimization_targets(self):
        """Test optimization target identification."""
        # High temperature config with expensive model
        config = {"temperature": 1.5, "model": "openai/gpt-4.1", "tools": list(range(15))}
        capabilities = {"domains": ["finance"]}
        
        profile = self.normalizer.create_arc_profile(config, capabilities)
        targets = profile["optimization_targets"]
        
        assert "temperature_reduction" in targets
        assert "cost_optimization" in targets
        assert "tool_consolidation" in targets
        assert "numerical_precision" in targets
    
    def test_validate_normalized_config(self):
        """Test validation of normalized configurations."""
        # Valid config
        valid_config = {
            "model": "gpt-4.1",
            "temperature": 0.7,
            "system_prompt": "Assistant",
            "tools": ["search"],
            "arc_metadata": {"version": "1.0"}
        }
        assert self.normalizer.validate_normalized_config(valid_config) is True
        
        # Invalid config - missing field
        invalid_config = valid_config.copy()
        del invalid_config["arc_metadata"]
        assert self.normalizer.validate_normalized_config(invalid_config) is False
        
        # Invalid config - wrong type
        invalid_config = valid_config.copy()
        invalid_config["tools"] = "not_a_list"
        assert self.normalizer.validate_normalized_config(invalid_config) is False