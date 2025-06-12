"""
Pydantic models for agent and tool configurations.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolConfig(BaseModel):
    """Configuration for a single tool."""
    name: str = Field(..., description="The name of the tool.")
    description: str | None = Field(None, description="A description of what the tool does.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="The parameters the tool accepts, in JSON schema format.")


class AgentConfig(BaseModel):
    """Represents the complete configuration for an agent."""
    model: str = Field(..., description="The name of the language model to use (e.g., 'openai/gpt-4.1').")
    temperature: float = Field(0.0, description="The sampling temperature for the model.")
    tools: list[ToolConfig] = Field(default_factory=list, description="A list of tools available to the agent.")
    version_id: str | None = None # This will be assigned by the system
    system_prompt: str | None = Field(None, description="The system prompt for the agent.")

    model_config = ConfigDict(extra="allow")  # Allow extra fields to be present in the config

    def to_dict(self) -> dict[str, Any]:
        """Serializes the config to a dictionary."""
        return self.model_dump(by_alias=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Deserializes a dictionary into an AgentConfig object."""
        return cls(**data)

    @field_validator('tools', mode='before')
    @classmethod
    def validate_tools(cls, v: Any) -> list[ToolConfig]:
        """Convert string tool names to ToolConfig objects."""
        if not isinstance(v, list):
            raise ValueError("Tools must be a list")

        validated_tools = []
        for tool in v:
            if isinstance(tool, str):
                # Convert string to ToolConfig
                validated_tools.append(ToolConfig(name=tool, description=None))
            elif isinstance(tool, dict):
                # Convert dict to ToolConfig
                validated_tools.append(ToolConfig(**tool))
            elif isinstance(tool, ToolConfig):
                # Already a ToolConfig
                validated_tools.append(tool)
            else:
                # Invalid format
                raise ValueError(f"Tool must be a string, dict, or ToolConfig object, got {type(tool)}")

        return validated_tools
