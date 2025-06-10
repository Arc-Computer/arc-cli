"""
Trajectory Capture and Serialization for Arc-Eval Production
Production version adapted from experiments/src/tracing/trajectory_capture.py
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
    BatchSpanProcessor
)
from opentelemetry.sdk.resources import Resource


@dataclass
class SpanData:
    """Serializable representation of an OpenTelemetry span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    status_description: Optional[str]
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    kind: str
    resource_attributes: Dict[str, Any]
    
    @classmethod
    def from_readable_span(cls, span: ReadableSpan) -> 'SpanData':
        """Convert OpenTelemetry ReadableSpan to serializable SpanData"""
        return cls(
            span_id=format(span.context.span_id, '016x'),
            trace_id=format(span.context.trace_id, '032x'),
            parent_span_id=format(span.parent.span_id, '016x') if span.parent else None,
            name=span.name,
            start_time=datetime.fromtimestamp(span.start_time / 1e9).isoformat() if span.start_time else "",
            end_time=datetime.fromtimestamp(span.end_time / 1e9).isoformat() if span.end_time else "",
            duration_ms=(span.end_time - span.start_time) / 1e6 if span.end_time and span.start_time else 0,
            status=span.status.status_code.name,
            status_description=span.status.description,
            attributes=dict(span.attributes) if span.attributes else {},
            events=[{
                "name": event.name,
                "timestamp": datetime.fromtimestamp(event.timestamp / 1e9).isoformat(),
                "attributes": dict(event.attributes) if event.attributes else {}
            } for event in span.events],
            links=[{
                "trace_id": format(link.context.trace_id, '032x'),
                "span_id": format(link.context.span_id, '016x'),
                "attributes": dict(link.attributes) if link.attributes else {}
            } for link in span.links],
            kind=span.kind.name,
            resource_attributes=dict(span.resource.attributes) if span.resource else {}
        )


@dataclass
class TrajectoryData:
    """Complete trajectory data for a simulation run"""
    scenario_id: str
    scenario_name: str
    task_prompt: str
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    
    # Execution trajectory
    tool_calls: List[Dict[str, Any]]
    llm_interactions: List[Dict[str, Any]]
    decision_points: List[Dict[str, Any]]
    
    # OpenTelemetry data
    spans: List[SpanData]
    trace_tree: Dict[str, List[str]]  # parent_id -> [child_ids]
    
    # Metrics
    token_usage: Dict[str, int]
    latency_breakdown: Dict[str, float]
    error_events: List[Dict[str, Any]]
    
    # Agent behavior analysis
    tool_selection_rationale: List[Dict[str, Any]]
    parameter_choices: List[Dict[str, Any]]
    error_recovery_attempts: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "task_prompt": self.task_prompt,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tool_calls": self.tool_calls,
            "llm_interactions": self.llm_interactions,
            "decision_points": self.decision_points,
            "spans": [asdict(span) for span in self.spans],
            "trace_tree": self.trace_tree,
            "token_usage": self.token_usage,
            "latency_breakdown": self.latency_breakdown,
            "error_events": self.error_events,
            "tool_selection_rationale": self.tool_selection_rationale,
            "parameter_choices": self.parameter_choices,
            "error_recovery_attempts": self.error_recovery_attempts
        }


class TrajectoryCapture:
    """Captures complete execution trajectory with 4-level behavioral tracking"""
    
    def __init__(self):
        """Initialize trajectory capture with OpenTelemetry"""
        # Create a custom span exporter that stores spans in memory
        self.span_storage = []
        
        # Set up tracing
        resource = Resource.create({
            "service.name": "arc-eval-production",
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        provider = TracerProvider(resource=resource)
        
        # Add our custom exporter
        exporter = InMemorySpanExporter(self.span_storage)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer("arc-eval.trajectory", "1.0.0")
        
        # Storage for trajectory events
        self.tool_calls = []
        self.llm_interactions = []
        self.decision_points = []
        self.error_events = []
        self.tool_selection_rationale = []
        self.parameter_choices = []
        self.error_recovery_attempts = []
        
        # Metrics tracking
        self.latency_breakdown = defaultdict(float)
        self.start_time = time.time()
    
    def capture_tool_call(self, tool_name: str, tool_input: Any, tool_output: str, 
                         duration_ms: float, success: bool):
        """Capture tool call event with full context"""
        self.tool_calls.append({
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "duration_ms": duration_ms,
            "success": success,
            "trace_context": self._get_current_trace_context()
        })
        
        # Update latency breakdown
        self.latency_breakdown[f"tool_{tool_name}"] += duration_ms
        
        # Add as trace event
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(
                "tool_call",
                attributes={
                    "tool.name": tool_name,
                    "tool.duration_ms": duration_ms,
                    "tool.success": success
                }
            )
    
    def capture_llm_interaction(self, model: str, prompt: str, response: str,
                               tokens_in: int, tokens_out: int, duration_ms: float):
        """Capture LLM interaction with token usage"""
        self.llm_interactions.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt": prompt,
            "response": response,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration_ms": duration_ms,
            "trace_context": self._get_current_trace_context()
        })
        
        # Update latency breakdown
        self.latency_breakdown["llm_inference"] += duration_ms
        
        # Add as trace event
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(
                "llm_interaction",
                attributes={
                    "llm.model": model,
                    "llm.tokens_in": tokens_in,
                    "llm.tokens_out": tokens_out,
                    "llm.duration_ms": duration_ms
                }
            )
    
    def capture_decision_point(self, decision_type: str, options: List[str],
                              selected: str, rationale: str):
        """Capture agent decision point"""
        self.decision_points.append({
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "options": options,
            "selected": selected,
            "rationale": rationale,
            "trace_context": self._get_current_trace_context()
        })
        
        # Add as trace event
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(
                "decision_point",
                attributes={
                    "decision.type": decision_type,
                    "decision.selected": selected,
                    "decision.options_count": len(options)
                }
            )
    
    def capture_error(self, error_type: str, error_message: str, 
                     recovery_attempted: bool, recovery_successful: bool = False):
        """Capture error event and recovery attempt"""
        error_event = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "trace_context": self._get_current_trace_context()
        }
        
        self.error_events.append(error_event)
        
        if recovery_attempted:
            self.error_recovery_attempts.append(error_event)
        
        # Add as trace event
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(
                "error",
                attributes={
                    "error.type": error_type,
                    "error.message": error_message,
                    "error.recovery_attempted": recovery_attempted,
                    "error.recovery_successful": recovery_successful
                }
            )
            
            # Set span status to error
            current_span.set_status(Status(StatusCode.ERROR, error_message))
    
    def create_trajectory(self, scenario: Dict[str, Any], status: str,
                         start_time: datetime, end_time: datetime,
                         token_usage: Dict[str, int]) -> TrajectoryData:
        """Create complete trajectory data object"""
        # Build trace tree from spans
        trace_tree = self._build_trace_tree()
        
        return TrajectoryData(
            scenario_id=scenario.get("id", "unknown"),
            scenario_name=scenario.get("name", "unknown"),
            task_prompt=scenario.get("task_prompt", ""),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_ms=(end_time - start_time).total_seconds() * 1000,
            status=status,
            tool_calls=self.tool_calls,
            llm_interactions=self.llm_interactions,
            decision_points=self.decision_points,
            spans=[SpanData.from_readable_span(span) for span in self.span_storage],
            trace_tree=trace_tree,
            token_usage=token_usage,
            latency_breakdown=dict(self.latency_breakdown),
            error_events=self.error_events,
            tool_selection_rationale=self.tool_selection_rationale,
            parameter_choices=self.parameter_choices,
            error_recovery_attempts=self.error_recovery_attempts
        )
    
    def _get_current_trace_context(self) -> Dict[str, str]:
        """Get current trace context"""
        span = trace.get_current_span()
        if span:
            span_context = span.get_span_context()
            return {
                "trace_id": format(span_context.trace_id, '032x'),
                "span_id": format(span_context.span_id, '016x')
            }
        return {}
    
    def _build_trace_tree(self) -> Dict[str, List[str]]:
        """Build parent->children trace tree from spans"""
        tree = defaultdict(list)
        for span in self.span_storage:
            if hasattr(span, 'parent') and span.parent:
                parent_id = format(span.parent.span_id, '016x')
                span_id = format(span.context.span_id, '016x')
                tree[parent_id].append(span_id)
        return dict(tree)


class InMemorySpanExporter(SpanExporter):
    """Custom span exporter that stores spans in memory for trajectory capture"""
    
    def __init__(self, storage: List[ReadableSpan]):
        self.storage = storage
    
    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        """Export spans to in-memory storage"""
        self.storage.extend(spans)
        return SpanExportResult.SUCCESS
    
    def shutdown(self) -> None:
        """Shutdown the exporter"""
        pass