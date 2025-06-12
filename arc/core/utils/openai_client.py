"""OpenAI client utilities with background mode support for o3 models."""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

import openai
from openai.types.responses import Response

logger = logging.getLogger(__name__)


class BackgroundTaskStatus(Enum):
    """Status of a background reasoning task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    """Represents an o3 background reasoning task."""
    task_id: str
    status: BackgroundTaskStatus
    created_at: float
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_estimate: float = 0.0


class OpenAIBackgroundClient:
    """OpenAI client with background mode support for o3 models."""
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 600):
        """Initialize the background client.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            timeout: Maximum time to wait for background tasks (seconds)
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.timeout = timeout
        self.active_tasks: Dict[str, BackgroundTask] = {}
        
    async def create_background_analysis(
        self,
        prompt: str,
        model: str = "o3",
        max_reasoning_tokens: int = 100000,
        temperature: float = 0.7
    ) -> str:
        """Create a background reasoning task with o3.
        
        Args:
            prompt: Analysis prompt for the o3 model
            model: Model to use (o3 or o3-mini)
            max_reasoning_tokens: Maximum tokens for reasoning phase
            temperature: Sampling temperature
            
        Returns:
            Task ID for polling
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                background=True,
                max_reasoning_tokens=max_reasoning_tokens,
                temperature=temperature
            )
            
            task_id = response.id
            
            # Estimate cost based on model and prompt length
            cost_estimate = self._estimate_cost(prompt, model)
            
            task = BackgroundTask(
                task_id=task_id,
                status=BackgroundTaskStatus.PENDING,
                created_at=time.time(),
                cost_estimate=cost_estimate
            )
            
            self.active_tasks[task_id] = task
            
            logger.info(f"Created background task {task_id} with estimated cost ${cost_estimate:.3f}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create background task: {str(e)}")
            raise
    
    async def poll_task(self, task_id: str, poll_interval: int = 5) -> BackgroundTask:
        """Poll a background task until completion.
        
        Args:
            task_id: ID of the task to poll
            poll_interval: Seconds between polls
            
        Returns:
            Completed BackgroundTask
        """
        if task_id not in self.active_tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        task = self.active_tasks[task_id]
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > self.timeout:
                task.status = BackgroundTaskStatus.FAILED
                task.error = f"Task timed out after {self.timeout}s"
                logger.error(f"Task {task_id} timed out")
                return task
            
            try:
                # Check task status
                response = await self.client.responses.retrieve(task_id)
                
                if hasattr(response, 'status'):
                    if response.status == 'completed':
                        task.status = BackgroundTaskStatus.COMPLETED
                        task.completed_at = time.time()
                        task.result = {
                            "content": response.output.content,
                            "usage": response.usage.__dict__ if hasattr(response, 'usage') else None
                        }
                        logger.info(f"Task {task_id} completed in {task.completed_at - task.created_at:.1f}s")
                        return task
                    elif response.status == 'failed':
                        task.status = BackgroundTaskStatus.FAILED
                        task.error = getattr(response, 'error', 'Unknown error')
                        logger.error(f"Task {task_id} failed: {task.error}")
                        return task
                    else:
                        task.status = BackgroundTaskStatus.IN_PROGRESS
                
            except Exception as e:
                logger.warning(f"Error polling task {task_id}: {str(e)}")
                # Continue polling unless it's a permanent error
                if "not found" in str(e).lower():
                    task.status = BackgroundTaskStatus.FAILED
                    task.error = f"Task not found: {str(e)}"
                    return task
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def get_task_status(self, task_id: str) -> BackgroundTask:
        """Get current status of a background task without polling."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        return self.active_tasks[task_id]
    
    def _estimate_cost(self, prompt: str, model: str) -> float:
        """Estimate cost for background task.
        
        Args:
            prompt: Input prompt
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (4 chars per token)
        input_tokens = len(prompt) / 4
        
        # Model pricing (per 1M tokens)
        if model == "o3":
            input_cost_per_million = 2.00
            output_cost_per_million = 8.00
        elif model == "o3-mini":
            # Estimated pricing for o3-mini (not yet available)
            input_cost_per_million = 0.20
            output_cost_per_million = 0.80
        else:
            # Default to o3 pricing
            input_cost_per_million = 2.00
            output_cost_per_million = 8.00
        
        # Assume roughly equal input/output tokens for analysis tasks
        estimated_output_tokens = input_tokens * 0.8
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (estimated_output_tokens / 1_000_000) * output_cost_per_million
        
        return input_cost + output_cost
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Remove old completed tasks from memory.
        
        Args:
            max_age_hours: Maximum age of completed tasks to keep
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task in self.active_tasks.items():
            if (task.status in [BackgroundTaskStatus.COMPLETED, BackgroundTaskStatus.FAILED] and
                task.created_at < cutoff_time):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.active_tasks[task_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old background tasks")


# Global client instance
_background_client: Optional[OpenAIBackgroundClient] = None


def get_background_client() -> OpenAIBackgroundClient:
    """Get or create the global background client instance."""
    global _background_client
    if _background_client is None:
        _background_client = OpenAIBackgroundClient()
    return _background_client