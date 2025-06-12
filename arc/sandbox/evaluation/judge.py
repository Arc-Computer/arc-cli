"""Calibrated LLM judge ensemble for reliable evaluation with chain-of-thought reasoning."""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import openai
from openai import AsyncOpenAI


class JudgeConfidence(Enum):
    """Confidence levels for judge evaluations."""
    HIGH = "high"
    MEDIUM = "medium"  
    LOW = "low"


@dataclass
class JudgeEvaluation:
    """Result from a single judge evaluation."""
    judge_id: str
    dimension_scores: Dict[str, float]
    overall_score: float
    confidence: JudgeConfidence
    reasoning: str
    chain_of_thought: List[str] = field(default_factory=list)
    calibration_weight: float = 1.0


@dataclass
class EnsembleEvaluation:
    """Consensus result from calibrated judge ensemble."""
    dimension_scores: Dict[str, float]
    overall_score: float
    confidence_interval: tuple[float, float]
    consensus_reasoning: str
    individual_evaluations: List[JudgeEvaluation] = field(default_factory=list)
    calibration_quality: float = 1.0


class CalibratedJudgeEnsemble:
    """Ensemble of calibrated LLM judges for reliable evaluation."""
    
    def __init__(self, models: Optional[List[str]] = None):
        self.models = models or [
            "gpt-4o-mini",  # Fast and efficient for evaluation
            "gpt-4o-mini",  # Second instance for consensus
            "gpt-4o-mini"   # Third instance for tie-breaking
        ]
        self.client = AsyncOpenAI()
        self.calibration_history = {}
        
        # Importance weights for different aspects
        self.importance_weights = {
            "tool_execution": {
                "correct_tool_selection": 0.4,
                "proper_parameters": 0.3,
                "error_handling": 0.3
            },
            "response_quality": {
                "accuracy": 0.4,
                "completeness": 0.3,
                "clarity": 0.3
            },
            "error_handling": {
                "graceful_degradation": 0.5,
                "recovery_attempts": 0.3,
                "error_communication": 0.2
            },
            "performance": {
                "speed": 0.4,
                "efficiency": 0.3,
                "resource_usage": 0.3
            },
            "completeness": {
                "task_fulfillment": 0.6,
                "requirement_coverage": 0.4
            }
        }
    
    async def evaluate_trajectory(
        self,
        trajectory: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> EnsembleEvaluation:
        """Evaluate trajectory using calibrated judge ensemble."""
        
        # Step 1: Create importance-weighted prompt
        weighted_prompt = self._create_weighted_prompt(trajectory, scenario)
        
        # Step 2: Multi-judge evaluation with CoT
        judge_evaluations = []
        tasks = []
        
        for i, model in enumerate(self.models):
            task = self._evaluate_with_single_judge(
                judge_id=f"judge_{i}_{model}",
                model=model,
                prompt=weighted_prompt,
                trajectory=trajectory,
                scenario=scenario
            )
            tasks.append(task)
        
        judge_evaluations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_evaluations = [
            eval for eval in judge_evaluations 
            if isinstance(eval, JudgeEvaluation)
        ]
        
        if not valid_evaluations:
            # Fallback to basic scoring if all judges fail
            return self._create_fallback_evaluation(trajectory, scenario)
        
        # Step 3: Calibration-aware consensus
        consensus = self._compute_calibrated_consensus(
            evaluations=valid_evaluations,
            calibration_data=self.calibration_history
        )
        
        return consensus
    
    def _create_weighted_prompt(
        self,
        trajectory: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> str:
        """Create importance-weighted evaluation prompt."""
        
        prompt = f"""You are an expert AI system evaluator. Evaluate this agent execution using the 5-dimensional reliability framework.

**SCENARIO:**
Task: {scenario.get('task_prompt', 'Not provided')}
Expected Tools: {scenario.get('expected_tools', [])}
Success Criteria: {scenario.get('success_criteria', {})}

**EXECUTION TRAJECTORY:**
Status: {trajectory.get('status', 'unknown')}
Final Response: {trajectory.get('final_response', 'Not provided')}
Execution Time: {trajectory.get('execution_time_seconds', 0)} seconds
Tool Calls: {len([e for e in trajectory.get('full_trajectory', []) if e.get('type') == 'tool_call'])}

**EVALUATION DIMENSIONS (score 0-100 for each):**

1. **Tool Execution (30% weight):**
   - Correct tool selection (40% importance)
   - Proper parameters (30% importance) 
   - Error handling (30% importance)

2. **Response Quality (25% weight):**
   - Accuracy (40% importance)
   - Completeness (30% importance)
   - Clarity (30% importance)

3. **Error Handling (20% weight):**
   - Graceful degradation (50% importance)
   - Recovery attempts (30% importance)
   - Error communication (20% importance)

4. **Performance (15% weight):**
   - Speed (40% importance)
   - Efficiency (30% importance)
   - Resource usage (30% importance)

5. **Completeness (10% weight):**
   - Task fulfillment (60% importance)
   - Requirement coverage (40% importance)

**CHAIN-OF-THOUGHT EVALUATION:**
Think step by step through each dimension. For each:
1. Identify what went well
2. Identify issues/problems
3. Assess against importance weights
4. Assign score with justification

**OUTPUT FORMAT:**
Return a JSON object with:
{{
    "chain_of_thought": ["step1 reasoning", "step2 reasoning", ...],
    "dimension_scores": {{
        "tool_execution": score,
        "response_quality": score,
        "error_handling": score, 
        "performance": score,
        "completeness": score
    }},
    "overall_reasoning": "concise summary of key findings",
    "confidence": "high|medium|low"
}}"""

        return prompt
    
    async def _evaluate_with_single_judge(
        self,
        judge_id: str,
        model: str,
        prompt: str,
        trajectory: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate with a single judge using chain-of-thought."""
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert AI system evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Calculate overall score using weights
            dimension_scores = result.get("dimension_scores", {})
            weights = {
                "tool_execution": 0.3,
                "response_quality": 0.25,
                "error_handling": 0.2,
                "performance": 0.15,
                "completeness": 0.1
            }
            
            overall_score = sum(
                dimension_scores.get(dim, 0) * weight
                for dim, weight in weights.items()
            )
            
            confidence_str = result.get("confidence", "medium").lower()
            confidence = JudgeConfidence(confidence_str) if confidence_str in ["high", "medium", "low"] else JudgeConfidence.MEDIUM
            
            return JudgeEvaluation(
                judge_id=judge_id,
                dimension_scores=dimension_scores,
                overall_score=overall_score,
                confidence=confidence,
                reasoning=result.get("overall_reasoning", "No reasoning provided"),
                chain_of_thought=result.get("chain_of_thought", []),
                calibration_weight=self._get_judge_calibration_weight(judge_id)
            )
            
        except Exception as e:
            # Return fallback evaluation on error
            return JudgeEvaluation(
                judge_id=judge_id,
                dimension_scores={
                    "tool_execution": 50.0,
                    "response_quality": 50.0,
                    "error_handling": 50.0,
                    "performance": 50.0,
                    "completeness": 50.0
                },
                overall_score=50.0,
                confidence=JudgeConfidence.LOW,
                reasoning=f"Judge evaluation failed: {str(e)}",
                chain_of_thought=[],
                calibration_weight=0.5
            )
    
    def _compute_calibrated_consensus(
        self,
        evaluations: List[JudgeEvaluation],
        calibration_data: Dict[str, Any]
    ) -> EnsembleEvaluation:
        """Compute calibration-aware consensus from multiple judge evaluations."""
        
        if not evaluations:
            return self._create_fallback_evaluation({}, {})
        
        # Weight evaluations by calibration quality and confidence
        total_weight = 0
        weighted_dimension_scores = {}
        
        for eval in evaluations:
            weight = eval.calibration_weight
            if eval.confidence == JudgeConfidence.HIGH:
                weight *= 1.2
            elif eval.confidence == JudgeConfidence.LOW:
                weight *= 0.8
            
            total_weight += weight
            
            for dim, score in eval.dimension_scores.items():
                if dim not in weighted_dimension_scores:
                    weighted_dimension_scores[dim] = 0
                weighted_dimension_scores[dim] += score * weight
        
        # Calculate consensus scores
        consensus_dimension_scores = {}
        for dim, weighted_sum in weighted_dimension_scores.items():
            consensus_dimension_scores[dim] = weighted_sum / total_weight if total_weight > 0 else 50.0
        
        # Calculate overall consensus score
        weights = {
            "tool_execution": 0.3,
            "response_quality": 0.25,
            "error_handling": 0.2,
            "performance": 0.15,
            "completeness": 0.1
        }
        
        consensus_overall = sum(
            consensus_dimension_scores.get(dim, 0) * weight
            for dim, weight in weights.items()
        )
        
        # Calculate confidence interval (simplified)
        scores = [eval.overall_score for eval in evaluations]
        min_score = min(scores) if scores else consensus_overall
        max_score = max(scores) if scores else consensus_overall
        
        # Generate consensus reasoning
        consensus_reasoning = self._generate_consensus_reasoning(evaluations, consensus_dimension_scores)
        
        return EnsembleEvaluation(
            dimension_scores=consensus_dimension_scores,
            overall_score=consensus_overall,
            confidence_interval=(min_score, max_score),
            consensus_reasoning=consensus_reasoning,
            individual_evaluations=evaluations,
            calibration_quality=total_weight / len(evaluations) if evaluations else 1.0
        )
    
    def _get_judge_calibration_weight(self, judge_id: str) -> float:
        """Get calibration weight for a specific judge based on historical performance."""
        # In production, this would use historical calibration data
        # For now, return a default weight
        return 1.0
    
    def _generate_consensus_reasoning(
        self,
        evaluations: List[JudgeEvaluation],
        consensus_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable consensus reasoning."""
        
        # Find the dimension with the lowest score
        lowest_dim = min(consensus_scores.items(), key=lambda x: x[1])
        highest_dim = max(consensus_scores.items(), key=lambda x: x[1])
        
        reasoning_parts = []
        
        # Overall assessment
        overall_score = sum(consensus_scores.get(dim, 0) * weight for dim, weight in {
            "tool_execution": 0.3, "response_quality": 0.25, "error_handling": 0.2,
            "performance": 0.15, "completeness": 0.1
        }.items())
        
        if overall_score >= 80:
            reasoning_parts.append("Strong performance across most dimensions.")
        elif overall_score >= 60:
            reasoning_parts.append("Adequate performance with some areas for improvement.")
        else:
            reasoning_parts.append("Performance below expectations with significant issues.")
        
        # Highlight key findings
        reasoning_parts.append(f"Strongest area: {highest_dim[0].replace('_', ' ').title()} ({highest_dim[1]:.0f}/100)")
        reasoning_parts.append(f"Needs improvement: {lowest_dim[0].replace('_', ' ').title()} ({lowest_dim[1]:.0f}/100)")
        
        # Judge agreement
        if len(evaluations) > 1:
            score_variance = sum((eval.overall_score - overall_score) ** 2 for eval in evaluations) / len(evaluations)
            if score_variance < 100:
                reasoning_parts.append("High judge agreement on assessment.")
            else:
                reasoning_parts.append("Some disagreement between judges.")
        
        return " ".join(reasoning_parts)
    
    def _create_fallback_evaluation(
        self,
        trajectory: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> EnsembleEvaluation:
        """Create fallback evaluation when judges fail."""
        
        # Basic heuristic scoring based on trajectory status
        status = trajectory.get("status", "unknown")
        base_score = 80.0 if status == "success" else 30.0
        
        fallback_scores = {
            "tool_execution": base_score,
            "response_quality": base_score,
            "error_handling": base_score,
            "performance": base_score,
            "completeness": base_score
        }
        
        overall_score = sum(fallback_scores.get(dim, 0) * weight for dim, weight in {
            "tool_execution": 0.3, "response_quality": 0.25, "error_handling": 0.2,
            "performance": 0.15, "completeness": 0.1
        }.items())
        
        return EnsembleEvaluation(
            dimension_scores=fallback_scores,
            overall_score=overall_score,
            confidence_interval=(overall_score - 10, overall_score + 10),
            consensus_reasoning="Fallback evaluation due to judge system unavailability.",
            individual_evaluations=[],
            calibration_quality=0.5
        )
