# RL Integration Plan for Arc-Eval (v2.0 - Predibase-Informed)

## Executive Summary
This document outlines reinforcement learning (RL) enhancements for Arc-Eval based on **validated patterns from Predibase's 2x improvement with 10 training examples** and analysis of our existing infrastructure. We prioritize proven, low-risk approaches that leverage our Configuration→Outcome data asset.

## When to Implement This
✅ **READY NOW** - Your infrastructure supports immediate RL implementation  
✅ Configuration→Outcome graph with PostgreSQL JSONB storage  
✅ Multi-model testing across 9+ providers (competitive advantage)  
✅ Rich trajectory capture with OpenTelemetry  
✅ Modal parallel execution for training scalability  
✅ Failure clustering and recommendation generation  

## Revised Implementation Roadmap (Predibase-Informed)

### Phase 1: Simple RFT for Config Optimization (Week 1-2) 
**What**: Fine-tune recommendation generation using your existing Configuration→Outcome data  
**Why**: Predibase showed 2x improvement with 10 examples - you have 1K+ trajectories ready  
**Model**: Start with 7B-13B models (sufficient for config optimization vs code generation)  

**Simple Reward Structure** (Based on Predibase Success):
```python
# SIMPLE BINARY REWARDS (Proven Stable)
{
    "diff_accepted": 1.0,           # User applied recommendation
    "diff_rejected": -1.0,          # User ignored recommendation  
    "reliability_improved": 0.5,    # Applied diff improved score
    "format_valid": 0.1             # Recommendation properly formatted
}

# AVOID COMPLEX MULTI-REWARDS (Predibase showed instability)
# No tool-level rewards, no process rewards initially
```

**Data Preparation** (You Already Have This):
```python
# Your PostgreSQL outcomes table contains perfect training data
SELECT 
    o.config_hash,
    o.trajectory->'failure_patterns' as failure_context,
    r.recommendation_type,
    r.description as recommendation,
    -- Simulate diff_acceptance from recommendation confidence
    CASE WHEN r.confidence > 0.8 THEN 1.0 ELSE -1.0 END as reward
FROM outcomes o
JOIN recommendations r ON o.outcome_id = r.outcome_id
WHERE o.reliability_score < 0.8  -- Focus on failed scenarios
```

**Expected Results**: 2x improvement in recommendation acceptance rate (following Predibase pattern)

### Phase 2: Multi-Model Contextual Bandits (Week 3-4)
**What**: Use RFT model within bandit framework for model-neutral recommendations  
**Why**: Leverages your unique cross-model data advantage  

**Enhanced State Features**:
```python
{
    "failure_cluster_embedding": vector,     # From your clustering system
    "current_config": {
        "model": "gpt-4.1",
        "temperature": 0.3,
        "tools": ["web_search", "calculator"]
    },
    "historical_patterns": {
        "model_switch_acceptance": 0.75,     # Cross-model recommendation history
        "user_role": "senior_ml_engineer",   # From user context tracking
        "domain": "fintech"                  # Inferred from scenarios
    },
    "cost_performance_ratio": 0.85          # Current config efficiency
}
```

**Recommendation Types**:
```python
# Your existing recommendation_generator.py already supports these
{
    "cheapest": "Switch to claude-3-haiku (98.7% lower cost, 96% reliability)",
    "highest_perf": "Upgrade to gpt-4.1 + retry logic (20% reliability gain)", 
    "balanced": "Use gpt-4.1 + temperature=0.3 (optimal cost/performance)",
    "model_switch": "Switch to gemini-1.5-pro for this failure pattern (+15% success)"
}
```

### Phase 3: Online RL with Process Rewards (Month 2)
**What**: GRPO with simplified process rewards  
**Why**: Only after proving value with simpler approaches  

**Simplified Process Rewards** (Learned from Predibase):
```python
# START SIMPLE, EXPAND GRADUALLY
{
    "tool_timeout": -0.1,              # Clear negative signal
    "successful_tool_call": 0.05,      # Small positive reinforcement
    "config_applied": 1.0,             # Main reward signal
    "reliability_delta": actual_improvement  # Real outcome measurement
}

# AVOID INITIALLY (Caused Predibase instability):
# - Complex multi-tool scoring
# - Linting/type-checking rewards  
# - Too many simultaneous metrics
```

## Critical Infrastructure Advantages (Already Built)

### 1. **PostgreSQL Performance Foundation**
Your database performance analysis shows:
- **34% higher throughput** than SQLite under concurrent load
- **12.5x faster write performance** for RL training data storage
- **Real-time analytics** during training (SQLite failed completely)
- **JSONB indexing** for fast trajectory queries

### 2. **Modal Parallel Training Infrastructure**
Your existing `parallel_evaluator.py` enables:
- **50 concurrent containers** for RL training environments
- **@modal.concurrent** decorators for LLM calls
- **Autoscaling** (buffer_containers=5, scaledown_window=60)
- **Cost-efficient** batch processing

### 3. **Multi-Model Data Advantage**
Your `multi_model_tester.py` provides unique competitive moat:
- **9+ model providers** tested per scenario
- **Model-neutral optimization** (no vendor lock-in)
- **Cross-model recommendation validation**
- **Cost-performance optimization** across all providers

### 4. **Rich Trajectory Data**
Your `trajectory_capture.py` provides perfect RL training signals:
```python
# Already captured in your TrajectoryData class
{
    "tool_calls": [{"success": bool, "duration_ms": float}],
    "llm_interactions": [{"tokens_in": int, "tokens_out": int}],
    "error_events": [{"error_type": str, "recovery_successful": bool}],
    "decision_points": [{"selected": str, "rationale": str}]
}
```

## Implementation Timeline (Accelerated)

### Week 1: Data Preparation & Simple RFT
**Leverage Existing Infrastructure**:
```bash
# Your data is already formatted perfectly
python src/tracing/trajectory_storage.py --export-rl-training-data
python src/analysis/failure_clustering.py --generate-embeddings
python src/orchestration/run_full_loop.py --prepare-rl-dataset
```

**RFT Training Setup**:
- Use existing Modal infrastructure for GPU allocation
- Start with Qwen2.5-7B (sufficient for config optimization)
- Simple binary reward: diff_accepted (1.0) vs diff_rejected (-1.0)

### Week 2: RFT Training & Validation
**Training Pipeline**:
```python
# Leverage your existing parallel evaluation
@app.function(gpu="A100", timeout=3600)
def train_rft_model(training_data, base_model="Qwen/Qwen2.5-7B"):
    # Use Predibase's proven simple approach
    # Focus on format + API accuracy graders only
    return trained_model

# Validate using your existing evaluation suite
python src/core/arc_eval_sandbox.py --use-rft-recommendations
```

### Week 3-4: Bandit Integration
**Enhanced Recommendation Engine**:
```python
# Extend your existing recommendation_generator.py
class RFTEnhancedRecommendationGenerator:
    def __init__(self):
        self.rft_model = load_trained_model()
        self.bandit_optimizer = ThompsonSampling()
        
    def generate_recommendations(self, failure_cluster, user_context):
        # Use RFT model for recommendation generation
        base_recs = self.rft_model.generate(failure_cluster)
        
        # Use bandit for recommendation selection/ranking
        selected = self.bandit_optimizer.select(base_recs, user_context)
        
        return selected
```

### Month 2: Online Learning (If Validated)
Only proceed if Phase 1-2 show clear value.

## Success Metrics (Predibase-Aligned)

### Phase 1 Targets
- **2x improvement** in recommendation acceptance rate
- **Stable training** with simple rewards (no instability)
- **Format validity**: 95%+ properly formatted recommendations
- **Model neutrality**: Balanced recommendations across providers

### Phase 2 Targets  
- **15% improvement** in cross-model recommendation accuracy
- **Sub-100ms** recommendation latency with bandit selection
- **User segmentation**: Different strategies for different user types

### Phase 3 Targets
- **20% reliability improvement** from applied configurations
- **Positive ROI** within 4 weeks of online deployment
- **No reward hacking** behaviors

## Risk Mitigation (Predibase Lessons)

### Avoid Complex Rewards Initially
❌ **Don't**: Multi-tool scoring, linting checks, complex process rewards  
✅ **Do**: Simple binary diff_acceptance + format validation  

### Start with Proven Model Sizes
❌ **Don't**: Models below 7B (insufficient capability)  
✅ **Do**: 7B-13B for config optimization (not code generation)  

### Leverage Existing Infrastructure
❌ **Don't**: Build new training infrastructure  
✅ **Do**: Use Modal, PostgreSQL, and existing evaluation pipeline  

### Validate Incrementally
❌ **Don't**: Jump to complex online RL  
✅ **Do**: Prove value with offline methods first  

## Competitive Advantage Summary

Your Arc-Eval platform is uniquely positioned for RL success because:

1. **Multi-Model Data Moat**: Only platform with cross-provider optimization data
2. **Production-Ready Infrastructure**: PostgreSQL + Modal + OpenTelemetry stack
3. **Rich Trajectory Capture**: Perfect training signals already collected
4. **Model-Neutral Optimization**: Defensible value no single provider can match
5. **Proven Evaluation Pipeline**: 91.6% reliability baseline with 40s execution time

## Next Steps (This Week)

### Immediate Actions
1. **Export RL Training Data**: Use existing PostgreSQL queries to format training examples
2. **Simple RFT Experiment**: 10-50 examples following Predibase pattern
3. **Baseline Measurement**: Current recommendation acceptance rate in human review
4. **Infrastructure Prep**: Allocate Modal GPU resources for training

### Success Criteria for Continuation
- **Training Stability**: No divergence with simple rewards
- **Quality Improvement**: Measurable increase in recommendation acceptance
- **Infrastructure Validation**: Smooth integration with existing systems

---

**Remember**: The Predibase case study validates that **simple approaches with minimal data can achieve breakthrough results**. Your existing infrastructure and data quality position you for immediate success using proven patterns.

**Key Insight**: You're not building RL from scratch - you're enhancing an already sophisticated system with targeted RL improvements where they add the most value.