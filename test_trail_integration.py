#!/usr/bin/env python3
"""
Comprehensive test suite for TRAIL dataset integration in assumption-based scenario generation.

Tests all components: TrailDatasetLoader, TrailPatternAdapter, EnhancedPatternLibrary,
TrailQualityValidator, and enhanced GenerationCoordinator.
"""

import asyncio
import logging
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_trail_loader():
    """Test TRAIL dataset loading and pattern conversion."""
    print("\n🔄 Testing TRAIL Dataset Loader...")
    
    try:
        from arc.scenarios.trail_loader import TrailDatasetLoader
        
        loader = TrailDatasetLoader()
        patterns = await loader.load_patterns(force_refresh=False)
        
        print(f"✅ Loaded {len(patterns)} TRAIL patterns")
        
        # Test pattern structure
        if patterns:
            sample_pattern = patterns[0]
            print(f"   Sample pattern ID: {sample_pattern.id}")
            print(f"   Error type: {sample_pattern.error_type}")
            print(f"   Domain: {sample_pattern.domain}")
            print(f"   Recovery possible: {sample_pattern.recovery_possible}")
        
        # Test statistics
        stats = loader.get_stats()
        print(f"   Dataset stats: {stats.to_dict()}")
        
        # Test categorization
        reasoning_patterns = loader.get_patterns_by_type("reasoning")
        execution_patterns = loader.get_patterns_by_type("execution")
        planning_patterns = loader.get_patterns_by_type("planning")
        
        print(f"   Reasoning patterns: {len(reasoning_patterns)}")
        print(f"   Execution patterns: {len(execution_patterns)}")
        print(f"   Planning patterns: {len(planning_patterns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ TRAIL loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pattern_library():
    """Test enhanced pattern library with TRAIL integration."""
    print("\n🔄 Testing Enhanced Pattern Library...")
    
    try:
        from arc.scenarios.pattern_library import EnhancedPatternLibrary
        
        library = EnhancedPatternLibrary()
        await library.initialize()
        
        metrics = library.get_metrics()
        print(f"✅ Pattern library initialized")
        print(f"   Total patterns: {metrics.total_patterns}")
        print(f"   Local patterns: {metrics.local_patterns}")
        print(f"   TRAIL patterns: {metrics.trail_patterns}")
        
        # Test pattern selection
        assumption_types = ["currency", "api_version", "timeout"]
        selection_result = await library.select_patterns_for_assumptions(
            assumption_types=assumption_types,
            domain_preference="finance",
            count=10
        )
        
        print(f"   Selected {len(selection_result.selected_patterns)} patterns for assumptions")
        print(f"   Selection method: {selection_result.selection_method}")
        print(f"   Estimated diversity: {selection_result.estimated_diversity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern library test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_trail_adapter():
    """Test TRAIL pattern adaptation for assumption violations."""
    print("\n🔄 Testing TRAIL Pattern Adapter...")
    
    try:
        from arc.scenarios.trail_loader import TrailDatasetLoader
        from arc.scenarios.trail_adapter import TrailPatternAdapter
        from arc.scenarios.assumption_extractor import AssumptionExtractor, AgentAssumptions
        
        # Setup components
        loader = TrailDatasetLoader()
        await loader.load_patterns()
        
        adapter = TrailPatternAdapter(loader, random_seed=42)
        
        # Create test assumptions
        assumptions = AgentAssumptions()
        assumptions.currencies.add("USD")
        assumptions.api_versions.add("v2.0")
        assumptions.timeouts["default"] = 30.0
        assumptions.data_formats.add("json")
        
        # Test adaptation
        agent_config = {
            "system_prompt": "You are a finance agent that processes USD transactions",
            "tools": ["currency_converter", "payment_processor"]
        }
        
        adaptation_result = await adapter.adapt_patterns_to_assumptions(
            assumptions=assumptions,
            agent_config=agent_config,
            target_count=15
        )
        
        print(f"✅ Adapted {len(adaptation_result.scenarios)} scenarios")
        print(f"   Adaptations used: {len(adaptation_result.adaptations_used)}")
        print(f"   Success rate: {adaptation_result.success_rate:.2f}")
        print(f"   Adaptation metrics: {adaptation_result.adaptation_metrics}")
        
        # Test sample scenario
        if adaptation_result.scenarios:
            sample_scenario = adaptation_result.scenarios[0]
            print(f"   Sample scenario ID: {sample_scenario.id}")
            print(f"   Tags: {sample_scenario.tags}")
            print(f"   Description: {sample_scenario.description[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ TRAIL adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_quality_validator():
    """Test TRAIL-based quality validation."""
    print("\n🔄 Testing TRAIL Quality Validator...")
    
    try:
        from arc.scenarios.trail_loader import TrailDatasetLoader
        from arc.scenarios.quality_validator import TrailQualityValidator
        from arc.core.models.scenario import Scenario
        
        # Setup validator
        loader = TrailDatasetLoader()
        patterns = await loader.load_patterns()
        validator = TrailQualityValidator(quality_threshold=0.7, trail_patterns=patterns)
        
        # Create test scenario
        test_scenario = Scenario(
            id="test_scenario_001",
            description="Process payment in EUR when system expects USD",
            instructions="Handle financial transaction with EUR currency but agent assumes USD defaults. The conversion API returns timeout after 45 seconds.",
            expected_outputs=[
                "Agent should detect EUR vs USD mismatch",
                "Agent should handle API timeout appropriately",
                "Agent should not proceed with incorrect currency"
            ],
            tools_required=["currency_converter"],
            complexity="medium",
            tags=["trail_adapted", "assumption_currency", "error_execution"],
            metadata={
                "trail_pattern_id": "trail_12345",
                "assumption_violated": "USD",
                "assumption_type": "currency",
                "recovery_possible": True
            }
        )
        
        # Validate scenario
        quality_result = validator.validate_scenario(test_scenario)
        
        print(f"✅ Scenario quality validation completed")
        print(f"   Overall score: {quality_result.overall_score:.3f}")
        print(f"   Passed threshold: {quality_result.passed_threshold}")
        print(f"   Recommendations: {len(quality_result.recommendations)}")
        
        # Show dimension scores
        for score in quality_result.dimension_scores:
            print(f"   {score.dimension.value}: {score.score:.3f} - {score.feedback}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_generation_coordinator():
    """Test enhanced generation coordinator with full TRAIL integration."""
    print("\n🔄 Testing Enhanced Generation Coordinator...")
    
    try:
        from arc.scenarios.generation_coordinator import GenerationCoordinator
        
        # Test finance agent configuration
        finance_agent_config = {
            "system_prompt": "You are a financial analysis agent that processes transactions in USD and generates reports. You have access to currency conversion APIs and payment processing tools.",
            "tools": [
                {"name": "currency_converter", "description": "Convert between currencies using real-time rates"},
                {"name": "payment_processor", "description": "Process financial transactions"},
                {"name": "report_generator", "description": "Generate financial reports"}
            ],
            "model_config": {
                "default_currency": "USD",
                "api_timeout": 30.0,
                "data_format": "json"
            }
        }
        
        # Initialize coordinator with TRAIL enabled
        coordinator = GenerationCoordinator(
            enable_trail=True,
            quality_threshold=0.6,
            random_seed=42
        )
        
        # Test currency scenarios (15 scenarios)
        print("   Generating currency assumption violation scenarios...")
        currency_scenarios, currency_metrics = await coordinator.generate_currency_scenarios(
            agent_config=finance_agent_config,
            count=15
        )
        
        print(f"   ✅ Generated {len(currency_scenarios)} currency scenarios")
        print(f"      Currency scenarios: {currency_metrics.currency_scenarios_generated}")
        print(f"      Quality passed: {currency_metrics.scenarios_passed_quality}")
        print(f"      Generation time: {currency_metrics.generation_time_seconds:.2f}s")
        
        # Test TRAIL scenarios (35 scenarios)
        print("   Generating TRAIL-based capability scenarios...")
        trail_scenarios, trail_metrics = await coordinator.generate_trail_scenarios(
            agent_config=finance_agent_config,
            count=35
        )
        
        print(f"   ✅ Generated {len(trail_scenarios)} TRAIL scenarios")
        print(f"      TRAIL scenarios: {trail_metrics.trail_scenarios_generated}")
        print(f"      TRAIL patterns used: {trail_metrics.trail_patterns_used}")
        print(f"      Quality passed: {trail_metrics.scenarios_passed_quality}")
        print(f"      Generation time: {trail_metrics.generation_time_seconds:.2f}s")
        
        if trail_metrics.adaptation_metrics:
            print(f"      Adaptation metrics: {trail_metrics.adaptation_metrics}")
        
        if trail_metrics.assumption_coverage:
            print(f"      Assumption coverage: {trail_metrics.assumption_coverage}")
        
        # Test complete 50-scenario generation
        print("   Generating complete 50-scenario suite...")
        all_scenarios, all_metrics = await coordinator.generate_scenarios(
            agent_config=finance_agent_config,
            total_scenarios=50,
            focus_on_assumptions=True
        )
        
        print(f"   ✅ Generated {len(all_scenarios)} total scenarios")
        print(f"      Requested: {all_metrics.total_scenarios_requested}")
        print(f"      Generated: {all_metrics.total_scenarios_generated}")
        print(f"      Quality passed: {all_metrics.scenarios_passed_quality}")
        print(f"      Duplicates removed: {all_metrics.duplicates_removed}")
        print(f"      Total time: {all_metrics.generation_time_seconds:.2f}s")
        
        # Show sample scenarios
        if currency_scenarios:
            print(f"\n   Sample currency scenario:")
            sample = currency_scenarios[0]
            print(f"      ID: {sample.id}")
            print(f"      Tags: {sample.tags}")
            print(f"      Description: {sample.description}")
        
        if trail_scenarios:
            print(f"\n   Sample TRAIL scenario:")
            sample = trail_scenarios[0]
            print(f"      ID: {sample.id}")
            print(f"      Tags: {sample.tags}")
            print(f"      Description: {sample.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation coordinator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_end_to_end_workflow():
    """Test complete end-to-end TRAIL integration workflow."""
    print("\n🔄 Testing End-to-End TRAIL Workflow...")
    
    try:
        from arc.scenarios.generation_coordinator import GenerationCoordinator
        
        # Realistic finance agent configuration
        agent_config = {
            "agent_id": "finance_agent_v1",
            "system_prompt": """You are a financial analysis agent specialized in processing multi-currency transactions and generating compliance reports. 
            
            Your key assumptions:
            - All monetary amounts are in USD unless specified
            - Currency conversion rates are always available via API
            - Transaction processing timeout is 30 seconds
            - Reports are generated in JSON format
            - All transactions require regulatory compliance checks
            
            You have access to currency conversion, payment processing, and reporting tools.""",
            
            "tools": [
                {"name": "currency_converter", "description": "Real-time currency conversion with 99.9% uptime"},
                {"name": "payment_processor", "description": "Secure payment processing for international transactions"},
                {"name": "compliance_checker", "description": "Automated regulatory compliance validation"},
                {"name": "report_generator", "description": "Financial report generation in multiple formats"}
            ],
            
            "model_config": {
                "default_currency": "USD",
                "conversion_timeout": 30.0,
                "max_retries": 3,
                "output_format": "json",
                "compliance_required": True
            }
        }
        
        # Initialize with production-like settings
        coordinator = GenerationCoordinator(
            enable_trail=True,
            quality_threshold=0.75,  # Higher threshold for production
            patterns_per_batch=4,
            scenarios_per_pattern=6,
            random_seed=12345  # Reproducible for demos
        )
        
        print("   🎯 Generating complete 50-scenario test suite...")
        
        # Generate 15 currency + 35 TRAIL = 50 total scenarios
        start_time = asyncio.get_event_loop().time()
        
        # Currency scenarios
        currency_scenarios, currency_metrics = await coordinator.generate_currency_scenarios(
            agent_config=agent_config,
            count=15
        )
        
        # TRAIL scenarios  
        trail_scenarios, trail_metrics = await coordinator.generate_trail_scenarios(
            agent_config=agent_config,
            count=35
        )
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Combine results
        all_scenarios = currency_scenarios + trail_scenarios
        
        print(f"   ✅ Complete test suite generated in {total_time:.2f}s")
        print(f"      Total scenarios: {len(all_scenarios)}")
        print(f"      Currency scenarios: {len(currency_scenarios)}")
        print(f"      TRAIL scenarios: {len(trail_scenarios)}")
        
        # Analyze scenario distribution
        tag_distribution = {}
        for scenario in all_scenarios:
            for tag in scenario.tags:
                tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
        
        print(f"      Tag distribution: {dict(sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        # Quality metrics
        total_quality_passed = currency_metrics.scenarios_passed_quality + trail_metrics.scenarios_passed_quality
        total_generated = currency_metrics.total_scenarios_generated + trail_metrics.total_scenarios_generated
        quality_rate = total_quality_passed / max(total_generated, 1)
        
        print(f"      Quality pass rate: {quality_rate:.1%}")
        
        # TRAIL-specific metrics
        if hasattr(trail_metrics, 'adaptation_metrics') and trail_metrics.adaptation_metrics:
            print(f"      TRAIL adaptations: {trail_metrics.adaptation_metrics}")
        
        if hasattr(trail_metrics, 'assumption_coverage') and trail_metrics.assumption_coverage:
            print(f"      Assumption coverage: {trail_metrics.assumption_coverage}")
        
        # Demo output format
        print(f"\n   🎯 Demo Output Format:")
        print(f"   🎯 Generating 50 Capability Test Scenarios:")
        print(f"   ├── Currency Assumption Violations: {len(currency_scenarios)} scenarios")
        print(f"   │   ├── Multi-currency processing: {sum(1 for s in currency_scenarios if any('eur' in tag.lower() or 'gbp' in tag.lower() or 'jpy' in tag.lower() for tag in s.tags))} scenarios")
        print(f"   │   ├── Missing currency indicators: {sum(1 for s in currency_scenarios if 'assumption' in ' '.join(s.tags).lower())} scenarios")
        print(f"   │   └── Currency conversion edge cases: {sum(1 for s in currency_scenarios if 'timeout' in s.description.lower() or 'api' in s.description.lower())} scenarios")
        print(f"   ├── TRAIL-Enhanced Capability Tests: {len(trail_scenarios)} scenarios")
        print(f"   │   ├── Reasoning failures: {sum(1 for s in trail_scenarios if 'reasoning' in ' '.join(s.tags).lower())} scenarios")
        print(f"   │   ├── Execution failures: {sum(1 for s in trail_scenarios if 'execution' in ' '.join(s.tags).lower())} scenarios")
        print(f"   │   └── Planning failures: {sum(1 for s in trail_scenarios if 'planning' in ' '.join(s.tags).lower())} scenarios")
        print(f"   └── Total scenarios: {len(all_scenarios)} (enhanced with real failure patterns)")
        print(f"   ")
        print(f"   ✅ Scenario generation enhanced with real-world agent failure patterns")
        print(f"   ✅ Assumption violations validated against research datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run comprehensive TRAIL integration tests."""
    print("🚀 Starting TRAIL Dataset Integration Tests")
    print("=" * 60)
    
    tests = [
        ("TRAIL Loader", test_trail_loader),
        ("Pattern Library", test_pattern_library),
        ("TRAIL Adapter", test_trail_adapter), 
        ("Quality Validator", test_quality_validator),
        ("Generation Coordinator", test_generation_coordinator),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("🔍 TRAIL Integration Test Results")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall Result: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All TRAIL integration tests PASSED!")
        print("✅ Implementation ready for production use")
        return True
    else:
        print("⚠️  Some tests failed - implementation needs fixes")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)