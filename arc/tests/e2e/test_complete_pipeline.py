#!/usr/bin/env python3
"""
Arc End-to-End Real Agent Testing
Simple test that only calls existing functions - no custom code
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add the arc module to the path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing working functions
from arc.database.api import create_arc_api
from arc.scenarios.generator import ScenarioGenerator
from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer
from arc.cli.commands.run import _execute_with_modal, _check_modal_available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_e2e_pipeline():
    """Test complete pipeline using only existing functions."""
    
    logger.info("🚀 Starting E2E Pipeline Test")
    
    # Setup
    logger.info("📋 Setting up test environment...")
    arc_api = await create_arc_api()
    logger.info("✅ Database API created")
    
    # Check Modal
    modal_available = _check_modal_available()
    if not modal_available:
        raise RuntimeError("Modal not available")
    logger.info("✅ Modal authentication verified")
    
    # Load config - go back 4 levels from arc/tests/e2e/ to get to root
    config_file = Path(__file__).parent.parent.parent.parent / "examples" / "configs" / "minimal_agent.yaml"
    logger.info(f"📄 Loading config: {config_file}")
    
    # Verify config file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    parser = AgentConfigParser()
    normalizer = ConfigNormalizer()
    
    parsed_config = parser.parse(str(config_file))
    normalized_config = normalizer.normalize(parsed_config)
    logger.info(f"✅ Config normalized: {normalized_config['model']}")
    
    # Generate scenarios
    logger.info("📝 Generating scenarios...")
    scenario_generator = ScenarioGenerator(
        agent_config_path=str(config_file),
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        use_patterns=True,
        quality_threshold=0.7
    )
    scenarios = await scenario_generator.generate_scenarios_batch(count=2, pattern_ratio=0.7)
    logger.info(f"✅ Generated {len(scenarios)} scenarios")
    
    # Execute with Modal
    logger.info("🔥 Executing scenarios with Modal...")
    results, execution_time, total_cost = _execute_with_modal(
        scenarios=scenarios,
        agent_config=normalized_config,
        json_output=True
    )
    logger.info(f"✅ Modal execution completed: {len(results)} results")
    logger.info(f"   - Execution time: {execution_time:.2f}s")
    logger.info(f"   - Total cost: ${total_cost:.4f}")
    
    # Test database operations
    logger.info("💾 Testing database operations...")
    
    # Create configuration
    config_id = await arc_api.db.create_configuration(
        name=f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        user_id=str(uuid.uuid4()),
        initial_config=normalized_config
    )
    logger.info(f"✅ Configuration created: {config_id}")
    
    # Start simulation
    simulation_info = await arc_api.start_simulation(
        config_version_id=config_id,
        scenarios=scenarios,  # Function handles format conversion automatically
        simulation_name=f"E2E_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        modal_app_id="e2e_test",
        modal_environment="development",
        sandbox_instances=len(scenarios),
        metadata={"test_type": "e2e_simple"}
    )
    simulation_id = simulation_info["simulation_id"]
    logger.info(f"✅ Simulation started: {simulation_id}")
    
    # Record outcomes
    logger.info("📊 Recording outcomes...")
    for i, (scenario, result) in enumerate(zip(scenarios, results)):
        outcome_id = await arc_api.record_modal_result(
            simulation_id=simulation_id,
            scenario=scenario,
            modal_result=result,
            call_index=i
        )
        logger.info(f"✅ Recorded outcome: {outcome_id}")
    
    # Complete simulation
    completion_result = await arc_api.complete_simulation_from_results(
        simulation_id=simulation_id,
        scenarios=scenarios,
        modal_results=results,
        execution_time=execution_time,
        total_cost=total_cost
    )
    logger.info(f"✅ Simulation completed: {completion_result['status']}")
    
    logger.info("🎉 E2E Pipeline Test PASSED")
    
    return {
        "status": "PASSED",
        "simulation_id": simulation_id,
        "scenarios": len(scenarios),
        "results": len(results),
        "execution_time": execution_time,
        "total_cost": total_cost
    }

async def main():
    """Main entry point."""
    try:
        result = await test_e2e_pipeline()
        
        # Save results
        results_file = f"e2e_simple_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved to: {results_file}")
        logger.info("✅ SUCCESS: All functions work together!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ FAILURE: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 