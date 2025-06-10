#!/usr/bin/env python3
"""
Test Arc-Eval TimescaleDB Integration
=====================================

Comprehensive test suite for TimescaleDB deployment including:
- Connection and health checks
- CRUD operations on all tables  
- Hypertable functionality
- Time-series queries
- Modal integration patterns
- Performance benchmarks

Usage:
    python arc/scripts/test_db.py
"""

import asyncio
import os
import sys
import time
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy import text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from arc.database.client import ArcDBClient


class TimescaleDBTester:
    """Comprehensive tester for TimescaleDB functionality."""
    
    def __init__(self, client: ArcDBClient):
        self.client = client
        self.test_data = {}
        
    async def test_connection_health(self) -> Dict[str, Any]:
        """Test 1: Connection and Health Checks"""
        print("🧪 Test 1: Connection and Health Checks")
        
        try:
            status = await self.client.initialize()
            
            if status["status"] != "healthy":
                return {"status": "failed", "error": "Unhealthy connection"}
            
            # Check required extensions
            extensions = status["extensions"]
            required = {"timescaledb", "uuid-ossp", "vector"}
            
            if not required.issubset(set(extensions.keys())):
                missing = required - set(extensions.keys())
                return {"status": "failed", "error": f"Missing extensions: {missing}"}
            
            # Check hypertables
            hypertables = status["hypertables"]
            expected_hypertables = {"outcomes", "failure_patterns", "tool_usage"}
            actual_hypertables = {ht["hypertable_name"] for ht in hypertables}
            
            if not expected_hypertables.issubset(actual_hypertables):
                missing = expected_hypertables - actual_hypertables
                return {"status": "failed", "error": f"Missing hypertables: {missing}"}
            
            print("✅ Connection healthy, all extensions and hypertables present")
            return {"status": "passed", "extensions": len(extensions), "hypertables": len(hypertables)}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_configuration_management(self) -> Dict[str, Any]:
        """Test 2: Configuration and Version Management"""
        print("\n🧪 Test 2: Configuration and Version Management")
        
        try:
            # Create test configuration with proper UUID
            test_user_id = str(uuid.uuid4())
            config_id = await self.client.create_configuration(
                name="test_gpt4_config",
                user_id=test_user_id,
                initial_config={
                    "model": "gpt-4",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "tools": ["web_search", "code_execution"]
                }
            )
            
            # Get the version_id for this configuration
            async with self.client.engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT version_id FROM config_versions 
                    WHERE config_id = :config_id 
                    ORDER BY version_number DESC LIMIT 1
                """), {"config_id": config_id})
                version_row = result.fetchone()
                # Handle UUID object properly
                version_id = str(version_row[0]) if version_row else None
            
            if not version_id:
                return {"status": "failed", "error": "Could not retrieve version_id"}
            
            self.test_data["config_id"] = config_id
            self.test_data["version_id"] = version_id
            self.test_data["user_id"] = test_user_id
            print(f"✅ Created configuration: {config_id[:8]}... (version: {version_id[:8]}...)")
            
            return {"status": "passed", "config_id": config_id, "version_id": version_id}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_scenario_simulation(self) -> Dict[str, Any]:
        """Test 3: Simulation and Scenario Management"""
        print("\n🧪 Test 3: Simulation and Scenario Management")
        
        try:
            if "version_id" not in self.test_data:
                return {"status": "skipped", "reason": "No version_id from previous test"}
            
            # First create scenarios that we'll reference using correct schema
            # Use UPSERT to handle existing scenarios gracefully
            scenario_names = ["math_reasoning", "code_debugging", "text_analysis"]
            scenario_ids = []
            
            async with self.client.engine.begin() as conn:
                for i, scenario_name in enumerate(scenario_names):
                    # Use TEXT scenario_id as per schema, not UUID
                    scenario_id = f"test_{scenario_name}_{i}"
                    await conn.execute(text("""
                        INSERT INTO scenarios (scenario_id, name, task_prompt, difficulty_level)
                        VALUES (:scenario_id, :name, :task_prompt, :difficulty_level)
                        ON CONFLICT (scenario_id) DO NOTHING
                    """), {
                        "scenario_id": scenario_id,
                        "name": scenario_name,
                        "task_prompt": f"Test task prompt for {scenario_name} scenario",
                        "difficulty_level": "medium"
                    })
                    scenario_ids.append(scenario_id)
            
            # Create test simulation using version_id
            simulation_id = await self.client.create_simulation(
                config_version_id=self.test_data["version_id"],
                scenario_set=scenario_names,
                simulation_name="comprehensive_test_run",
                modal_app_id="arc-eval-test"
            )
            
            self.test_data["simulation_id"] = simulation_id
            self.test_data["scenario_ids"] = scenario_ids
            print(f"✅ Created simulation: {simulation_id[:8]}... with {len(scenario_ids)} scenarios")
            
            return {"status": "passed", "simulation_id": simulation_id, "scenarios_created": len(scenario_ids)}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_outcome_recording(self) -> Dict[str, Any]:
        """Test 4: Outcome Recording to Hypertable"""
        print("\n🧪 Test 4: Outcome Recording (TimescaleDB Hypertable)")
        
        try:
            if "simulation_id" not in self.test_data or "scenario_ids" not in self.test_data:
                return {"status": "skipped", "reason": "No simulation_id or scenario_ids from previous test"}
            
            # Record multiple outcomes with different timestamps
            outcome_ids = []
            scenario_ids = self.test_data["scenario_ids"]
            
            for i in range(5):
                # Vary the execution time to test time-series functionality
                execution_time = datetime.now(timezone.utc) - timedelta(minutes=i*10)
                # Use actual scenario IDs from the database
                scenario_id = scenario_ids[i % len(scenario_ids)]
                
                outcome_id = await self.client.record_outcome({
                    "simulation_id": self.test_data["simulation_id"],
                    "scenario_id": scenario_id,
                    "execution_time": execution_time,
                    "status": "success" if random.random() > 0.3 else "error",
                    "reliability_score": random.uniform(0.7, 1.0),
                    "execution_time_ms": random.randint(1000, 5000),
                    "tokens_used": random.randint(100, 500),
                    "cost_usd": random.uniform(0.001, 0.05),
                    "trajectory": {
                        "start_time": execution_time.isoformat(),
                        "status": "completed",
                        "steps": [f"step_{j}" for j in range(random.randint(3, 8))],
                        "result": f"test_result_{i}"
                    },
                    "modal_call_id": f"modal-test-{i}-{int(time.time())}",
                    "sandbox_id": f"sandbox-{i}",
                    "metrics": {
                        "accuracy": random.uniform(0.8, 1.0),
                        "latency": random.uniform(0.5, 2.0)
                    }
                })
                
                outcome_ids.append(outcome_id)
            
            self.test_data["outcome_ids"] = outcome_ids
            print(f"✅ Recorded {len(outcome_ids)} outcomes to hypertable")
            
            return {"status": "passed", "outcomes_count": len(outcome_ids)}
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_batch_operations(self) -> Dict[str, Any]:
        """Test 5: Batch Operations for High-Throughput"""
        print("\n🧪 Test 5: Batch Operations (High-Throughput Modal)")
        
        try:
            if "simulation_id" not in self.test_data or "scenario_ids" not in self.test_data:
                return {"status": "skipped", "reason": "No simulation_id or scenario_ids from previous test"}
            
            # Create batch of outcomes
            batch_outcomes = []
            batch_size = 20
            scenario_ids = self.test_data["scenario_ids"]
            
            for i in range(batch_size):
                batch_outcomes.append({
                    "simulation_id": self.test_data["simulation_id"],
                    "scenario_id": scenario_ids[i % len(scenario_ids)],  # Cycle through actual scenario IDs
                    "execution_time": datetime.now(timezone.utc) - timedelta(seconds=i*30),
                    "status": random.choice(["success", "error", "timeout"]),
                    "reliability_score": random.uniform(0.6, 1.0),
                    "execution_time_ms": random.randint(500, 3000),
                    "tokens_used": random.randint(50, 300),
                    "cost_usd": random.uniform(0.001, 0.02),
                    "trajectory": {
                        "start_time": datetime.now(timezone.utc).isoformat(),
                        "status": "completed",
                        "batch_id": i
                    },
                    "modal_call_id": f"batch-modal-{i}",
                    "error_category": random.choice(["timeout", "model_error", "system_error"]) if random.random() > 0.7 else None
                })
            
            # Time the batch operation
            start_time = time.time()
            batch_outcome_ids = await self.client.record_outcomes_batch(batch_outcomes)
            batch_time = time.time() - start_time
            
            throughput = len(batch_outcomes) / batch_time
            
            print(f"✅ Batch inserted {len(batch_outcome_ids)} outcomes in {batch_time:.2f}s")
            print(f"📊 Throughput: {throughput:.1f} outcomes/second")
            
            return {
                "status": "passed", 
                "batch_size": len(batch_outcome_ids),
                "batch_time_seconds": batch_time,
                "throughput_per_second": throughput
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_time_series_queries(self) -> Dict[str, Any]:
        """Test 6: Time-Series Analysis (TimescaleDB Features)"""
        print("\n🧪 Test 6: Time-Series Analysis (TimescaleDB Features)")
        
        try:
            if "simulation_id" not in self.test_data:
                return {"status": "skipped", "reason": "No simulation_id from previous test"}
            
            # Test simulation performance query
            performance = await self.client.get_simulation_performance(
                simulation_id=self.test_data["simulation_id"],
                time_range=timedelta(hours=24)
            )
            
            print(f"✅ Retrieved performance data for {performance['time_range_hours']} hours")
            print(f"📊 Total outcomes: {performance['summary']['total_outcomes']}")
            print(f"📈 Average reliability: {performance['summary']['avg_reliability']:.3f}")
            
            # Test recent failures query
            failures = await self.client.get_recent_failures(limit=10)
            
            print(f"🔍 Found {len(failures)} recent failures")
            
            return {
                "status": "passed",
                "performance_data_points": len(performance["hourly_metrics"]),
                "total_outcomes": performance["summary"]["total_outcomes"],
                "recent_failures": len(failures)
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_compression_and_chunks(self) -> Dict[str, Any]:
        """Test 7: TimescaleDB Compression and Chunks"""
        print("\n🧪 Test 7: TimescaleDB Compression and Chunk Analysis")
        
        try:
            # Use a simpler query that works with TimescaleDB Cloud
            async with self.client.engine.begin() as conn:
                # Just check hypertables and basic chunk info
                result = await conn.execute(text("""
                    SELECT 
                        hypertable_name,
                        num_chunks as total_chunks,
                        compression_enabled
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_name IN ('outcomes', 'failure_patterns', 'tool_usage')
                """))
                
                compression_stats = [dict(row._mapping) for row in result]
            
            print(f"📦 Hypertable stats for {len(compression_stats)} tables")
            
            for stat in compression_stats:
                print(f"📊 {stat['hypertable_name']}: {stat['total_chunks']} chunks, compression: {stat['compression_enabled']}")
            
            return {
                "status": "passed",
                "compression_stats": compression_stats
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def test_modal_integration_patterns(self) -> Dict[str, Any]:
        """Test 8: Modal Integration Patterns"""
        print("\n🧪 Test 8: Modal Integration Patterns")
        
        try:
            # Use simulation_id from previous tests if available, otherwise create a new one
            if "simulation_id" in self.test_data and "scenario_ids" in self.test_data:
                simulation_id = self.test_data["simulation_id"]
                scenario_ids = self.test_data["scenario_ids"]
            else:
                # Create a standalone test simulation and scenarios
                # First create a test configuration if we don't have one
                if "version_id" not in self.test_data:
                    test_user_id = str(uuid.uuid4())
                    config_id = await self.client.create_configuration(
                        name="modal_test_config",
                        user_id=test_user_id,
                        initial_config={
                            "model": "gpt-4",
                            "temperature": 0.3,
                            "max_tokens": 1000,
                            "tools": ["modal_test"]
                        }
                    )
                    
                    # Get version_id
                    async with self.client.engine.begin() as conn:
                        result = await conn.execute(text("""
                            SELECT version_id FROM config_versions 
                            WHERE config_id = :config_id 
                            ORDER BY version_number DESC LIMIT 1
                        """), {"config_id": config_id})
                        version_row = result.fetchone()
                        version_id = str(version_row[0]) if version_row else None
                    
                    if not version_id:
                        return {"status": "failed", "error": "Could not create test configuration"}
                    
                    self.test_data["version_id"] = version_id
                
                # Create test scenarios for Modal integration
                scenario_ids = []
                async with self.client.engine.begin() as conn:
                    for i in range(3):
                        # Use TEXT scenario_id as per schema
                        scenario_id = f"modal_test_scenario_{i}"
                        await conn.execute(text("""
                            INSERT INTO scenarios (scenario_id, name, task_prompt, difficulty_level)
                            VALUES (:scenario_id, :name, :task_prompt, :difficulty_level)
                            ON CONFLICT (scenario_id) DO NOTHING
                        """), {
                            "scenario_id": scenario_id,
                            "name": f"Modal Test Scenario {i}",
                            "task_prompt": f"Modal integration test task prompt for scenario {i}",
                            "difficulty_level": "medium"
                        })
                        scenario_ids.append(scenario_id)
                
                # Create simulation
                simulation_id = await self.client.create_simulation(
                    config_version_id=self.test_data["version_id"],
                    scenario_set=[f"Modal Test Scenario {i}" for i in range(3)],
                    simulation_name="modal_integration_test",
                    modal_app_id="modal-test-app"
                )
                
                self.test_data["simulation_id"] = simulation_id
                self.test_data["scenario_ids"] = scenario_ids
            
            # Simulate Modal execution patterns
            modal_outcomes = []
            
            # Simulate concurrent Modal executions
            for modal_instance in range(3):
                for scenario_batch in range(5):
                    modal_outcomes.append({
                        "simulation_id": simulation_id,
                        "scenario_id": scenario_ids[modal_instance % len(scenario_ids)],  # Use actual scenario IDs
                        "status": "success",
                        "reliability_score": 0.9,
                        "execution_time_ms": 1500,
                        "tokens_used": 200,
                        "cost_usd": 0.01,
                        "trajectory": {
                            "start_time": datetime.now(timezone.utc).isoformat(),
                            "status": "completed",
                            "modal_instance": modal_instance,
                            "batch": scenario_batch
                        },
                        "modal_call_id": f"modal-{modal_instance}-{scenario_batch}",
                        "sandbox_id": f"sandbox-{modal_instance}"
                    })
            
            # Batch insert all Modal outcomes
            modal_outcome_ids = await self.client.record_outcomes_batch(modal_outcomes)
            
            print(f"✅ Simulated {len(modal_outcome_ids)} Modal executions")
            print(f"🔗 Modal call IDs: {len([o for o in modal_outcomes if o['modal_call_id']])}")
            
            return {
                "status": "passed",
                "modal_executions": len(modal_outcome_ids),
                "modal_instances": 3,
                "scenarios_per_instance": 5
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        tests = [
            ("Connection Health", self.test_connection_health),
            ("Configuration Management", self.test_configuration_management),
            ("Scenario Simulation", self.test_scenario_simulation),
            ("Outcome Recording", self.test_outcome_recording),
            ("Batch Operations", self.test_batch_operations),
            ("Time-Series Queries", self.test_time_series_queries),
            ("Compression & Chunks", self.test_compression_and_chunks),
            ("Modal Integration", self.test_modal_integration_patterns)
        ]
        
        results = {}
        passed = 0
        failed = 0
        skipped = 0
        
        print("🚀 Running TimescaleDB Test Suite")
        print("=" * 60)
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
                
                if result["status"] == "passed":
                    passed += 1
                elif result["status"] == "failed":
                    failed += 1
                    print(f"❌ {test_name} failed: {result.get('error', 'Unknown error')}")
                elif result["status"] == "skipped":
                    skipped += 1
                    print(f"⏭️  {test_name} skipped: {result.get('reason', 'Unknown reason')}")
                    
            except Exception as e:
                results[test_name] = {"status": "failed", "error": str(e)}
                failed += 1
                print(f"💥 {test_name} crashed: {str(e)}")
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%" if (passed+failed) > 0 else "N/A")
        
        if failed == 0:
            print("\n🎉 All tests passed! TimescaleDB integration is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check the errors above.")
        
        return {
            "summary": {"passed": passed, "failed": failed, "skipped": skipped},
            "details": results
        }


async def main():
    """Main test runner."""
    try:
        # Initialize client
        client = ArcDBClient()
        tester = TimescaleDBTester(client)
        
        # Run all tests
        results = await tester.run_all_tests()
        
        # Close client
        await client.close()
        
        # Return exit code based on results
        return 0 if results["summary"]["failed"] == 0 else 1
        
    except Exception as e:
        print(f"💥 Test suite failed to initialize: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 