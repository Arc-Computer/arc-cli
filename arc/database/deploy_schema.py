#!/usr/bin/env python3
"""
Deploy Arc TimescaleDB Schema
==================================

This script deploys the complete TimescaleDB schema to your cloud instance.
It handles tables, indexes, hypertables, and all TimescaleDB optimizations.

Usage:
    python arc/scripts/deploy_schema.py
    
Environment Variables (.env file):
    TIMESCALE_SERVICE_URL - Full TimescaleDB connection string
    PGPASSWORD          - Database password
    PGUSER              - Database user  
    PGHOST              - Database host
    PGPORT              - Database port
    PGDATABASE          - Database name

Create a .env file in your project root with the required variables.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arc.database.client import ArcDBClient


async def verify_connection(client: ArcDBClient) -> Dict[str, Any]:
    """Verify database connection and TimescaleDB setup."""
    print("🔍 Verifying TimescaleDB connection...")
    
    status = await client.initialize()
    
    if status["status"] == "healthy":
        print("✅ Connection successful!")
        print(f"📊 Extensions: {list(status['extensions'].keys())}")
        print(f"📈 Hypertables: {len(status['hypertables'])}")
        return status
    else:
        print(f"❌ Connection failed: {status['error']}")
        return status


async def deploy_schema(client: ArcDBClient) -> Dict[str, Any]:
    """Deploy the complete TimescaleDB schema."""
    print("\n🚀 Deploying TimescaleDB schema...")
    
    result = await client.deploy_schema()
    
    if result["status"] == "success":
        print("✅ Schema deployed successfully!")
        print("📋 Deployed components:")
        print("   • 12 core tables")
        print("   • 3 hypertables (outcomes, failure_patterns, tool_usage)")
        print("   • Compression policies")
        print("   • Retention policies") 
        print("   • Performance indexes")
        print("   • Continuous aggregates")
        print("   • Vector similarity indexes")
    else:
        print(f"❌ Schema deployment failed: {result['error']}")
    
    return result


async def verify_deployment(client: ArcDBClient) -> Dict[str, Any]:
    """Verify the deployment was successful."""
    print("\n🔎 Verifying deployment...")
    
    try:
        # Check extensions
        extensions = await client.health.check_extensions()
        required_extensions = {"timescaledb", "uuid-ossp", "vector"}
        
        if not required_extensions.issubset(set(extensions.keys())):
            missing = required_extensions - set(extensions.keys())
            print(f"⚠️  Missing extensions: {missing}")
            return {"status": "warning", "missing_extensions": list(missing)}
        
        # Check hypertables
        hypertables = await client.health.check_hypertables()
        expected_hypertables = {"outcomes", "failure_patterns", "tool_usage"}
        actual_hypertables = {ht["hypertable_name"] for ht in hypertables}
        
        if not expected_hypertables.issubset(actual_hypertables):
            missing = expected_hypertables - actual_hypertables
            print(f"⚠️  Missing hypertables: {missing}")
            return {"status": "warning", "missing_hypertables": list(missing)}
        
        print("✅ All components verified!")
        print(f"📊 Extensions: {len(extensions)}")
        print(f"📈 Hypertables: {len(hypertables)}")
        
        return {
            "status": "success",
            "extensions": extensions,
            "hypertables": hypertables
        }
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return {"status": "error", "error": str(e)}





def get_connection_info() -> Dict[str, str]:
    """Get database connection information from environment."""
    # Try TimescaleDB service URL first
    service_url = os.getenv("TIMESCALE_SERVICE_URL")
    if service_url:
        # Ensure we use postgresql+asyncpg:// for SQLAlchemy
        if service_url.startswith("postgres://"):
            service_url = service_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif service_url.startswith("postgresql://") and "+asyncpg" not in service_url:
            service_url = service_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Remove sslmode parameter since SSL is handled in connect_args
        if "sslmode=" in service_url:
            import re
            service_url = re.sub(r'[?&]sslmode=[^&]*', '', service_url)
            # Clean up any trailing ? or & characters
            service_url = service_url.rstrip('?&')
        
        return {"connection_string": service_url}
    
    # Fall back to individual components
    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    
    if not all([host, database, user, password]):
        missing = [var for var, val in {
            "PGHOST": host,
            "PGDATABASE": database, 
            "PGUSER": user,
            "PGPASSWORD": password
        }.items() if not val]
        
        raise ValueError(f"Missing required environment variables: {missing}")
    
    connection_string = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
    
    return {
        "connection_string": connection_string,
        "host": host,
        "port": port,
        "database": database,
        "user": user
    }


async def main():
    """Main deployment function."""
    print("Arc TimescaleDB Schema Deployment")
    print("=" * 50)
    
    try:
        # Get connection info
        conn_info = get_connection_info()
        print(f"🔗 Connecting to: {conn_info.get('host', 'TimescaleDB Cloud')}")
        
        # Initialize client
        client = ArcDBClient(conn_info["connection_string"])
        
        # Step 1: Verify connection
        connection_status = await verify_connection(client)
        if connection_status["status"] != "healthy":
            print("❌ Cannot proceed with unhealthy connection")
            return 1
        
        # Step 2: Deploy schema
        deployment_result = await deploy_schema(client)
        if deployment_result["status"] != "success":
            print("❌ Schema deployment failed")
            return 1
        
        # Step 3: Verify deployment
        verification_result = await verify_deployment(client)
        if verification_result["status"] == "error":
            print("❌ Deployment verification failed")
            return 1
        
        print("\n🎉 TimescaleDB schema deployment completed successfully!")
        print("\n📋 Next Steps:")
        print("   1. Run the test script: python -m arc.tests.integration.test_database")
        print("   2. Check the database health with the ArcDBClient.initialize() method") 
        print("   3. Start using the ArcDBClient in your Modal functions")
        
        await client.close()
        return 0
        
    except Exception as e:
        print(f"💥 Deployment failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 