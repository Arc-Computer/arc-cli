#!/usr/bin/env python3
"""
Deep Database Data Inspection
Show top 5 records from each table with full data structure
"""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the arc module to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from arc.database.api import create_arc_api
from sqlalchemy import text

def print_table_header(table_name):
    """Print a formatted table header."""
    print(f"\n{'='*80}")
    print(f"📋 TABLE: {table_name.upper()}")
    print(f"{'='*80}")

def print_record(record_num, data):
    """Print a single record in a readable format."""
    print(f"\n  📝 RECORD {record_num}:")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"    {key}: (JSON) {json.dumps(value, indent=6)}")
        elif isinstance(value, str) and len(str(value)) > 100:
            print(f"    {key}: {str(value)[:100]}...")
        else:
            print(f"    {key}: {value}")

async def inspect_all_tables():
    """Inspect all tables with top 5 records each."""
    
    print("🔍 DEEP DATABASE INSPECTION - TOP 5 RECORDS PER TABLE")
    print("=" * 80)
    
    arc_api = await create_arc_api()
    
    async with arc_api.db.engine.begin() as conn:
        
        # 1. CONFIGURATIONS TABLE
        print_table_header("configurations")
        result = await conn.execute(text("""
            SELECT * FROM configurations 
            ORDER BY created_at DESC 
            LIMIT 5
        """))
        
        configs = []
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            configs.append(record)
            print_record(i, record)
        
        # 2. CONFIG_VERSIONS TABLE  
        print_table_header("config_versions")
        result = await conn.execute(text("""
            SELECT * FROM config_versions 
            ORDER BY created_at DESC 
            LIMIT 5
        """))
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            # Parse JSON fields for better display
            if record.get('parsed_config'):
                try:
                    record['parsed_config'] = json.loads(record['parsed_config'])
                except:
                    pass
            print_record(i, record)
        
        # 3. SIMULATIONS TABLE
        print_table_header("simulations")
        result = await conn.execute(text("""
            SELECT * FROM simulations 
            ORDER BY started_at DESC 
            LIMIT 5
        """))
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            # Parse JSON fields
            if record.get('metadata'):
                try:
                    if isinstance(record['metadata'], str):
                        record['metadata'] = json.loads(record['metadata'])
                except:
                    pass
            print_record(i, record)
        
        # 4. SCENARIOS TABLE
        print_table_header("scenarios")
        result = await conn.execute(text("""
            SELECT * FROM scenarios 
            ORDER BY created_at DESC 
            LIMIT 5
        """))
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            print_record(i, record)
        
        # 5. OUTCOMES TABLE (most important)
        print_table_header("outcomes")
        result = await conn.execute(text("""
            SELECT * FROM outcomes 
            ORDER BY execution_time DESC 
            LIMIT 5
        """))
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            # Parse JSON fields
            if record.get('trajectory'):
                if isinstance(record['trajectory'], str):
                    try:
                        record['trajectory'] = json.loads(record['trajectory'])
                    except:
                        pass
            if record.get('metrics'):
                if isinstance(record['metrics'], str):
                    try:
                        record['metrics'] = json.loads(record['metrics'])
                    except:
                        pass
            print_record(i, record)
        
        # 6. SIMULATIONS_SCENARIOS TABLE (junction table)
        print_table_header("simulations_scenarios")
        result = await conn.execute(text("""
            SELECT * FROM simulations_scenarios 
            ORDER BY started_at DESC NULLS LAST
            LIMIT 5
        """))
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            print_record(i, record)
        
        # 7. SCHEMA OVERVIEW
        print_table_header("complete schema overview")
        
        # Show all tables in the schema
        print("  🗂️  ALL TABLES IN SCHEMA:")
        result = await conn.execute(text("""
            SELECT 
                table_name,
                table_type,
                CASE 
                    WHEN table_name IN (
                        SELECT hypertable_name 
                        FROM timescaledb_information.hypertables 
                        WHERE hypertable_schema = 'public'
                    ) THEN 'HYPERTABLE'
                    ELSE 'REGULAR'
                END as table_category
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_category DESC, table_name
        """))
        
        regular_count = 0
        hypertable_count = 0
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            if record['table_category'] == 'HYPERTABLE':
                hypertable_count += 1
                print(f"    {i:2d}. 🕰️  {record['table_name']} ({record['table_category']})")
            else:
                regular_count += 1
                print(f"    {i:2d}. 📋 {record['table_name']} ({record['table_category']})")
        
        total_tables = regular_count + hypertable_count
        print(f"\n  📊 SCHEMA SUMMARY:")
        print(f"    Total tables: {total_tables}")
        print(f"    Regular tables: {regular_count}")
        print(f"    Hypertables: {hypertable_count}")
        
        # 8. HYPERTABLES INSPECTION
        print_table_header("timescaledb hypertables details")
        
        # Show hypertable information with basic compatible columns
        result = await conn.execute(text("""
            SELECT 
                hypertable_name,
                hypertable_schema,
                num_dimensions,
                num_chunks
            FROM timescaledb_information.hypertables
        """))
        
        print("  🕰️  HYPERTABLE DETAILS:")
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            print_record(i, record)
        
        # Show chunk information for each hypertable
        print(f"\n  📦 CHUNKS (partitions) for all hypertables:")
        result = await conn.execute(text("""
            SELECT 
                hypertable_name,
                chunk_name,
                range_start,
                range_end,
                chunk_schema
            FROM timescaledb_information.chunks
            ORDER BY hypertable_name, range_start DESC
        """))
        
        current_hypertable = None
        chunk_count = 0
        
        for row in result:
            record = dict(zip(result.keys(), row))
            if record['hypertable_name'] != current_hypertable:
                if current_hypertable is not None:
                    print(f"      Total chunks for {current_hypertable}: {chunk_count}")
                current_hypertable = record['hypertable_name']
                chunk_count = 0
                print(f"\n    {current_hypertable} chunks:")
            
            chunk_count += 1
            if chunk_count <= 3:  # Show first 3 chunks per hypertable
                print(f"      {chunk_count}. {record['chunk_name']} ({record['range_start']} to {record['range_end']})")
        
        if current_hypertable is not None:
            print(f"      Total chunks for {current_hypertable}: {chunk_count}")
        
        # Show time-series data distribution for outcomes
        print(f"\n  📈 TIME-SERIES DATA DISTRIBUTION (outcomes):")
        result = await conn.execute(text("""
            SELECT 
                DATE_TRUNC('hour', execution_time) as hour,
                COUNT(*) as outcomes_count,
                MIN(execution_time) as first_execution,
                MAX(execution_time) as last_execution
            FROM outcomes
            GROUP BY hour
            ORDER BY hour DESC
            LIMIT 10
        """))
        
        for i, row in enumerate(result, 1):
            record = dict(zip(result.keys(), row))
            print_record(i, record)
        
        # 9. SUMMARY STATISTICS
        print_table_header("summary statistics")
        
        tables = [
            "configurations", "config_versions", "simulations", 
            "scenarios", "outcomes", "simulations_scenarios"
        ]
        
        for table in tables:
            result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table}: {count} records")
        
        # 10. OUTCOME STATUS BREAKDOWN
        print(f"\n  📊 OUTCOME STATUS BREAKDOWN:")
        result = await conn.execute(text("""
            SELECT status, COUNT(*) as count, AVG(reliability_score) as avg_score
            FROM outcomes 
            GROUP BY status
        """))
        
        for row in result:
            print(f"    {row[0]}: {row[1]} outcomes (avg score: {row[2]:.3f})")

async def main():
    """Main entry point."""
    try:
        await inspect_all_tables()
        print(f"\n✅ Deep database inspection completed!")
        return 0
    except Exception as e:
        print(f"\n❌ Database inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 