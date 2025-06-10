# Example Agent Configurations

This directory contains example agent configurations for Arc-CLI. These realistic finance agents demonstrate how Arc can identify and fix potential issues before production deployment.

**Note**: The `arc/config/` directory contains Arc's internal configuration files (development.py, production.py, etc.), while this `examples/configs/` directory contains example agent configurations for users to reference.

## Overview

The configurations represent a real-world progression from a basic implementation to a more robust system. They serve as both examples for users and test cases for Arc's evaluation pipeline.

## Configurations

### 1. finance_agent_v1.yaml - Basic Finance Agent

**Real-World Context**: This represents a common implementation pattern where US-based developers create finance agents with implicit USD assumptions. Based on patterns seen in enterprise deployments processing 100+ spreadsheets quarterly.

**Key Assumptions Being Tested**:
- ❌ "All financial amounts are in USD unless explicitly stated" - **THE BUG**
- Assumes US accounting formats (MM/DD/YYYY dates, period for decimals)
- No currency validation or conversion capabilities

**Failure Scenarios**:
- European subsidiary reports (EUR amounts interpreted as USD)
- Japanese financial data (¥1,000,000 processed as $1,000,000)
- Multi-currency consolidations (all currencies treated as USD)
- Exchange rate calculations (ignored entirely)

**Common Impact**: 
- Currency conversion errors in multi-national contexts
- Manual intervention required for currency mismatches
- Time-consuming debugging when assumptions fail

### 2. finance_agent_v2.yaml - Enhanced Multi-Currency Finance Agent

**Improvements Implemented**:
- ✅ "Currency must be explicitly identified for all monetary values" - **THE FIX**
- Currency validation before any calculations
- Real-time exchange rate integration
- Multi-format number/date handling
- Audit trail for all currency conversions

**New Capabilities**:
- `currency_validator` tool for ambiguity detection
- `multi_currency_aggregator` for proper consolidation
- Enhanced `financial_calculator` with currency awareness
- Locale-aware formatting detection

**Improvements**:
- Zero currency assumption errors
- Reduced need for manual fixes
- Better compliance with international accounting standards
- Clear audit trails for all conversions

### 3. audit_agent.yaml - Different Finance Domain

**Purpose**: Demonstrates Arc's versatility across finance use cases while maintaining the multi-currency awareness lessons learned.

**Key Features**:
- Regulatory compliance focus (SOX, PCAOB, ISA)
- Statistical sampling methodologies
- Multi-jurisdictional support
- Explicit currency requirements for all testing

## Currency Assumption Testing Framework

### Test Scenarios Generated

1. **Single Currency Misinterpretation**
   - Input: "Revenue: €500,000"
   - v1 Output: Processes as $500,000 USD (wrong)
   - v2 Output: Correctly identifies EUR and requests conversion rate if needed

2. **Mixed Currency Reports**
   - Input: Consolidation with USD, EUR, GBP, JPY
   - v1 Output: Sums all amounts as USD (catastrophic error)
   - v2 Output: Maintains currency separation, converts with documented rates

3. **Ambiguous Currency Symbols**
   - Input: "$1,000" (could be USD, CAD, AUD, etc.)
   - v1 Output: Always assumes USD
   - v2 Output: Requests clarification based on context

4. **Format Variations**
   - Input: "1.234,56" (European format)
   - v1 Output: Misinterprets as 1.23456
   - v2 Output: Correctly identifies as 1,234.56

## Integration with Arc Pipeline

These configurations integrate with:

1. **Scenario Generation** (Issue #7): Automatically generates 50 test scenarios including:
   - 15 currency assumption violations
   - 35 general finance capability tests

2. **Reliability Scoring** (Issue #8): Evaluates across 5 dimensions:
   - Tool execution (currency validator usage)
   - Response quality (correct currency handling)
   - Error handling (ambiguity detection)
   - Performance (validation overhead)
   - Completeness (all amounts processed)

3. **Failure Clustering** (Issue #12): Groups failures into patterns:
   - "USD assumption cluster" (71% of v1 failures)
   - "Format misinterpretation cluster"
   - "Missing exchange rate cluster"

4. **A/B Testing** (Issue #25): Statistical validation framework:
   - Compares baseline vs improved configurations
   - Provides statistical significance testing
   - Measures effect size of improvements

## Expected Value

These configurations help demonstrate:
- **Before (v1)**: How implicit assumptions can cause systematic failures
- **After (v2)**: How explicit validation and proper handling prevent issues
- **Quick Identification**: Arc can quickly identify patterns in agent failures
- **Actionable Insights**: Clear recommendations for improving agent reliability

## Usage

```bash
# Example workflow
arc run examples/configs/finance_agent_v1.yaml  # Run baseline agent
arc analyze                                      # Analyze failures
arc recommend                                    # Get improvement suggestions
arc validate-improvement finance_agent_v1.yaml finance_agent_v2.yaml
arc run examples/configs/finance_agent_v2.yaml  # Run improved agent
```

## Conclusion

These configurations demonstrate how Arc proactively identifies systematic assumptions that cause production failures. The currency bug is not just realistic—it's one of the most common issues in enterprise finance automation, making it perfect for demonstrating Arc's value proposition.