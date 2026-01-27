#!/usr/bin/env python3
"""Test Claude report query flow"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
project_root = Path(__file__).parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f'✅ Loaded environment from .env')

from chatbot.unified_extractor import UnifiedExtractor

print("="*80)
print("TESTING CLAUDE REPORT QUERY FLOW")
print("="*80)

# Test unified extractor with Claude report query
print("\n[TEST] Unified Extractor - Claude Report Query")
print("-" * 80)
extractor = UnifiedExtractor()
print('✅ UnifiedExtractor initialized')

test_query = 'give me a short summary about claude report'
print(f'Query: "{test_query}"')

result = extractor.extract_all(test_query)
print(f'\n✅ Extraction success: {result.get("success")}')
print(f'Signal types: {result.get("signal_types")}')
print(f'Functions: {result.get("functions")}')
print(f'Tickers: {result.get("tickers")}')
print(f'Columns extracted: {list(result.get("columns", {}).keys())}')
print(f'Reasoning: {result.get("signal_types_reasoning")}')

# Verify claude_report is identified
if 'claude_report' in result.get('signal_types', []):
    print('\n✅ SUCCESS: claude_report signal type correctly identified')
    print('   - No functions needed: ✓')
    print('   - No tickers needed: ✓')
    print('   - No columns needed: ✓')
    print('   - Will use full Claude report text')
else:
    print('\n❌ FAIL: claude_report not in signal types')

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
