#!/usr/bin/env python3
"""Simple test Claude report data loading without API calls"""

from chatbot.data_processor import DataProcessor

print("="*80)
print("TESTING CLAUDE REPORT DATA LOADING")
print("="*80)

# Test Data Processor - Load Claude Report
print("\n[TEST] Data Processor - Load Claude Report")
print("-" * 80)
data_processor = DataProcessor()
print('✅ DataProcessor initialized')

claude_report = data_processor.load_claude_report()
if claude_report:
    print(f'✅ Claude report loaded successfully!')
    print(f'   - Characters: {len(claude_report):,}')
    print(f'   - Lines: {claude_report.count(chr(10)):,}')
    print(f'\n   First 500 characters:')
    print(f'   {"-"*76}')
    for line in claude_report[:500].split('\n'):
        print(f'   {line}')
    print(f'   ...')
    print(f'   {"-"*76}')
    print(f'\n   Last 200 characters:')
    print(f'   {"-"*76}')
    print(f'   ...{claude_report[-200:]}')
    print(f'   {"-"*76}')
else:
    print('❌ Failed to load Claude report')

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
