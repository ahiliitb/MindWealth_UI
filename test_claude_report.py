#!/usr/bin/env python3
"""Test Claude report integration with unified extractor"""

import os
import sys
from pathlib import Path

# Load .env file (for local testing)
from dotenv import load_dotenv
project_root = Path(__file__).parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f'✅ Loaded environment from .env')
    print(f'CLAUDE_API_KEY set: {bool(os.environ.get("CLAUDE_API_KEY"))}')
else:
    print(f'⚠ .env file not found at {env_file}')
    sys.exit(1)

# NOW import chatbot modules after env vars are set
from chatbot.unified_extractor import UnifiedExtractor
from chatbot.data_processor import DataProcessor
from chatbot.chatbot_engine import ChatbotEngine

print("\n" + "="*80)
print("TESTING CLAUDE REPORT INTEGRATION")
print("="*80)

# Test 1: Unified Extractor
print("\n[TEST 1] Unified Extractor - Claude Report Query")
print("-" * 80)
extractor = UnifiedExtractor()
print('✅ UnifiedExtractor initialized')

result = extractor.extract_all('Show me the Claude comprehensive analysis report')
print(f'\n✅ Extraction result: {result.get("success")}')
print(f'Signal types: {result.get("signal_types")}')
print(f'Functions: {result.get("functions")}')
print(f'Tickers: {result.get("tickers")}')
print(f'Columns extracted for: {list(result.get("columns", {}).keys())}')
print(f'Reasoning: {result.get("signal_types_reasoning")}')

# Test 2: Data Processor - Load Claude Report
print("\n[TEST 2] Data Processor - Load Claude Report")
print("-" * 80)
data_processor = DataProcessor()
print('✅ DataProcessor initialized')

claude_report = data_processor.load_claude_report()
if claude_report:
    print(f'✅ Claude report loaded: {len(claude_report)} characters')
    print(f'First 200 chars: {claude_report[:200]}...')
else:
    print('❌ Failed to load Claude report')

# Test 3: Full Chatbot Engine Query
print("\n[TEST 3] Full Chatbot Engine - Claude Report Query")
print("-" * 80)
try:
    chatbot = ChatbotEngine()
    print('✅ ChatbotEngine initialized')
    
    # Query for Claude report
    response, metadata = chatbot.query('What are the top 5 signals from the Claude comprehensive analysis?')
    
    print(f'\n✅ Response generated')
    print(f'Signal types used: {metadata.get("signal_types", [])}')
    print(f'Claude report loaded: {metadata.get("claude_report_loaded", False)}')
    print(f'Tokens used: {metadata.get("tokens_used", {}).get("total", "N/A")}')
    print(f'\nResponse preview (first 300 chars):')
    print(response[:300])
    print('...')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
