"""
Main chatbot engine for processing queries and generating responses.
"""

import logging
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import time

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    SYSTEM_PROMPT,
    MAX_HISTORY_LENGTH,
    MAX_INPUT_TOKENS_PER_CALL,
    MAX_TICKERS_PER_QUERY,
    MAX_SEQUENTIAL_BATCHES,
    BATCH_DELAY_SECONDS,
    ESTIMATED_CHARS_PER_TOKEN,
    MIN_HISTORY_MESSAGES,
    ENABLE_BATCH_PROCESSING
)
from .data_processor import DataProcessor
from .history_manager import HistoryManager
from .function_extractor import FunctionExtractor
from .ticker_extractor import TickerExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotEngine:
    """
    Main chatbot engine that coordinates data processing, 
    GPT-4o interaction, and history management.
    """
    
    def __init__(
        self, 
        session_id: Optional[str] = None, 
        api_key: Optional[str] = None,
        use_new_data_structure: bool = True
    ):
        """
        Initialize chatbot engine.
        
        Args:
            session_id: Optional session ID for continuing previous conversation
            api_key: Optional OpenAI API key (uses env var if not provided)
            use_new_data_structure: Use new chatbot/data/{ticker}/YYYY-MM-DD.csv structure
        """
        self.api_key = api_key or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.data_processor = DataProcessor(use_new_structure=use_new_data_structure)
        self.history_manager = HistoryManager(session_id=session_id)
        self.function_extractor = FunctionExtractor(api_key=self.api_key)
        self.ticker_extractor = TickerExtractor(api_key=self.api_key)
        
        # Set available tickers for ticker extractor
        available_tickers = self.data_processor.get_available_tickers()
        self.ticker_extractor.set_available_tickers(available_tickers)
        
        # Add system prompt to history if this is a new session
        if not self.history_manager.conversation_history:
            self.history_manager.add_message("system", SYSTEM_PROMPT)
        
        logger.info(f"Initialized ChatbotEngine with session {self.history_manager.session_id}")
    
    def query(
        self,
        user_message: str,
        tickers: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        functions: Optional[List[str]] = None,
        signal_types: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        dedup_columns: Optional[List[str]] = None,
        auto_extract_functions: bool = True,
        auto_extract_tickers: bool = False
    ) -> Tuple[str, Dict]:
        """
        Process a user query with optional data context.
        
        Args:
            user_message: User's question or request
            tickers: List of ticker/asset symbols to include in context
            from_date: Start date for data filtering (YYYY-MM-DD)
            to_date: End date for data filtering (YYYY-MM-DD)
            functions: List of function names to filter (None = auto-extract or all functions)
            signal_types: List of signal types to filter (entry_exit, potential_achievement) - from UI checkboxes
            additional_context: Any additional text context to include
            dedup_columns: Columns to use for deduplication (None = use config default)
            auto_extract_functions: If True and functions=None, use GPT-4o-mini to extract
            auto_extract_tickers: If True and tickers=None, use GPT-4o-mini to extract asset names
            
        Note: Automatically loads data from BOTH signal and target folders
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            # Auto-extract tickers if enabled and not provided
            extracted_tickers = None
            if tickers is None and auto_extract_tickers:
                logger.info("Auto-extracting tickers from user query...")
                extracted_tickers = self.ticker_extractor.extract_tickers(user_message)
                
                if extracted_tickers:
                    # Empty list means ALL tickers (no specific ones mentioned)
                    if len(extracted_tickers) == 0:
                        logger.info(f"No specific tickers mentioned - limiting to top {MAX_TICKERS_PER_QUERY} for speed")
                        all_tickers = self.data_processor.get_available_tickers()
                        # Take first N tickers for fast response
                        tickers = all_tickers[:MAX_TICKERS_PER_QUERY]
                        logger.info(f"Using {len(tickers)} tickers for fast response: {tickers[:5]}...")
                    else:
                        logger.info(f"Auto-extracted specific tickers: {extracted_tickers}")
                        tickers = extracted_tickers
                else:
                    # None returned = error or no extraction, use limited set
                    logger.info(f"Using top {MAX_TICKERS_PER_QUERY} tickers (fallback)")
                    all_tickers = self.data_processor.get_available_tickers()
                    tickers = all_tickers[:MAX_TICKERS_PER_QUERY]
            
            # Auto-extract functions from user message if not provided
            extracted_functions = None
            if functions is None and auto_extract_functions and tickers:
                logger.info("Auto-extracting functions from user query...")
                extracted_functions = self.function_extractor.extract_functions(user_message)
                
                if extracted_functions:
                    logger.info(f"Auto-extracted functions: {extracted_functions}")
                    functions = extracted_functions
                else:
                    logger.info("No functions extracted from query - will load ALL available functions")
                    # Leave functions as None, which will load all functions in data_processor
            

            # Build context from data if parameters provided
            data_context = ""
            metadata = {
                "tickers": tickers or [],
                "tickers_auto_extracted": extracted_tickers or [],
                "from_date": from_date,
                "to_date": to_date,
                "functions": functions or [],
                "functions_auto_extracted": extracted_functions or [],
                "signal_types": signal_types or []
            }
            
            # Check if these exact parameters were used before in this conversation
            data_already_in_context = self._check_if_data_in_history(
                tickers, from_date, to_date, functions, signal_types
            )
            
            if tickers and not data_already_in_context:
                # Filter signal_types for stock data (exclude breadth)
                stock_signal_types = [st for st in (signal_types or []) if st != 'breadth'] if signal_types else None
                
                # Load stock data from selected folders based on signal_types
                # signal_types controls which folders to load from:
                # - ['entry_exit'] → signal/ folder only
                # - ['potential_achievement'] → target/ folder only  
                # - Both or None → both folders
                stock_data = self.data_processor.load_stock_data(
                    tickers, from_date, to_date, dedup_columns, functions, stock_signal_types
                )
                
                # Load breadth data if requested
                if signal_types and 'breadth' in signal_types:
                    breadth_data = self.data_processor.load_breadth_data(from_date, to_date)
                    if breadth_data is not None:
                        # Add breadth data as a special "MARKET_BREADTH" ticker
                        stock_data['MARKET_BREADTH'] = breadth_data
                        logger.info("Added breadth report to data context")
                
                # Format data for prompt
                data_context = self.data_processor.format_data_for_prompt(stock_data)
                
                metadata["data_loaded"] = {
                    "assets": list(stock_data.keys()),
                    "total_records": sum(len(df) for df in stock_data.values())
                }
            elif tickers and data_already_in_context:
                # Data already in conversation history, skip reloading
                metadata["data_reused_from_history"] = True
                metadata["note"] = "Using data from previous query in conversation history"
                logger.info(f"Reusing data from history for tickers: {tickers}, dates: {from_date} to {to_date}")
            
            # Build complete user message
            complete_message = user_message
            
            if data_context:
                complete_message = f"""User Query: {user_message}

{data_context}"""
            
            if additional_context:
                complete_message += f"\n\nAdditional Context:\n{additional_context}"
            
            # Add user message to history
            self.history_manager.add_message("user", complete_message, metadata)
            
            # Get conversation history for API
            messages = self.history_manager.get_messages_for_api()
            
            # PRE-FLIGHT TOKEN CHECK
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
            
            logger.info(f"Estimated input tokens: {estimated_tokens}")
            
            # Check if we should use batch processing
            use_batching = ENABLE_BATCH_PROCESSING and estimated_tokens > MAX_INPUT_TOKENS_PER_CALL
            
            if use_batching:
                logger.info(f"Input too large ({estimated_tokens} tokens) - using SEQUENTIAL BATCH processing with rate limiting")
                assistant_message = self._batch_query(messages, user_message, stock_data if tickers else {})
                
                # Create mock response object for metadata
                metadata["model"] = OPENAI_MODEL
                metadata["tokens_used"] = {
                    "prompt": estimated_tokens,
                    "completion": len(assistant_message) // ESTIMATED_CHARS_PER_TOKEN,
                    "total": estimated_tokens + (len(assistant_message) // ESTIMATED_CHARS_PER_TOKEN)
                }
                metadata["finish_reason"] = "batch_aggregation"
                metadata["batch_processing_used"] = True
            else:
                # Single API call - trim if needed
                trimmed_count = 0
                while estimated_tokens > MAX_INPUT_TOKENS_PER_CALL and len(messages) > MIN_HISTORY_MESSAGES:
                    if len(messages) > 2:
                        removed_msg = messages.pop(1)
                        trimmed_count += 1
                        logger.warning(f"Trimmed old message (role: {removed_msg.get('role', 'unknown')})")
                        
                        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                        estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
                    else:
                        break
                
                if trimmed_count > 0:
                    logger.warning(f"Trimmed {trimmed_count} old messages to stay under token limit")
                
                # Call GPT
                logger.info(f"Calling {OPENAI_MODEL} with {len(messages)} messages, ~{estimated_tokens} tokens")
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                assistant_message = response.choices[0].message.content
                
                # Add response metadata
                metadata["model"] = OPENAI_MODEL
                metadata["tokens_used"] = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
                metadata["finish_reason"] = response.choices[0].finish_reason
            
            # Add assistant response to history
            self.history_manager.add_message("assistant", assistant_message, metadata)
            
            logger.info(f"Generated response with {metadata['tokens_used']['total']} tokens")
            
            return assistant_message, metadata
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message)
            return error_message, {"error": str(e)}
    
    def query_with_csv_text(
        self,
        user_message: str,
        csv_text: str,
        additional_context: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Process a query with raw CSV text provided directly.
        
        Args:
            user_message: User's question or request
            csv_text: CSV data as text
            additional_context: Any additional text context
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        complete_message = f"""User Query: {user_message}

=== PROVIDED DATA ===
{csv_text}"""
        
        if additional_context:
            complete_message += f"\n\nAdditional Context:\n{additional_context}"
        
        metadata = {"input_type": "csv_text"}
        
        # Add user message to history
        self.history_manager.add_message("user", complete_message, metadata)
        
        # Get conversation history for API
        messages = self.history_manager.get_messages_for_api()
        
        try:
            # Call GPT-4o
            logger.info("Calling GPT-4o with CSV text input")
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Add response metadata
            metadata["model"] = OPENAI_MODEL
            metadata["tokens_used"] = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
            
            # Add assistant response to history
            self.history_manager.add_message("assistant", assistant_message, metadata)
            
            return assistant_message, metadata
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message)
            return error_message, {"error": str(e)}
    
    def get_session_id(self) -> str:
        """Get current session ID."""
        return self.history_manager.session_id
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history."""
        return self.history_manager.get_full_history()
    
    def get_session_summary(self) -> Dict:
        """Get session statistics and metadata."""
        return self.history_manager.get_session_summary()
    
    def clear_history(self):
        """Clear conversation history and reinitialize with system prompt."""
        self.history_manager.clear_history()
        self.history_manager.add_message("system", SYSTEM_PROMPT)
        logger.info("Conversation history cleared")
    
    def _batch_query(
        self,
        base_messages: List[Dict],
        user_query: str,
        stock_data: Dict
    ) -> str:
        """
        Execute SEQUENTIAL batch API calls for large data sets with rate limiting.
        Splits data by ticker groups and makes sequential calls with delays to avoid TPM limits.
        
        Args:
            base_messages: Base conversation messages
            user_query: User's query text
            stock_data: Dictionary of stock DataFrames by ticker
            
        Returns:
            Aggregated response text
        """
        import pandas as pd
        
        logger.info(f"Starting batch query with {len(stock_data)} tickers")
        
        # Split tickers into groups
        ticker_list = list(stock_data.keys())
        tickers_per_batch = max(1, len(ticker_list) // MAX_SEQUENTIAL_BATCHES)
        
        ticker_groups = []
        for i in range(0, len(ticker_list), tickers_per_batch):
            ticker_groups.append(ticker_list[i:i + tickers_per_batch])
        
        logger.info(f"Split into {len(ticker_groups)} batches: {[len(g) for g in ticker_groups]} tickers each")
        
        # Execute SEQUENTIAL batch calls with delays
        results = []
        
        for group_idx, group_tickers in enumerate(ticker_groups):
            try:
                # Create data context for this group
                group_data = {t: stock_data[t] for t in group_tickers if t in stock_data}
                group_context = self.data_processor.format_data_for_prompt(
                    group_data, 
                    max_tokens=MAX_INPUT_TOKENS_PER_CALL
                )
                
                # Create message for this group
                group_message = f"""User Query: {user_query}

{group_context}

Note: This is batch {group_idx + 1} of {len(ticker_groups)} analyzing assets: {', '.join(group_tickers)}"""
                
                # Create messages for this call
                group_messages = base_messages[:-1]  # All except last message
                group_messages.append({"role": "user", "content": group_message})
                
                # Add delay before API call (except for first call)
                if group_idx > 0:
                    logger.info(f"Waiting {BATCH_DELAY_SECONDS}s before next batch to avoid rate limits...")
                    time.sleep(BATCH_DELAY_SECONDS)
                
                # Call API
                logger.info(f"Calling API for batch {group_idx + 1}/{len(ticker_groups)}")
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=group_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                
                result_text = response.choices[0].message.content
                tokens = response.usage.total_tokens
                
                logger.info(f"Batch {group_idx + 1} completed: {tokens} tokens")
                results.append({
                    'group_idx': group_idx,
                    'tickers': group_tickers,
                    'response': result_text,
                    'tokens': tokens
                })
                
            except Exception as e:
                logger.error(f"Error in batch {group_idx + 1}: {e}")
                results.append({
                    'group_idx': group_idx,
                    'tickers': group_tickers,
                    'response': f"Error analyzing {', '.join(group_tickers)}: {str(e)}",
                    'tokens': 0
                })
        
        # Aggregate responses
        total_tokens = sum(r['tokens'] for r in results)
        logger.info(f"Batch processing complete: {len(results)} batches, {total_tokens} total tokens")
        
        # Combine responses
        combined_response = f"""# Comprehensive Analysis ({len(ticker_list)} assets analyzed in {len(results)} batches)

"""
        
        for result in results:
            if result['response']:
                combined_response += f"\n## Batch {result['group_idx'] + 1}: {', '.join(result['tickers'][:5])}"
                if len(result['tickers']) > 5:
                    combined_response += f" ... and {len(result['tickers']) - 5} more"
                combined_response += f"\n\n{result['response']}\n\n---\n"
        
        # Add overall note
        combined_response += f"\n\n**Analysis Complete**: Processed {len(ticker_list)} assets using sequential batch processing with rate limiting."
        
        return combined_response
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available ticker/asset symbols from both signal and target."""
        return self.data_processor.get_available_tickers()
    
    def get_available_functions(self, ticker: Optional[str] = None) -> List[str]:
        """
        Get list of available function names from both signal and target.
        
        Args:
            ticker: Optional ticker to get functions for
            
        Returns:
            List of function names
        """
        return self.data_processor.get_available_functions(ticker)
    
    def _check_if_data_in_history(
        self,
        tickers: Optional[List[str]],
        from_date: Optional[str],
        to_date: Optional[str],
        functions: Optional[List[str]],
        signal_types: Optional[List[str]] = None
    ) -> bool:
        """
        Check if the same data parameters were used in a previous query.
        
        Args:
            tickers: List of ticker symbols
            from_date: Start date
            to_date: End date
            functions: List of function names
            signal_types: List of signal types to filter
            
        Returns:
            True if exact same parameters found in history, False otherwise
        """
        if not tickers:
            return False
        
        # Get conversation history
        history = self.history_manager.get_full_history()
        
        # Check last few messages for matching parameters
        for msg in reversed(history[-10:]):  # Check last 10 messages
            if msg.get("role") != "user":
                continue
            
            msg_metadata = msg.get("metadata", {})
            
            # Check if tickers match
            prev_tickers = msg_metadata.get("tickers", [])
            if not prev_tickers or set(prev_tickers) != set(tickers):
                continue
            
            # Check if dates match
            prev_from = msg_metadata.get("from_date")
            prev_to = msg_metadata.get("to_date")
            if prev_from != from_date or prev_to != to_date:
                continue
            
            # Check if functions match
            prev_functions = msg_metadata.get("functions", [])
            current_functions = functions or []
            if set(prev_functions) != set(current_functions):
                continue
            
            # Check if signal_types match
            prev_signal_types = msg_metadata.get("signal_types", [])
            current_signal_types = signal_types or []
            if set(prev_signal_types) != set(current_signal_types):
                continue
            
            # If we got here, parameters match!
            logger.info(
                f"Found matching data in history: tickers={tickers}, "
                f"from={from_date}, to={to_date}, functions={functions}, "
                f"signal_types={signal_types}"
            )
            return True
        
        return False

