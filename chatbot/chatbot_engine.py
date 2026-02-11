"""
Main chatbot engine for processing queries and generating responses.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from anthropic import Anthropic
import time

from .config import (
    CLAUDE_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
    OPENAI_API_KEY,
    SYSTEM_PROMPT,
    MAX_HISTORY_LENGTH,
    MAX_INPUT_TOKENS_PER_CALL,
    MAX_SEQUENTIAL_BATCHES,
    BATCH_DELAY_SECONDS,
    ESTIMATED_CHARS_PER_TOKEN,
    MIN_HISTORY_MESSAGES,
    MAX_ROWS_TO_INCLUDE,
    MAX_TOKENS,  # Deprecated, for backward compatibility
    TEMPERATURE  # Deprecated, for backward compatibility
)
from .data_processor import DataProcessor
from .history_manager import HistoryManager
from .unified_extractor import UnifiedExtractor
from .smart_data_fetcher import SmartDataFetcher
from .signal_extractor import SignalExtractor
from .signal_type_selector import SignalTypeSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotEngine:
    """
    Main chatbot engine that coordinates data processing, 
    using Claude Sonnet 4.5 for all AI operations (extraction and responses).
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
            api_key: Optional Claude API key (uses env var if not provided)
            use_new_data_structure: Use new chatbot/data/{ticker}/YYYY-MM-DD.csv structure
        """
        self.claude_api_key = api_key or CLAUDE_API_KEY
        
        if not self.claude_api_key:
            raise ValueError(
                "Claude API key not provided. Set CLAUDE_API_KEY in .streamlit/secrets.toml "
                "or .env file."
            )
        
        # Initialize Claude client (for all AI operations)
        try:
            self.claude_client = Anthropic(api_key=self.claude_api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Claude client: {e}. Check your API key.")
        
        self.data_processor = DataProcessor(use_new_structure=use_new_data_structure)
        self.history_manager = HistoryManager(session_id=session_id)
        # Use OpenAI API key for extraction with GPT-5.2
        self.unified_extractor = UnifiedExtractor(api_key=OPENAI_API_KEY)
        self.smart_data_fetcher = SmartDataFetcher()
        self.signal_extractor = SignalExtractor()
        self.signal_type_selector = SignalTypeSelector(api_key=OPENAI_API_KEY)
        
        # Set available tickers for unified extractor
        available_tickers = self.data_processor.get_available_tickers()
        self.unified_extractor.set_available_tickers(available_tickers)
        
        # Add system prompt to history if this is a new session
        if not self.history_manager.conversation_history:
            self.history_manager.add_message("system", SYSTEM_PROMPT)
        
        logger.info(f"Initialized ChatbotEngine with session {self.history_manager.session_id}")
    
    def _convert_to_claude_format(self, messages: List[Dict]) -> Dict:
        """
        Convert message history to Claude format.
        Claude requires system message separate from conversation messages.
        
        Args:
            messages: List of messages with 'role' and 'content'
            
        Returns:
            Dict with 'system' and 'messages' keys for Claude API
        """
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # Claude requires system message to be separate
                system_message = content
            elif role in ["user", "assistant"]:
                # Claude uses same role names
                claude_messages.append({
                    "role": role,
                    "content": content
                })
        
        return {
            "system": system_message if system_message else SYSTEM_PROMPT,
            "messages": claude_messages
        }
    
    def _call_openai_api(self, messages: List[Dict], model: str = None, temperature: float = None) -> str:
        """
        Helper method to call OpenAI API and return the response content.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to CLAUDE_MODEL from config)
            temperature: Temperature setting (defaults to TEMPERATURE from config)
            
        Returns:
            Response content as string
        """
        try:
            response = self.client.chat.completions.create(
                model=model or CLAUDE_MODEL,
                messages=messages,
                max_completion_tokens=MAX_TOKENS,
                temperature=temperature or TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error: {str(e)}"
    
    def determine_signal_types(self, user_message: str) -> Tuple[List[str], str]:
        """
        Determine which signal types should be fetched for a given user message.
        Uses the unified extractor to determine signal types.
        """
        # Get conversation history for context
        from .config import MAX_EXTRACTION_HISTORY_LENGTH
        conversation_history = self.history_manager.get_messages_for_api(max_pairs=MAX_EXTRACTION_HISTORY_LENGTH)
        
        extraction_result = self.unified_extractor.extract_all(user_message, conversation_history=conversation_history)
        
        if extraction_result.get("success", False):
            signal_types = extraction_result.get("signal_types", ["entry", "exit", "portfolio_target_achieved"])
            reasoning = extraction_result.get("signal_types_reasoning", "")
            return signal_types, reasoning
        else:
            # Fallback to defaults if extraction fails
            logger.warning(f"Signal type extraction failed: {extraction_result.get('error', 'Unknown error')}")
            return ["entry", "exit", "portfolio_target_achieved"], "Using default signal types due to extraction error"
    
    def _prepare_user_metadata(self, metadata: Optional[Dict[str, Any]], user_message: str) -> Dict[str, Any]:
        """
        Create a safe copy of metadata with the original user prompt for UI display.
        """
        meta_copy: Dict[str, Any] = dict(metadata) if metadata else {}
        meta_copy["display_prompt"] = (user_message or "").strip()
        return meta_copy
    
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
        auto_extract_tickers: bool = False,
        is_followup: bool = False
    ) -> Tuple[str, Dict]:
        """
        Process a user query with optional data context.
        
        Args:
            user_message: User's question or request
            tickers: List of ticker/asset symbols to include in context
            from_date: Start date for data filtering (YYYY-MM-DD)
            to_date: End date for data filtering (YYYY-MM-DD)
            functions: List of function names to filter (None = auto-extract or all functions)
            signal_types: List of signal types to filter (entry_exit, portfolio_target_achieved, breadth) - from UI checkboxes
            additional_context: Any additional text context to include
            dedup_columns: Columns to use for deduplication (None = use config default)
            auto_extract_functions: If True and functions=None, use GPT-5.2 to extract
            auto_extract_tickers: If True and tickers=None, use GPT-5.2 to extract asset names
            is_followup: If True, skip data loading and use existing conversation context
            
        Note: Automatically loads data from BOTH signal and portfolio_target_achieved folders
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            # FOLLOW-UP QUESTION MODE: Skip data loading, use existing context
            if is_followup:
                logger.info("Follow-up question mode: Using existing conversation context")
                
                metadata = {
                    "is_followup": True,
                    "message": "Follow-up question using previous data context"
                }
                
                # Add user message to history
                self.history_manager.add_message(
                    "user",
                    user_message,
                    self._prepare_user_metadata(metadata, user_message)
                )
                
                # Get conversation history for API
                messages = self.history_manager.get_messages_for_api()
                
                # Estimate tokens
                total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
                
                logger.info(f"Follow-up query with ~{estimated_tokens} tokens")
                
                # Use simple batch processing for follow-up
                assistant_message, batch_metadata = self._simple_batch_query(messages, estimated_tokens)
                
                # Update metadata
                metadata["model"] = CLAUDE_MODEL
                metadata["tokens_used"] = batch_metadata["tokens_used"]
                metadata["finish_reason"] = batch_metadata["finish_reason"]
                metadata["batch_processing_used"] = True
                metadata["batch_count"] = batch_metadata.get("batch_count", 1)
                metadata["batch_mode"] = batch_metadata.get("batch_mode", "single")
                
                # Add assistant response to history
                self.history_manager.add_message("assistant", assistant_message, metadata)
                
                logger.info(f"Follow-up response generated with {metadata['tokens_used']['total']} tokens")
                
                return assistant_message, metadata
            
            # REGULAR QUERY MODE: Load fresh data
            logger.info("Regular query mode: Loading fresh data")
            
            signal_type_reasoning = ""
            if not signal_types:
                signal_types, signal_type_reasoning = self.determine_signal_types(user_message)
            else:
                signal_types = [stype for stype in signal_types if stype]
                if not signal_types:
                    signal_types, signal_type_reasoning = self.determine_signal_types(user_message)
            selected_signal_types = signal_types
            
            # Auto-extract tickers if enabled and not provided
            extracted_tickers = None
            if tickers is None and auto_extract_tickers:
                logger.info("Auto-extracting tickers from user query using unified extractor...")
                from .config import MAX_EXTRACTION_HISTORY_LENGTH
                conversation_history = self.history_manager.get_messages_for_api(max_pairs=MAX_EXTRACTION_HISTORY_LENGTH)
                extraction_result = self.unified_extractor.extract_all(user_message, conversation_history=conversation_history)
                if extraction_result.get("success", False):
                    extracted_tickers = extraction_result.get("tickers")  # None means ALL
                    if extracted_tickers:
                        logger.info(f"Auto-extracted specific tickers: {extracted_tickers[:10]}{'...' if len(extracted_tickers) > 10 else ''}")
                        tickers = extracted_tickers
                    else:
                        logger.info("No specific tickers mentioned - will use smart filtering based on functions")
                        tickers = None
                else:
                    logger.warning("Ticker extraction failed, using all tickers")
                    tickers = None
            # Auto-extract functions from user message if not provided
            extracted_functions = None
            if functions is None and auto_extract_functions:
                logger.info("Auto-extracting functions from user query using unified extractor...")
                # Reuse extraction result if we already extracted for tickers
                if extracted_tickers is not None or not auto_extract_tickers:
                    from .config import MAX_EXTRACTION_HISTORY_LENGTH
                    conversation_history = self.history_manager.get_messages_for_api(max_pairs=MAX_EXTRACTION_HISTORY_LENGTH)
                    extraction_result = self.unified_extractor.extract_all(user_message, conversation_history=conversation_history)
                    if extraction_result.get("success", False):
                        extracted_functions = extraction_result.get("functions")  # None means ALL
                
                if extracted_functions:
                    logger.info(f"Auto-extracted functions: {extracted_functions}")
                    functions = extracted_functions
                else:
                    logger.info("No functions extracted from query - will load ALL available functions")
                    # Leave functions as None, which will load all functions in data_processor
            
            # SMART FILTERING: Intelligently determine which tickers to use
            if tickers is None and auto_extract_tickers:
                # No specific tickers mentioned - use smart filtering
                if functions:
                    # CASE 1: Function(s) specified, no specific tickers
                    # Get ALL tickers that have the requested function(s)
                    logger.info(f"Smart filtering: Finding ALL tickers with function(s): {functions}")
                    tickers_with_function = []
                    for ticker in self.data_processor.get_available_tickers():
                        available_functions = self.data_processor.get_available_functions(ticker)
                        # Check if ticker has any of the requested functions
                        if any(func in available_functions for func in functions):
                            tickers_with_function.append(ticker)
                    
                    if tickers_with_function:
                        tickers = tickers_with_function
                        logger.info(f"Smart filtering: Found {len(tickers)} tickers with function(s) {functions}")
                        logger.info(f"Tickers: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")
                    else:
                        logger.warning(f"No tickers found with function(s) {functions}")
                        tickers = []
                else:
                    # CASE 2: No specific tickers or functions mentioned
                    # Use ALL available tickers - batch processing will handle it
                    tickers = self.data_processor.get_available_tickers()
                    logger.info(f"Smart filtering: No specific tickers/functions - using ALL {len(tickers)} tickers")
                    logger.info(f"Batch processing will handle the load efficiently")
            elif tickers and functions and len(tickers) > 0:
                # CASE 3: Both tickers and functions specified
                # Filter to only those tickers that have the requested functions
                if extracted_tickers is not None and len(extracted_tickers) == 0:
                    logger.info(f"Smart filtering: Filtering tickers for function(s): {functions}")
                    tickers_with_function = []
                    for ticker in tickers:
                        available_functions = self.data_processor.get_available_functions(ticker)
                        if any(func in available_functions for func in functions):
                            tickers_with_function.append(ticker)
                    
                    if tickers_with_function:
                        tickers = tickers_with_function
                        logger.info(f"Smart filtering: {len(tickers)} tickers have the requested function(s)")
                    else:
                        logger.warning(f"No tickers found with function(s) {functions}, using all tickers")

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
            
            # Determine what data to load based on signal_types
            stock_signal_types = [st for st in (signal_types or []) if st not in ['breadth', 'claude_report']] if signal_types else None
            load_breadth = signal_types and 'breadth' in signal_types
            load_claude_report = signal_types and 'claude_report' in signal_types
            
            # Check if we need stock data (entry/exit/portfolio_target_achieved)
            need_stock_data = tickers and stock_signal_types and not data_already_in_context
            
            # Initialize stock_data
            stock_data = {}
            
            if need_stock_data:
                # Load stock data from selected folders based on signal_types
                # signal_types controls which folders to load from:
                # - ['entry'] → entry/ folder only (open positions)
                # - ['exit'] → exit/ folder only (completed trades)
                # - ['portfolio_target_achieved'] → portfolio_target_achieved/ folder only (portfolio target achieved)
                # - Multiple or None → load from selected folders
                stock_data = self.data_processor.load_stock_data(
                    tickers, from_date, to_date, dedup_columns, functions, stock_signal_types
                )
                logger.info(f"Loaded stock data for {len(stock_data)} assets")
                
            elif tickers and data_already_in_context:
                # Data already in conversation history, skip reloading
                metadata["data_reused_from_history"] = True
                metadata["note"] = "Using data from previous query in conversation history"
                logger.info(f"Reusing data from history for tickers: {tickers}, dates: {from_date} to {to_date}")
            
            # Load breadth data if requested (independent of tickers)
            if load_breadth:
                breadth_data = self.data_processor.load_breadth_data(from_date, to_date)
                if breadth_data is not None:
                    # Add breadth data as a special "MARKET_BREADTH" ticker
                    stock_data['MARKET_BREADTH'] = breadth_data
                    logger.info("Added breadth report to data context")
            
            # Load Claude report if requested (independent of tickers, no table data)
            claude_report_text = None
            if load_claude_report:
                claude_report_text = self.data_processor.load_claude_report()
                if claude_report_text:
                    logger.info("Loaded Claude comprehensive analysis report")
                    metadata["claude_report_loaded"] = True
                else:
                    logger.warning("Claude report requested but not found")
                    metadata["claude_report_loaded"] = False
            
            # Format data for prompt if we have any data
            if stock_data:
                data_context = self.data_processor.format_data_for_prompt(stock_data)
                
                total_records = sum(len(df) for df in stock_data.values())
                metadata["data_loaded"] = {
                    "assets": list(stock_data.keys()),
                    "total_records": total_records
                }
                
                # CHECK: If data loaded but empty (no records), return "No Signal Found"
                if total_records == 0:
                    no_signal_message = "**No Signal Found**\n\nNo trading signals were found for the specified criteria."
                    
                    if tickers:
                        no_signal_message += f"\n\n**Searched for:**\n- Assets: {', '.join(tickers[:10])}"
                        if len(tickers) > 10:
                            no_signal_message += f" (and {len(tickers) - 10} more)"
                    if functions:
                        no_signal_message += f"\n- Functions: {', '.join(functions)}"
                    if from_date and to_date:
                        no_signal_message += f"\n- Date range: {from_date} to {to_date}"
                    if signal_types:
                        no_signal_message += f"\n- Signal types: {', '.join(signal_types)}"
                    
                    no_signal_message += "\n\n💡 *Try expanding your date range or adjusting search criteria.*"
                    
                    metadata["no_data_found"] = True
                    self.history_manager.add_message(
                        "user",
                        user_message,
                        self._prepare_user_metadata(metadata, user_message)
                    )
                    self.history_manager.add_message("assistant", no_signal_message, metadata)
                    
                    return no_signal_message, metadata
            
            # CHECK: If no data found and tickers were expected, return "No Signal Found"
            if not stock_data and not data_already_in_context and (tickers is not None or auto_extract_tickers):
                # No data was loaded and we were looking for ticker-specific data
                no_signal_message = "**No Signal Found**\n\nNo trading signals were found matching your criteria."
                
                if tickers is not None and len(tickers) == 0:
                    # Tickers list is empty - no matching tickers found
                    if functions:
                        no_signal_message += f"\n\n**Searched for:**\n- Functions: {', '.join(functions)}"
                        no_signal_message += "\n\n*No assets found with the specified function(s).*"
                    else:
                        no_signal_message += "\n\n*No assets match your criteria.*"
                elif tickers:
                    # Tickers were specified but no data loaded
                    no_signal_message += f"\n\n**Searched for:**\n- Assets: {', '.join(tickers[:10])}"
                    if len(tickers) > 10:
                        no_signal_message += f" (and {len(tickers) - 10} more)"
                    if functions:
                        no_signal_message += f"\n- Functions: {', '.join(functions)}"
                    if from_date and to_date:
                        no_signal_message += f"\n- Date range: {from_date} to {to_date}"
                
                no_signal_message += "\n\n💡 *Try adjusting your search criteria or date range.*"
                
                metadata["no_data_found"] = True
                self.history_manager.add_message(
                    "user",
                    user_message,
                    self._prepare_user_metadata(metadata, user_message)
                )
                self.history_manager.add_message("assistant", no_signal_message, metadata)
                
                return no_signal_message, metadata
            
            # Build complete user message
            complete_message = user_message
            
            # Build message with data context and/or Claude report
            if claude_report_text:
                # For Claude report queries, add the report text directly (no table data)
                complete_message = f"""User Query: {user_message}

=== CLAUDE COMPREHENSIVE ANALYSIS REPORT ===

{claude_report_text}

=== END REPORT ===

Please answer the user's query based on the comprehensive analysis report above."""
            elif data_context:
                # For regular queries with table data
                complete_message = f"""User Query: {user_message}

{data_context}"""
            
            if additional_context:
                complete_message += f"\n\nAdditional Context:\n{additional_context}"
            
            # Add user message to history
            self.history_manager.add_message(
                "user",
                complete_message,
                self._prepare_user_metadata(metadata, user_message)
            )
            
            # Get conversation history for API
            messages = self.history_manager.get_messages_for_api()
            
            # PRE-FLIGHT TOKEN CHECK
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
            
            logger.info(f"Estimated input tokens: {estimated_tokens}")
            
            # ALWAYS USE SMART BATCH PROCESSING
            # Automatically handles small datasets (1 API call) vs large datasets (multiple API calls)
            # Number of API calls is proportional to input token count
            if stock_data and tickers:
                logger.info(f"Using smart batch processing for {len(stock_data)} tickers")
                assistant_message, batch_metadata = self._smart_batch_query(
                    messages, user_message, stock_data, estimated_tokens
                )
                
                # Update metadata with batch info
                metadata["model"] = CLAUDE_MODEL
                metadata["tokens_used"] = batch_metadata["tokens_used"]
                metadata["finish_reason"] = batch_metadata["finish_reason"]
                metadata["batch_processing_used"] = True
                metadata["batch_count"] = batch_metadata["batch_count"]
                metadata["batch_mode"] = batch_metadata["batch_mode"]
            else:
                # For non-ticker queries (breadth only, Claude report, CSV text, etc.)
                # Use simple batch method with automatic single/multi decision
                query_type = "Claude report" if load_claude_report else "non-ticker query"
                logger.info(f"Using simple batch processing for {query_type}")
                assistant_message, batch_metadata = self._simple_batch_query(
                    messages, estimated_tokens
                )
                
                # Update metadata
                metadata["model"] = CLAUDE_MODEL
                metadata["tokens_used"] = batch_metadata["tokens_used"]
                metadata["finish_reason"] = batch_metadata["finish_reason"]
                metadata["batch_processing_used"] = True
                metadata["batch_count"] = batch_metadata.get("batch_count", 1)
                metadata["batch_mode"] = batch_metadata.get("batch_mode", "single")
            
            # Extract full signal tables with all columns
            query_params = {
                'assets': tickers,
                'functions': functions,
                'from_date': from_date,
                'to_date': to_date,
                'signal_types': signal_types
            }
            # Convert stock_data format to fetched_data format for signal extraction
            fetched_data_for_signals = {}
            if 'stock_data' in locals() and stock_data:
                # Transform ticker-keyed data to signal-type-keyed data
                for ticker, ticker_df in stock_data.items():
                    if not ticker_df.empty and hasattr(ticker_df, 'columns'):
                        # Determine signal type from the data structure
                        if 'SignalType' in ticker_df.columns:
                            # Group by signal type
                            for signal_type in ticker_df['SignalType'].unique():
                                if signal_type not in fetched_data_for_signals:
                                    fetched_data_for_signals[signal_type] = []
                                signal_type_data = ticker_df[ticker_df['SignalType'] == signal_type].copy()
                                fetched_data_for_signals[signal_type].append(signal_type_data)
                        else:
                            # Default to 'entry' if no signal type column
                            if 'entry' not in fetched_data_for_signals:
                                fetched_data_for_signals['entry'] = []
                            fetched_data_for_signals['entry'].append(ticker_df)
                
                # Concatenate DataFrames for each signal type
                for signal_type, df_list in fetched_data_for_signals.items():
                    if df_list:
                        import pandas as pd
                        fetched_data_for_signals[signal_type] = pd.concat(df_list, ignore_index=True)
            
            full_signal_tables = self.signal_extractor.extract_full_signal_tables(
                assistant_message,
                fetched_data_for_signals,
                query_params
            )
            metadata["full_signal_tables"] = full_signal_tables
            
            # Keep legacy for compatibility
            signals_df = self.signal_extractor.extract_signals_from_response(
                assistant_message,
                stock_data if 'stock_data' in locals() else None
            )
            metadata["signals_table"] = signals_df
            
            # Add assistant response to history
            self.history_manager.add_message("assistant", assistant_message, metadata)
            
            logger.info(f"Generated response with {metadata['tokens_used']['total']} tokens")
            
            return assistant_message, metadata
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message)
            return error_message, {"error": str(e)}
    
    def smart_query(
        self,
        user_message: str,
        selected_signal_types: List[str],
        assets: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        functions: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        auto_extract_tickers: bool = False,
        signal_type_reasoning: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Process a query using the two-stage smart column selection system.
        
        Stage 1: Use GPT with chatbot.txt prompt to identify required columns
        Stage 2: Fetch only those columns from the data and process the query
        
        Args:
            user_message: User's question or request
            selected_signal_types: Signal types selected by user (checkboxes): entry, exit, portfolio_target_achieved, breadth
            assets: Optional list of asset/ticker names to filter
            from_date: Start date for data filtering (YYYY-MM-DD)
            to_date: End date for data filtering (YYYY-MM-DD)
            functions: Optional list of function names to filter
            additional_context: Any additional text context to include
            auto_extract_tickers: If True and assets=None, use GPT to extract asset names
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            logger.info("="*60)
            logger.info("SMART QUERY MODE - Two-stage column selection")
            logger.info("="*60)

            # Standard no-data message used by UI when there are no rows
            NO_DATA_MESSAGE = "No Signal for Choosen Date Range, Please change range and try again"

            # STAGE 1: UNIFIED EXTRACTION - Extract everything in ONE GPT call
            logger.info("STAGE 1: Using unified extractor (single GPT call for all extractions)...")
            
            # Get conversation history for context (helps with follow-up queries like "show me those" or "for the same tickers")
            from .config import MAX_EXTRACTION_HISTORY_LENGTH
            conversation_history = self.history_manager.get_messages_for_api(max_pairs=MAX_EXTRACTION_HISTORY_LENGTH)
            
            extraction_result = self.unified_extractor.extract_all(user_message, conversation_history=conversation_history)
            
            if not extraction_result.get("success", False):
                error_msg = extraction_result.get("error", "Unknown error")
                logger.error(f"Unified extraction failed: {error_msg}")
                return f"Error extracting query components: {error_msg}", {"error": error_msg}
            
            # Extract all components from unified result
            # Only use extracted signal types if none were provided by caller
            if not selected_signal_types:
                selected_signal_types = extraction_result.get("signal_types", [])
                signal_type_reasoning = extraction_result.get("signal_types_reasoning", "")
                logger.info(f"Using AI-extracted signal types: {selected_signal_types}")
            else:
                # Keep the provided signal types
                logger.info(f"Using provided signal types: {selected_signal_types}")
                if not signal_type_reasoning:
                    signal_type_reasoning = extraction_result.get("signal_types_reasoning", "")
            
            # SPECIAL CASE: If claude_report is in signal types, route to old query() method
            # Claude report doesn't need table data, functions, or column extraction
            if 'claude_report' in selected_signal_types:
                logger.info("CLAUDE_REPORT detected - routing to old query() method for text-based analysis")
                return self.query(
                    user_message=user_message,
                    signal_types=selected_signal_types,
                    tickers=None,  # Not needed for claude_report
                    from_date=from_date,
                    to_date=to_date,
                    dedup_columns=None,  # Not needed for claude_report
                    functions=None,  # Not needed for claude_report
                    additional_context=additional_context,
                    auto_extract_tickers=False,  # Not needed for claude_report
                    auto_extract_functions=False,  # Not needed for claude_report
                    is_followup=False
                )
            
            # Use extracted functions if not provided
            if functions is None:
                functions = extraction_result.get("functions")  # None means ALL
                if functions:
                    logger.info(f"Extracted functions: {functions}")
                else:
                    logger.info("No specific functions mentioned - will load ALL functions")
            
            # Use extracted tickers/assets if not provided
            if assets is None and auto_extract_tickers:
                assets = extraction_result.get("tickers")  # None means ALL
                if assets:
                    logger.info(f"Extracted tickers: {assets[:10]}{'...' if len(assets) > 10 else ''}")
                else:
                    logger.info("No specific tickers mentioned - will load ALL assets")
            
            # Extract columns and reasoning per signal type
            columns_data = extraction_result.get("columns", {})
            columns_by_signal_type = {}
            reasoning_by_signal_type = {}
            indices_by_signal_type = {}  # Store column indices for precise selection
            all_required_columns = set()
            
            # Check if claude_report is the only signal type (no table data needed)
            has_claude_report = 'claude_report' in selected_signal_types
            table_signal_types = [st for st in selected_signal_types if st != 'claude_report']
            
            for signal_type in selected_signal_types:
                # Skip column extraction for claude_report
                if signal_type == 'claude_report':
                    logger.info("CLAUDE_REPORT: No columns needed - will use full report text")
                    continue
                    
                if signal_type in columns_data:
                    signal_data = columns_data[signal_type]
                    if isinstance(signal_data, dict):
                        cols = signal_data.get('column_names', [])
                        reasoning = signal_data.get('reasoning', '')
                        indices = signal_data.get('column_indices', [])
                        columns_by_signal_type[signal_type] = cols
                        reasoning_by_signal_type[signal_type] = reasoning
                        if indices:
                            indices_by_signal_type[signal_type] = indices
                        all_required_columns.update(cols)
                        logger.info(f"{signal_type.upper()}: Selected {len(cols)} columns")
                        logger.info(f"  Reasoning: {reasoning[:100]}...")
            
            # Only validate columns if we have non-claude_report signal types
            if not columns_by_signal_type and not has_claude_report:
                logger.warning("No columns selected for any signal type")
                return "Could not determine required columns for your query.", {"warning": "no_columns"}
            
            # STAGE 2: Data Fetching (per signal type with its specific columns)
            # Skip data fetching for claude_report signal type
            logger.info("STAGE 2: Fetching data with selected columns for each signal type...")
            
            fetched_data = {}
            total_rows = 0
            
            # Only fetch table data for non-claude_report signal types
            for signal_type in table_signal_types:
                if signal_type not in columns_by_signal_type:
                    continue
                
                required_cols = columns_by_signal_type[signal_type]
                if not required_cols:
                    continue
                
                logger.info(f"Fetching {signal_type} data with {len(required_cols)} columns...")
                
                # Get column indices if available for this signal type
                col_indices = indices_by_signal_type.get(signal_type)
                indices_dict = {signal_type: col_indices} if col_indices else None
                
                signal_data = self.smart_data_fetcher.fetch_data(
                    signal_types=[signal_type],
                    required_columns=required_cols,
                    assets=assets,
                    functions=functions,
                    from_date=from_date,
                    to_date=to_date,
                    limit_rows=MAX_ROWS_TO_INCLUDE,
                    column_indices=indices_dict
                )
                
                if signal_data and signal_type in signal_data:
                    fetched_data[signal_type] = signal_data[signal_type]
                    total_rows += len(signal_data[signal_type])
            
            if not fetched_data:
                if from_date or to_date:
                    logger.warning("No data fetched for explicit date range; skipping automatic expansion.")
                    human_from = from_date or "start"
                    human_to = to_date or "end"
                    logger.warning(f"No data for interval {human_from} to {human_to}")
                    return NO_DATA_MESSAGE, {"warning": "no_data", "from_date": from_date, "to_date": to_date}
                
                logger.warning("No data fetched for the initial date range, trying to expand search...")
                
                # Try expanding the date range for queries like "top N signals" where date is less important
                if any(keyword in user_message.lower() for keyword in ['top', 'best', 'highest', 'lowest']):
                    logger.info("Query seems to be asking for 'top/best' signals - expanding date range")
                    
                    # Expand to last 30 days
                    from datetime import datetime, timedelta
                    expanded_from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    expanded_to_date = datetime.now().strftime('%Y-%m-%d')
                    
                    logger.info(f"Retrying with expanded date range: {expanded_from_date} to {expanded_to_date}")
                    
                    # Retry fetching with expanded date range
                    for signal_type in selected_signal_types:
                        if signal_type not in columns_by_signal_type:
                            continue
                        
                        required_cols = columns_by_signal_type[signal_type]
                        if not required_cols:
                            continue
                        
                        # Get column indices if available for this signal type
                        col_indices = indices_by_signal_type.get(signal_type)
                        indices_dict = {signal_type: col_indices} if col_indices else None
                        
                        signal_data = self.smart_data_fetcher.fetch_data(
                            signal_types=[signal_type],
                            required_columns=required_cols,
                            assets=assets,
                            functions=functions,
                            from_date=expanded_from_date,
                            to_date=expanded_to_date,
                            limit_rows=MAX_ROWS_TO_INCLUDE,
                            column_indices=indices_dict
                        )
                        
                        if signal_data and signal_type in signal_data:
                            fetched_data[signal_type] = signal_data[signal_type]
                            total_rows += len(signal_data[signal_type])
                
                if not fetched_data:
                    logger.warning("No data fetched even with expanded date range")
                    return NO_DATA_MESSAGE, {"warning": "no_data"}
            
            # Format the fetched data for the LLM
            data_context_parts = []
            
            for signal_type, df in fetched_data.items():
                if df.empty:
                    continue
                
                import json as _json_main
                records = df.to_dict('records')
                payload = {
                    "signal_type": signal_type,
                    "record_count": len(records),
                    "columns_selected": columns_by_signal_type[signal_type],
                    "reasoning": reasoning_by_signal_type.get(signal_type, ''),
                    "data": records
                }
                data_context_parts.append(f"\n=== {signal_type.upper()} SIGNALS (JSON) ===")
                data_context_parts.append(_json_main.dumps(payload, indent=2, default=str))
            
            data_context = "\n".join(data_context_parts)
            
            logger.info(f"Fetched {total_rows} total rows from {len(fetched_data)} signal types")
            
            # Build the complete message for GPT
            complete_message = f"""User Query: {user_message}

=== COLUMN SELECTION BY SIGNAL TYPE ==="""
            
            for signal_type in selected_signal_types:
                if signal_type in columns_by_signal_type:
                    complete_message += f"\n\n{signal_type.upper()}:"
                    complete_message += f"\n  Columns: {', '.join(columns_by_signal_type[signal_type])}"
                    complete_message += f"\n  Reasoning: {reasoning_by_signal_type.get(signal_type, '')}"
            
            complete_message += f"\n\n=== DATA CONTEXT ===\n{data_context}"

            if additional_context:
                complete_message += f"\n\n=== ADDITIONAL CONTEXT ===\n{additional_context}"
            
            # Prepare metadata
            metadata = {
                "input_type": "smart_query",
                "selected_signal_types": selected_signal_types,
                "assets": assets,
                "functions": functions,
                "from_date": from_date,
                "to_date": to_date,
                "columns_by_signal_type": columns_by_signal_type,
                "reasoning_by_signal_type": reasoning_by_signal_type,
                "rows_fetched": total_rows,
                "signal_types_with_data": list(fetched_data.keys()),
                "signal_type_reasoning": signal_type_reasoning,
            }
            
            # Add user message to history
            self.history_manager.add_message(
                "user",
                complete_message,
                self._prepare_user_metadata(metadata, user_message)
            )
            
            # Get conversation history for API
            messages = self.history_manager.get_messages_for_api()
            
            # Estimate tokens
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
            
            logger.info(f"Processing smart query with ~{estimated_tokens} tokens")
            
            # Call GPT with the data context
            assistant_message, batch_metadata = self._simple_batch_query(messages, estimated_tokens)
            
            # Update metadata
            metadata["model"] = CLAUDE_MODEL
            metadata["tokens_used"] = batch_metadata["tokens_used"]
            metadata["finish_reason"] = batch_metadata["finish_reason"]
            metadata["batch_processing_used"] = True
            metadata["batch_count"] = batch_metadata.get("batch_count", 1)
            metadata["batch_mode"] = batch_metadata.get("batch_mode", "single")
            
            # Extract full signal tables with all columns
            query_params = {
                'assets': assets,
                'functions': functions,
                'from_date': from_date,
                'to_date': to_date,
                'selected_signal_types': selected_signal_types
            }
            full_signal_tables = self.signal_extractor.extract_full_signal_tables(
                assistant_message, 
                fetched_data,
                query_params
            )
            metadata["full_signal_tables"] = full_signal_tables
            
            # Keep legacy signals_table for compatibility
            signals_df = self.signal_extractor.extract_signals_from_response(
                assistant_message, 
                fetched_data
            )
            metadata["signals_table"] = signals_df
            
            # Add assistant response to history
            self.history_manager.add_message("assistant", assistant_message, metadata)
            
            logger.info(f"Smart query completed with {metadata['tokens_used']['total']} tokens")
            logger.info("="*60)
            
            return assistant_message, metadata
            
        except Exception as e:
            error_message = f"Error processing smart query: {str(e)}"
            logger.error(error_message)
            import traceback
            traceback.print_exc()
            return error_message, {"error": str(e)}
    
    def smart_followup_query(
        self,
        user_message: str,
        selected_signal_types: List[str],
        assets: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        functions: Optional[List[str]] = None,
        additional_context: Optional[str] = None,
        auto_extract_tickers: bool = False,
        signal_type_reasoning: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Process a follow-up query with dynamic, fresh analysis for each query.
        
        NEW APPROACH: Each follow-up query gets fresh signal type/function/column analysis
        based on conversation context (text only, no raw data). This allows the AI to:
        - Reference previous analysis naturally
        - Choose different signal types/functions/columns per query
        - Adapt to changing user interests dynamically
        
        Args:
            user_message: User's follow-up question
            selected_signal_types: Signal types selected by user (checkboxes) - can be overridden by AI
            assets: Optional list of asset/ticker names to filter
            from_date: Start date for data filtering
            to_date: End date for data filtering
            functions: Optional list of function names to filter
            additional_context: Any additional text context to include
            auto_extract_tickers: If True and assets=None, use GPT to extract asset names
            signal_type_reasoning: Optional explanation for pre-selected signal types
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            from chatbot.config import MAX_HISTORY_LENGTH
            
            logger.info("="*60)
            logger.info("SMART FOLLOW-UP QUERY - Dynamic fresh analysis per query")
            logger.info("="*60)
            
            # Get last N exchanges from history (text only, no raw data)
            history_messages = self.history_manager.get_messages_for_api(max_pairs=MAX_HISTORY_LENGTH)
            
            if not history_messages or len(history_messages) < 2:
                logger.warning("No previous context found - treating as new query")
                return self.smart_query(
                    user_message=user_message,
                    selected_signal_types=selected_signal_types,
                    assets=assets,
                    from_date=from_date,
                    to_date=to_date,
                    functions=functions,
                    additional_context=additional_context,
                    auto_extract_tickers=auto_extract_tickers,
                    signal_type_reasoning=signal_type_reasoning
                )
            
            logger.info(f"Retrieved {len(history_messages)} messages from history for context")
            
            # Strip raw data from history to create clean conversation context
            clean_history = self._strip_data_from_history(history_messages)
            
            # Build conversation context (text-only history)
            conversation_context = self._build_text_only_context(clean_history)
            
            logger.info("DYNAMIC ANALYSIS: Treating follow-up as fresh query with conversation context")
            logger.info("This allows AI to freely choose new signal types, functions, columns based on context")
            
            # Treat this as a fresh smart_query but with conversation context
            # This ensures full column selection + data fetching for each query
            
            # Add conversation context to the user message
            enhanced_user_message = f"""CONVERSATION CONTEXT (for reference):
{conversation_context}

CURRENT QUESTION: {user_message}

NOTE: Use the conversation context above to understand what we've discussed, but perform fresh analysis for this specific question. You can choose different signal types, functions, or columns as needed."""
            
            # Call smart_query which will do fresh signal type determination, column selection, and data fetching
            response, metadata = self.smart_query(
                user_message=enhanced_user_message,
                selected_signal_types=selected_signal_types,  # Will be re-determined by AI if needed
                assets=assets,
                from_date=from_date,
                to_date=to_date,
                functions=functions,
                additional_context=additional_context,
                auto_extract_tickers=auto_extract_tickers,
                signal_type_reasoning=signal_type_reasoning
            )
            
            # Mark this as a followup query in metadata
            metadata["input_type"] = "smart_followup"
            metadata["followup_mode"] = "dynamic_fresh"
            metadata["conversation_context_used"] = True
            metadata["history_exchanges_used"] = MAX_HISTORY_LENGTH
            # Override display_prompt to show only the actual user question (not the enhanced context)
            metadata["display_prompt"] = user_message
            
            logger.info(f"Dynamic follow-up query completed with fresh analysis")
            logger.info("="*60)
            
            return response, metadata
            
        except Exception as e:
            error_message = f"Error processing follow-up query: {str(e)}"
            logger.error(error_message)
            import traceback
            traceback.print_exc()
            return error_message, {"error": str(e)}

    def _strip_data_from_history(self, history_messages: List[Dict]) -> List[Dict]:
        """
        Strip raw data payloads from historical messages, keeping only text conversation.
        
        Args:
            history_messages: List of message dictionaries with role and content
            
        Returns:
            List of cleaned message dictionaries
        """
        import re
        
        def _strip_data_payload(text: str) -> str:
            """Remove data sections from message text"""
            if not text:
                return text
            
            patterns = [
                r"===\s*COLUMN SELECTION BY SIGNAL TYPE\s*===[\s\S]*?(?====|$)",
                r"===\s*DATA CONTEXT\s*===[\s\S]*?(?====|$)",
                r"===\s*TRADING DATA[\s\S]*?(?====|$)",
                r"===\s*NEW DATA FETCHED[\s\S]*?(?====|$)",
                r"===\s*NEW COLUMNS ADDED[\s\S]*?(?====|$)",
                r"===\s*PROVIDED DATA\s*===[\s\S]*?(?====|$)",
                r"===\s*ENTRY SIGNALS \(JSON\)[\s\S]*?(?====|$)",
                r"===\s*EXIT SIGNALS \(JSON\)[\s\S]*?(?====|$)",
                r"===\s*PORTFOLIO_TARGET_ACHIEVED SIGNALS \(JSON\)[\s\S]*?(?====|$)",
                r"===\s*BREADTH SIGNALS \(JSON\)[\s\S]*?(?====|$)",
                r"User Query:.*?\n\n",  # Remove "User Query:" prefix
                r"FOLLOW-UP QUESTION:.*?\n\n",  # Remove "FOLLOW-UP QUESTION:" prefix
            ]
            
            cleaned = text
            for pat in patterns:
                cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
            
            # Clean up excessive whitespace
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
            cleaned = cleaned.strip()
            
            return cleaned
        
        clean_messages = []
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Strip data from user messages
            if role == "user":
                content = _strip_data_payload(content)
            
            clean_messages.append({"role": role, "content": content})
        
        return clean_messages
    
    def _build_text_only_context(self, clean_history: List[Dict]) -> str:
        """
        Build a clean conversation context string from history.
        
        Args:
            clean_history: List of cleaned message dictionaries
            
        Returns:
            Formatted conversation context string
        """
        context_parts = []
        
        for msg in clean_history:
            role = msg.get("role", "").upper()
            content = msg.get("content", "").strip()
            
            # Skip system messages
            if role == "SYSTEM":
                continue
            
            # Format as conversation
            if role == "USER":
                context_parts.append(f"Previous Question: {content}")
            elif role == "ASSISTANT":
                context_parts.append(f"Previous Response: {content}")
        
        return "\n\n".join(context_parts)
    
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
        self.history_manager.add_message(
            "user",
            complete_message,
            self._prepare_user_metadata(metadata, user_message)
        )
        
        # Get conversation history for API
        messages = self.history_manager.get_messages_for_api()
        
        try:
            # Estimate tokens
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
            
            # Use simple batch processing for CSV text queries
            logger.info(f"Processing CSV text query with batch method (~{estimated_tokens} tokens)")
            assistant_message, batch_metadata = self._simple_batch_query(messages, estimated_tokens)
            
            # Add response metadata
            metadata["model"] = CLAUDE_MODEL
            metadata["tokens_used"] = batch_metadata["tokens_used"]
            metadata["finish_reason"] = batch_metadata["finish_reason"]
            metadata["batch_processing_used"] = True
            metadata["batch_count"] = batch_metadata.get("batch_count", 1)
            metadata["batch_mode"] = batch_metadata.get("batch_mode", "single")
            
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
    
    def _smart_batch_query(
        self,
        base_messages: List[Dict],
        user_query: str,
        stock_data: Dict,
        estimated_tokens: int
    ) -> Tuple[str, Dict]:
        """
        Smart batch processing that automatically adjusts batch size based on token count.
        
        Features:
        - Single batch if tokens fit within limit
        - Multiple batches if tokens exceed limit
        - Intelligent batch size calculation based on token distribution
        - Sequential processing with rate limiting for multi-batch
        
        Args:
            base_messages: Base conversation messages
            user_query: User's query text
            stock_data: Dictionary of stock DataFrames by ticker
            estimated_tokens: Estimated total tokens for full data
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        import pandas as pd
        
        ticker_list = list(stock_data.keys())
        num_tickers = len(ticker_list)
        
        logger.info(f"Smart batch processing: {num_tickers} tickers, ~{estimated_tokens} tokens")
        
        # Calculate optimal batch strategy
        if estimated_tokens <= MAX_INPUT_TOKENS_PER_CALL:
            # SINGLE BATCH MODE: All data fits in one call
            logger.info("Token count within limit - using SINGLE BATCH mode")
            return self._execute_single_batch(base_messages, user_query, stock_data, estimated_tokens)
        else:
            # MULTI-BATCH MODE: Split intelligently based on token budget
            logger.info(f"Token count exceeds limit ({estimated_tokens} > {MAX_INPUT_TOKENS_PER_CALL}) - using MULTI-BATCH mode")
            
            # Calculate optimal number of batches
            # Reserve some tokens for base messages and overhead
            base_message_tokens = sum(len(str(msg.get('content', ''))) for msg in base_messages[:-1]) // ESTIMATED_CHARS_PER_TOKEN
            available_tokens_per_batch = MAX_INPUT_TOKENS_PER_CALL - base_message_tokens - 500  # 500 token buffer
            
            # Estimate tokens per ticker (average)
            tokens_per_ticker = estimated_tokens / num_tickers if num_tickers > 0 else estimated_tokens
            
            # Calculate tickers per batch
            tickers_per_batch = max(1, int(available_tokens_per_batch / tokens_per_ticker))
            num_batches = max(1, (num_tickers + tickers_per_batch - 1) // tickers_per_batch)  # Ceiling division
            
            # Enforce MAX_SEQUENTIAL_BATCHES limit
            if num_batches > MAX_SEQUENTIAL_BATCHES:
                logger.warning(f"Calculated {num_batches} batches exceeds MAX_SEQUENTIAL_BATCHES ({MAX_SEQUENTIAL_BATCHES})")
                logger.warning(f"Will process only first {MAX_SEQUENTIAL_BATCHES} batches with {MAX_SEQUENTIAL_BATCHES * tickers_per_batch} tickers")
                num_batches = MAX_SEQUENTIAL_BATCHES
            
            logger.info(f"Batch strategy: {num_batches} batches, ~{tickers_per_batch} tickers per batch (limited by MAX_SEQUENTIAL_BATCHES={MAX_SEQUENTIAL_BATCHES})")
            
            return self._execute_multi_batch(base_messages, user_query, stock_data, tickers_per_batch, num_batches)
    
    def _execute_single_batch(
        self,
        base_messages: List[Dict],
        user_query: str,
        stock_data: Dict,
        estimated_tokens: int
    ) -> Tuple[str, Dict]:
        """
        Execute a single batch API call with all data.
        
        Args:
            base_messages: Base conversation messages
            user_query: User's query text
            stock_data: Dictionary of stock DataFrames by ticker
            estimated_tokens: Estimated tokens
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            # Create data context for all tickers
            data_context = self.data_processor.format_data_for_prompt(stock_data)
            
            # Create complete message
            complete_message = f"""User Query: {user_query}

{data_context}"""
            
            # Create messages for API call
            messages = base_messages[:-1]  # All except last message
            messages.append({"role": "user", "content": complete_message})
            
            # Call API
            logger.info(f"Calling API with single batch ({len(stock_data)} tickers)")
            response = self.client.chat.completions.create(
                model=CLAUDE_MODEL,
                messages=messages,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            assistant_message = response.choices[0].message.content
            
            metadata = {
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason,
                "batch_count": 1,
                "batch_mode": "single"
            }
            
            logger.info(f"Single batch completed: {response.usage.total_tokens} tokens")
            
            return assistant_message, metadata
            
        except Exception as e:
            logger.error(f"Error in single batch processing: {e}")
            error_msg = f"Error processing query: {str(e)}"
            metadata = {
                "tokens_used": {"prompt": estimated_tokens, "completion": 0, "total": estimated_tokens},
                "finish_reason": "error",
                "batch_count": 1,
                "batch_mode": "single",
                "error": str(e)
            }
            return error_msg, metadata
    
    def _execute_multi_batch(
        self,
        base_messages: List[Dict],
        user_query: str,
        stock_data: Dict,
        tickers_per_batch: int,
        num_batches: int
    ) -> Tuple[str, Dict]:
        """
        Execute multiple sequential batch API calls with rate limiting.
        
        Args:
            base_messages: Base conversation messages
            user_query: User's query text
            stock_data: Dictionary of stock DataFrames by ticker
            tickers_per_batch: Number of tickers per batch
            num_batches: Total number of batches
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        import pandas as pd
        
        ticker_list = list(stock_data.keys())
        
        # Split tickers into groups
        ticker_groups = []
        for i in range(0, len(ticker_list), tickers_per_batch):
            ticker_groups.append(ticker_list[i:i + tickers_per_batch])
        
        # Enforce MAX_SEQUENTIAL_BATCHES limit
        if len(ticker_groups) > num_batches:
            logger.warning(f"Limiting to {num_batches} batches (from {len(ticker_groups)} calculated batches)")
            ticker_groups = ticker_groups[:num_batches]
            remaining_tickers = len(ticker_list) - (num_batches * tickers_per_batch)
            if remaining_tickers > 0:
                logger.warning(f"Excluding {remaining_tickers} tickers due to MAX_SEQUENTIAL_BATCHES limit")
        
        logger.info(f"Processing {len(ticker_groups)} batches: {[len(g) for g in ticker_groups]} tickers each")
        
        # Execute SEQUENTIAL batch calls with delays
        results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
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
                    model=CLAUDE_MODEL,
                    messages=group_messages,
                    max_completion_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                
                result_text = response.choices[0].message.content
                
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
                
                logger.info(f"Batch {group_idx + 1} completed: {response.usage.total_tokens} tokens")
                results.append({
                    'group_idx': group_idx,
                    'tickers': group_tickers,
                    'response': result_text,
                    'tokens': response.usage.total_tokens
                })
                
            except Exception as e:
                logger.error(f"Error in batch {group_idx + 1}: {e}")
                results.append({
                    'group_idx': group_idx,
                    'tickers': group_tickers,
                    'response': f"Error analyzing {', '.join(group_tickers)}: {str(e)}",
                    'tokens': 0
                })
        
        # Aggregate responses with final synthesis API call
        total_tokens = total_prompt_tokens + total_completion_tokens
        logger.info(f"Multi-batch processing complete: {len(results)} batches, {total_tokens} total tokens")
        
        # Combine all batch responses for synthesis
        batch_responses_text = ""
        for result in results:
            if result['response']:
                batch_responses_text += f"\n### Batch {result['group_idx'] + 1} Results ({', '.join(result['tickers'][:5])}{'...' if len(result['tickers']) > 5 else ''}):\n"
                batch_responses_text += f"{result['response']}\n\n"
        
        # FINAL AGGREGATION: Use one more API call to synthesize all batch results
        logger.info("Making final aggregation API call to synthesize multi-batch results...")
        
        try:
            aggregation_prompt = f"""You are analyzing data from {len(ticker_list)} assets that were processed in {len(results)} batches. 

**Original User Query:** {user_query}

**Batch Results:**
{batch_responses_text}

**Your Task:**
Synthesize the above batch results into a single, coherent, comprehensive answer to the user's original query. 

Requirements:
1. Combine all information into a unified response (don't mention batches)
2. Remove duplicate information
3. Organize the data logically (by asset, function, date, or relevance)
4. Use proper Markdown formatting with clear sections
5. Provide summary statistics if relevant
6. Answer the user's original question directly and completely

Create a professional, well-structured response that reads as one cohesive analysis."""

            synthesis_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": aggregation_prompt}
            ]
            
            # Add delay before final API call
            logger.info(f"Waiting {BATCH_DELAY_SECONDS}s before aggregation call...")
            time.sleep(BATCH_DELAY_SECONDS)
            
            # Make aggregation API call
            logger.info("Calling API for final aggregation")
            aggregation_response = self.client.chat.completions.create(
                model=CLAUDE_MODEL,
                messages=synthesis_messages,
                max_completion_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            final_response = aggregation_response.choices[0].message.content
            
            # Update token counts with aggregation call
            total_prompt_tokens += aggregation_response.usage.prompt_tokens
            total_completion_tokens += aggregation_response.usage.completion_tokens
            total_tokens = total_prompt_tokens + total_completion_tokens
            
            logger.info(f"Aggregation complete: {aggregation_response.usage.total_tokens} tokens")
            logger.info(f"Total tokens (including aggregation): {total_tokens}")
            
        except Exception as e:
            logger.error(f"Error in aggregation API call: {e}")
            # Fallback to manual combination if aggregation fails
            final_response = f"""# Comprehensive Analysis ({len(ticker_list)} assets analyzed)

"""
            for result in results:
                if result['response']:
                    final_response += f"\n## Batch {result['group_idx'] + 1}: {', '.join(result['tickers'][:5])}"
                if result['response']:
                    final_response += f"\n## Batch {result['group_idx'] + 1}: {', '.join(result['tickers'][:5])}"
                    if len(result['tickers']) > 5:
                        final_response += f" ... and {len(result['tickers']) - 5} more"
                    final_response += f"\n\n{result['response']}\n\n---\n"
        
        metadata = {
            "tokens_used": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "total": total_tokens
            },
            "finish_reason": "batch_aggregation_with_synthesis",
            "batch_count": len(results) + 1,  # +1 for aggregation call
            "batch_mode": "multi"
        }
        
        return final_response, metadata
    
    def _simple_batch_query(
        self,
        messages: List[Dict],
        estimated_tokens: int
    ) -> Tuple[str, Dict]:
        """
        Simple batch processing for non-ticker queries (breadth, CSV text, etc.).
        Automatically uses single API call for small data, no splitting needed.
        
        Args:
            messages: Complete conversation messages
            estimated_tokens: Estimated total tokens
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
            # For non-ticker queries, we just make a single API call
            # These queries typically don't have the massive data volume of ticker-based queries
            logger.info(f"Processing non-ticker query with ~{estimated_tokens} tokens")
            
            # Trim history if needed to stay under token limit
            trimmed_messages = list(messages)
            trimmed_count = 0
            
            while estimated_tokens > MAX_INPUT_TOKENS_PER_CALL and len(trimmed_messages) > MIN_HISTORY_MESSAGES:
                if len(trimmed_messages) > 2:
                    removed_msg = trimmed_messages.pop(1)
                    trimmed_count += 1
                    logger.warning(f"Trimmed old message (role: {removed_msg.get('role', 'unknown')})")
                    
                    total_chars = sum(len(str(msg.get('content', ''))) for msg in trimmed_messages)
                    estimated_tokens = total_chars // ESTIMATED_CHARS_PER_TOKEN
                else:
                    break
            
            if trimmed_count > 0:
                logger.warning(f"Trimmed {trimmed_count} old messages to stay under token limit")
            
            # Make single API call with Claude
            logger.info(f"Calling {CLAUDE_MODEL} with {len(trimmed_messages)} messages, ~{estimated_tokens} tokens")
            
            # Convert messages to Claude format
            claude_messages = self._convert_to_claude_format(trimmed_messages)
            
            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                temperature=CLAUDE_TEMPERATURE,
                system=claude_messages["system"],
                messages=claude_messages["messages"]
            )
            
            assistant_message = response.content[0].text
            
            metadata = {
                "tokens_used": {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens
                },
                "finish_reason": response.stop_reason,
                "batch_count": 1,
                "batch_mode": "single",
                "model": CLAUDE_MODEL
            }
            
            logger.info(f"Simple batch completed: {metadata['tokens_used']['total']} tokens")
            
            return assistant_message, metadata
            
        except Exception as e:
            logger.error(f"Error in simple batch processing: {e}")
            error_msg = f"Error processing query: {str(e)}"
            metadata = {
                "tokens_used": {"prompt": estimated_tokens, "completion": 0, "total": estimated_tokens},
                "finish_reason": "error",
                "batch_count": 1,
                "batch_mode": "single",
                "error": str(e)
            }
            return error_msg, metadata
    
    def _batch_followup_query(
        self,
        base_messages: List[Dict],
        user_query: str,
        signal_data: Dict,
        estimated_tokens: int
    ) -> Tuple[str, Dict]:
        """
        Batch processing for follow-up queries with large signal data.
        Splits data by signal type or rows to fit within token limits.
        
        Args:
            base_messages: Base conversation messages
            user_query: User's query text
            signal_data: Dictionary of DataFrames by signal type (entry/exit/portfolio_target_achieved/breadth)
            estimated_tokens: Estimated total tokens
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        import pandas as pd
        import time
        import json as _json

        try:
            logger.info(f"⚡ Batch processing follow-up query: {len(signal_data)} signal types, ~{estimated_tokens} tokens")
            
            # Calculate base message tokens
            base_message_tokens = sum(len(str(msg.get('content', ''))) for msg in base_messages) // ESTIMATED_CHARS_PER_TOKEN
            available_tokens_per_batch = MAX_INPUT_TOKENS_PER_CALL - base_message_tokens - 500  # 500 token buffer
            
            # Strategy: Split by signal type first, then by rows if needed
            batches = []
            
            for signal_type, df in signal_data.items():
                if df.empty:
                    continue
                
                # Estimate tokens for this signal type (JSON records)
                try:
                    data_str = _json.dumps(df.to_dict('records'), default=str)
                except Exception:
                    data_str = df.to_string(index=False)
                signal_tokens = len(data_str) // ESTIMATED_CHARS_PER_TOKEN
                
                if signal_tokens <= available_tokens_per_batch:
                    # Fits in one batch
                    batches.append({
                        'signal_types': [signal_type],
                        'data': {signal_type: df},
                        'estimated_tokens': signal_tokens
                    })
                else:
                    # Need to split rows
                    logger.info(f"Signal type {signal_type} too large ({signal_tokens} tokens), splitting rows")
                    
                    # Estimate rows per batch
                    tokens_per_row = signal_tokens / len(df) if len(df) > 0 else signal_tokens
                    rows_per_batch = max(1, int(available_tokens_per_batch / tokens_per_row))
                    
                    # Split into chunks
                    for i in range(0, len(df), rows_per_batch):
                        chunk_df = df.iloc[i:i+rows_per_batch]
                        try:
                            chunk_str = _json.dumps(chunk_df.to_dict('records'), default=str)
                        except Exception:
                            chunk_str = chunk_df.to_string(index=False)
                        chunk_tokens = len(chunk_str) // ESTIMATED_CHARS_PER_TOKEN
                        
                        batches.append({
                            'signal_types': [f"{signal_type}_part{i//rows_per_batch + 1}"],
                            'data': {signal_type: chunk_df},
                            'estimated_tokens': chunk_tokens
                        })
            
            num_batches = len(batches)
            logger.info(f"Split into {num_batches} batches for processing")
            
            if num_batches == 1:
                # Single batch - process directly
                batch = batches[0]
                data_context = "\n=== BATCH DATA (JSON) ===\n"
                for sig_type, df in batch['data'].items():
                    records = df.to_dict('records')
                    payload = {"signal_type": sig_type, "record_count": len(records), "data": records}
                    data_context += _json.dumps(payload, indent=2, default=str) + "\n"
                
                complete_message = f"""User Query: {user_query}

{data_context}"""
                
                messages = base_messages + [{"role": "user", "content": complete_message}]
                
                # Convert to Claude format and call Claude API
                claude_messages = self._convert_to_claude_format(messages)
                
                response = self.claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    temperature=CLAUDE_TEMPERATURE,
                    system=claude_messages["system"],
                    messages=claude_messages["messages"]
                )
                
                assistant_message = response.content[0].text
                
                metadata = {
                    "tokens_used": {
                        "prompt": response.usage.input_tokens,
                        "completion": response.usage.output_tokens,
                        "total": response.usage.input_tokens + response.usage.output_tokens
                    },
                    "finish_reason": response.stop_reason,
                    "batch_count": 1,
                    "batch_mode": "single",
                    "model": CLAUDE_MODEL
                }
                
                logger.info(f"Single batch completed: {metadata['tokens_used']['total']} tokens")
                return assistant_message, metadata
                
            else:
                # Multiple batches - process sequentially with synthesis
                logger.info(f"Processing {num_batches} batches with synthesis")
                
                batch_responses = []
                total_prompt_tokens = 0
                total_completion_tokens = 0
                
                for idx, batch in enumerate(batches, 1):
                    logger.info(f"Processing batch {idx}/{num_batches}")
                    
                    # Create data context for this batch
                    data_context = "\n=== BATCH DATA (JSON) ===\n"
                    for sig_type, df in batch['data'].items():
                        records = df.to_dict('records')
                        payload = {"signal_type": sig_type, "record_count": len(records), "data": records}
                        data_context += _json.dumps(payload, indent=2, default=str) + "\n"
                    
                    batch_message = f"""User Query: {user_query}

[BATCH {idx}/{num_batches}]

{data_context}

Please analyze this batch. Your response will be combined with other batches later."""
                    
                    messages = base_messages + [{"role": "user", "content": batch_message}]
                    
                    response = self.client.chat.completions.create(
                        model=CLAUDE_MODEL,
                        messages=messages,
                        max_completion_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE
                    )
                    
                    batch_response = response.choices[0].message.content
                    batch_responses.append(f"[Batch {idx}]\n{batch_response}")
                    
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                    
                    logger.info(f"Batch {idx} completed: {response.usage.total_tokens} tokens")
                    
                    # Rate limiting between batches
                    if idx < num_batches:
                        time.sleep(1)
                
                # Synthesize all batch responses
                logger.info("Synthesizing all batch responses into final answer")
                
                synthesis_prompt = f"""Original Query: {user_query}

I've analyzed the data in {num_batches} batches. Here are the individual batch analyses:

{chr(10).join(batch_responses)}

Please synthesize these batch analyses into a single, coherent response that:
1. Combines insights from all batches
2. Provides a unified answer to the original query
3. Maintains consistency across all data
4. Presents results in a clear, organized format

Final synthesized response:"""
                
                synthesis_messages = base_messages + [{"role": "user", "content": synthesis_prompt}]
                
                synthesis_response = self.client.chat.completions.create(
                    model=CLAUDE_MODEL,
                    messages=synthesis_messages,
                    max_completion_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                
                assistant_message = synthesis_response.choices[0].message.content
                
                total_prompt_tokens += synthesis_response.usage.prompt_tokens
                total_completion_tokens += synthesis_response.usage.completion_tokens
                
                metadata = {
                    "tokens_used": {
                        "prompt": total_prompt_tokens,
                        "completion": total_completion_tokens,
                        "total": total_prompt_tokens + total_completion_tokens
                    },
                    "finish_reason": f"synthesis_from_{num_batches}_batches",
                    "batch_count": num_batches + 1,  # +1 for synthesis
                    "batch_mode": "multi"
                }
                
                logger.info(f"Multi-batch synthesis completed: {num_batches} batches, {metadata['tokens_used']['total']} total tokens")
                return assistant_message, metadata
                
        except Exception as e:
            logger.error(f"Error in batch follow-up processing: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"Error processing batch query: {str(e)}"
            metadata = {
                "tokens_used": {"prompt": estimated_tokens, "completion": 0, "total": estimated_tokens},
                "finish_reason": "error",
                "batch_count": 1,
                "batch_mode": "single",
                "error": str(e)
            }
            return error_msg, metadata
    
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

