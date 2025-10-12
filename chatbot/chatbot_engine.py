"""
Main chatbot engine for processing queries and generating responses.
"""

import logging
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    SYSTEM_PROMPT
)
from .data_processor import DataProcessor
from .history_manager import HistoryManager
from .function_extractor import FunctionExtractor

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
        additional_context: Optional[str] = None,
        dedup_columns: Optional[List[str]] = None,
        auto_extract_functions: bool = True
    ) -> Tuple[str, Dict]:
        """
        Process a user query with optional data context.
        
        Args:
            user_message: User's question or request
            tickers: List of ticker/asset symbols to include in context
            from_date: Start date for data filtering (YYYY-MM-DD)
            to_date: End date for data filtering (YYYY-MM-DD)
            functions: List of function names to filter (None = auto-extract or all functions)
            additional_context: Any additional text context to include
            dedup_columns: Columns to use for deduplication (None = use config default)
            auto_extract_functions: If True and functions=None, use GPT-4o-mini to extract
            
        Note: Automatically loads data from BOTH signal and target folders
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        try:
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
                "from_date": from_date,
                "to_date": to_date,
                "functions": functions or [],
                "functions_auto_extracted": extracted_functions or []
            }
            
            # Check if these exact parameters were used before in this conversation
            data_already_in_context = self._check_if_data_in_history(
                tickers, from_date, to_date, functions
            )
            
            if tickers and not data_already_in_context:
                # Load and format stock data (from both signal and target)
                stock_data = self.data_processor.load_stock_data(
                    tickers, from_date, to_date, dedup_columns, functions
                )
                
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
            
            # Call GPT-4o
            logger.info(f"Calling GPT-4o with {len(messages)} messages in context")
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
        functions: Optional[List[str]]
    ) -> bool:
        """
        Check if the same data parameters were used in a previous query.
        
        Args:
            tickers: List of ticker symbols
            from_date: Start date
            to_date: End date
            functions: List of function names
            
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
            
            # If we got here, parameters match!
            logger.info(
                f"Found matching data in history: tickers={tickers}, "
                f"from={from_date}, to={to_date}, functions={functions}"
            )
            return True
        
        return False

