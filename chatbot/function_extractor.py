"""
Function extractor using GPT-5.2 to identify trading functions from user prompts.
"""

import logging
import json
from typing import List, Optional
from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prompt for extracting function names from user query
FUNCTION_EXTRACTION_PROMPT = """You are a function name extractor for a trading analysis system.

Your task: Analyze the user's query and identify which trading analysis FUNCTIONS they are asking about.

Available Functions (EXACT names):
1. ALTITUDE ALPHA
2. BAND MATRIX
3. BASELINEDIVERGENCE
4. FRACTAL TRACK
5. OSCILLATOR DELTA
6. PULSEGAUGE
7. SIGMASHELL
8. TRENDPULSE

Instructions:
- Extract ONLY the function names mentioned in the user's query
- Return EXACT function names as they appear in the list above
- If user mentions variations (e.g., "trendpulse", "Fractal Track"), match to the exact name
- If NO specific functions are mentioned, return an empty list []
- Return response as valid JSON array: ["FUNCTION1", "FUNCTION2", ...]

Examples:

User: "What TRENDPULSE signals exist for AAPL?"
Response: ["TRENDPULSE"]

User: "Compare TRENDPULSE and FRACTAL TRACK signals"
Response: ["TRENDPULSE", "FRACTAL TRACK"]

User: "Show me all signals for AAPL"
Response: []

User: "What are the baseline divergence signals?"
Response: ["BASELINEDIVERGENCE"]

User: "Analyze AAPL stock performance"
Response: []

User: "Show TRENDPULSE, BAND MATRIX and SIGMASHELL signals"
Response: ["TRENDPULSE", "BAND MATRIX", "SIGMASHELL"]

Now extract from the user's query below. Respond ONLY with a JSON array, nothing else.
"""


class FunctionExtractor:
    """Extracts trading function names from user prompts using GPT-5.2."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize function extractor.
        
        Args:
            api_key: Optional OpenAI API key (uses env var if not provided)
        """
        self.api_key = api_key or OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Available function names for validation
        self.available_functions = [
            "ALTITUDE ALPHA",
            "BAND MATRIX",
            "BASELINEDIVERGENCE",
            "FRACTAL TRACK",
            "OSCILLATOR DELTA",
            "PULSEGAUGE",
            "SIGMASHELL",
            "TRENDPULSE"
        ]
    
    def extract_functions(self, user_query: str) -> List[str]:
        """
        Extract function names from user query using GPT-5.2.
        
        Args:
            user_query: User's question/prompt
            
        Returns:
            List of function names mentioned in the query
        """
        try:
            # Build the prompt
            messages = [
                {"role": "system", "content": FUNCTION_EXTRACTION_PROMPT},
                {"role": "user", "content": user_query}
            ]
            
            # Call GPT-5.2 for extraction
            logger.info(f"Extracting functions from user query using {OPENAI_MODEL}...")
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,  # Use GPT-5.2 for intelligent extraction
                messages=messages,
                max_completion_tokens=100,  # Short response for function names
                temperature=TEMPERATURE  # Use configured temperature
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Function extraction response: {response_text}")
            
            # Parse JSON response
            try:
                functions = json.loads(response_text)
                
                if not isinstance(functions, list):
                    logger.warning(f"Response is not a list: {response_text}")
                    return []
                
                # Validate extracted functions
                valid_functions = []
                for func in functions:
                    if func in self.available_functions:
                        valid_functions.append(func)
                    else:
                        logger.warning(f"Invalid function extracted: {func}")
                
                logger.info(f"Extracted {len(valid_functions)} valid functions: {valid_functions}")
                return valid_functions
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {response_text}")
                # Try to extract functions manually as fallback
                return self._fallback_extraction(user_query)
        
        except Exception as e:
            logger.error(f"Error extracting functions: {e}")
            return []
    
    def _fallback_extraction(self, user_query: str) -> List[str]:
        """
        Fallback method using simple string matching.
        
        Args:
            user_query: User's question
            
        Returns:
            List of function names found in query
        """
        logger.info("Using fallback extraction method")
        
        query_upper = user_query.upper()
        found_functions = []
        
        for func in self.available_functions:
            # Check for exact match or close variations
            if func.upper() in query_upper:
                found_functions.append(func)
            # Check for variations without spaces
            elif func.upper().replace(" ", "") in query_upper.replace(" ", ""):
                found_functions.append(func)
        
        return found_functions
    
    def get_available_functions(self) -> List[str]:
        """Get list of available function names."""
        return self.available_functions.copy()

