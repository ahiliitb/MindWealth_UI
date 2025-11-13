"""
Signal type selector that uses GPT to determine which data categories are needed.
"""

import json
import logging
from typing import List, Optional, Tuple

from openai import OpenAI

from .config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ALLOWED_SIGNAL_TYPES = ["entry", "exit", "target", "breadth"]
DEFAULT_SIGNAL_TYPES = ["entry", "exit", "target"]

SIGNAL_TYPE_DESCRIPTIONS = {
    "entry": (
        "Entry Signals",
        "Fresh trading ideas that have triggered but are still open (no exit yet). "
        "Useful when the user wants current opportunities or new setups."
    ),
    "exit": (
        "Exit Signals",
        "Trades that have completed with recorded exits. "
        "Relevant for reviewing performance, closed trades, or outcomes."
    ),
    "target": (
        "Target Achieved",
        "Signals where price targets have been hit or high-conviction targets are outlined. "
        "Helpful when the request mentions profit-taking, targets, or milestones."
    ),
    "breadth": (
        "Market Breadth",
        "Market-wide sentiment metrics (e.g., bull/bear breadth). "
        "Apply when the user asks about overall market health or breadth indicators."
    ),
}


class SignalTypeSelector:
    """Determine which signal types are needed for a user prompt."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided for SignalTypeSelector.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def select_signal_types(self, user_query: str) -> Tuple[List[str], str]:
        """
        Analyze the user query and decide which signal types to include.

        Returns:
            Tuple[List[str], str]: (selected signal type identifiers, reasoning string)
        """
        if not user_query or not user_query.strip():
            return (
                DEFAULT_SIGNAL_TYPES.copy(),
                "No specific request detected; using default entry/exit/target signals."
            )

        options_text = "\n".join(
            [
                f"- {name} ({title}): {description}"
                for name, (title, description) in SIGNAL_TYPE_DESCRIPTIONS.items()
            ]
        )

        prompt = f"""You are an AI assistant that selects which trading signal categories are needed to answer a question.

Available signal categories:
{options_text}

Selection rules:
1. Always choose at least one category from the list.
2. Choose only the categories that are genuinely required for the user's request.
3. If the request is broad or unclear, default to ["entry", "exit", "target"].
4. Select "breadth" ONLY if the user asks about overall market health, sentiment, or breadth indicators.
5. Preserve the order: entry → exit → target → breadth.

User query: \"\"\"{user_query}\"\"\"

Respond strictly as a JSON object with this schema:
{{
  "signal_types": ["entry", "exit"],
  "reasoning": "Short explanation of why these categories are needed."
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze trading questions and decide which signal data types are needed."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = json.loads(content)
            raw_selection = data.get("signal_types", [])
            reasoning = data.get("reasoning", "").strip()

            if isinstance(raw_selection, str):
                raw_selection = [raw_selection]

            selection_set = {item.lower() for item in raw_selection if isinstance(item, str)}

            ordered_selection = [
                signal_type for signal_type in ALLOWED_SIGNAL_TYPES if signal_type in selection_set
            ]

            if not ordered_selection:
                logger.info("Signal type selector returned empty or invalid selection; using defaults.")
                ordered_selection = DEFAULT_SIGNAL_TYPES.copy()
                if not reasoning:
                    reasoning = "Defaulted to entry/exit/target due to unclear selection."

            logger.info(f"Signal type selection: {ordered_selection} | Reason: {reasoning}")
            return ordered_selection, reasoning or "Auto-selected signal types based on the query."

        except Exception as exc:
            logger.error(f"Failed to select signal types via OpenAI: {exc}")
            return (
                DEFAULT_SIGNAL_TYPES.copy(),
                "Fallback to default entry/exit/target due to selection error."
            )

