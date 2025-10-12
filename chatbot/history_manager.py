"""
History manager for maintaining conversation context.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from uuid import uuid4

from .config import HISTORY_DIR, MAX_HISTORY_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages conversation history and context for chatbot sessions."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize history manager.
        
        Args:
            session_id: Unique session identifier. If None, creates new session.
        """
        self.session_id = session_id or str(uuid4())
        self.history_file = HISTORY_DIR / f"{self.session_id}.json"
        self.conversation_history: List[Dict[str, str]] = []
        self.metadata: Dict = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Load existing history if available
        if self.history_file.exists():
            self.load_history()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata for the message
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
        
        self.conversation_history.append(message)
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > MAX_HISTORY_LENGTH * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-(MAX_HISTORY_LENGTH * 2):]
        
        # Auto-save after each message
        self.save_history()
        
        logger.info(f"Added {role} message to session {self.session_id}")
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for OpenAI API.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
    
    def get_full_history(self) -> List[Dict[str, str]]:
        """
        Get complete conversation history with all metadata.
        
        Returns:
            List of all message dictionaries
        """
        return self.conversation_history.copy()
    
    def save_history(self):
        """Save conversation history to file."""
        try:
            data = {
                "metadata": self.metadata,
                "conversation": self.conversation_history
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved history for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def load_history(self):
        """Load conversation history from file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metadata = data.get("metadata", {})
            self.conversation_history = data.get("conversation", [])
            
            logger.info(f"Loaded history for session {self.session_id} with {len(self.conversation_history)} messages")
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            self.conversation_history = []
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.save_history()
        logger.info(f"Cleared history for session {self.session_id}")
    
    def delete_session(self):
        """Delete session history file."""
        try:
            if self.history_file.exists():
                self.history_file.unlink()
                logger.info(f"Deleted session {self.session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    
    def get_session_summary(self) -> Dict:
        """
        Get summary of current session.
        
        Returns:
            Dictionary with session metadata and statistics
        """
        return {
            "session_id": self.session_id,
            "created_at": self.metadata.get("created_at"),
            "last_updated": self.metadata.get("last_updated"),
            "message_count": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m["role"] == "user"]),
            "assistant_messages": len([m for m in self.conversation_history if m["role"] == "assistant"])
        }
    
    @staticmethod
    def list_all_sessions() -> List[str]:
        """
        List all available session IDs.
        
        Returns:
            List of session ID strings
        """
        try:
            session_files = list(HISTORY_DIR.glob("*.json"))
            return [f.stem for f in session_files]
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
    
    @staticmethod
    def load_session(session_id: str) -> Optional['HistoryManager']:
        """
        Load an existing session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            HistoryManager instance or None if session doesn't exist
        """
        session_file = HISTORY_DIR / f"{session_id}.json"
        if session_file.exists():
            return HistoryManager(session_id=session_id)
        return None

