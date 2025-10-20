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
    
    def __init__(self, session_id: Optional[str] = None, session_title: Optional[str] = None):
        """
        Initialize history manager.
        
        Args:
            session_id: Unique session identifier. If None, creates new session.
            session_title: Optional title for the session.
        """
        self.session_id = session_id or str(uuid4())
        self.history_file = HISTORY_DIR / f"{self.session_id}.json"
        self.conversation_history: List[Dict[str, str]] = []
        self.metadata: Dict = {
            "session_id": self.session_id,
            "title": session_title or "New Chat",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": 0
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
        self.metadata["message_count"] = len(self.conversation_history)
        
        # Auto-generate title from first user message if title is still "New Chat"
        if role == "user" and self.metadata.get("title") == "New Chat":
            self.update_session_title_from_message(content)
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > MAX_HISTORY_LENGTH * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-(MAX_HISTORY_LENGTH * 2):]
        
        # Auto-save after each message
        self.save_history()
        
        logger.info(f"Added {role} message to session {self.session_id}")
    
    def get_messages_for_api(self, max_pairs: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for OpenAI API.
        
        Args:
            max_pairs: Optional maximum number of user-assistant message pairs to include.
                      If specified, returns the last N pairs plus the system message.
                      If None, returns all messages.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
        
        if max_pairs is None:
            return messages
        
        # Separate system message from conversation
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # Get last N pairs (each pair is user + assistant = 2 messages)
        num_messages_to_keep = max_pairs * 2
        last_messages = conversation_messages[-num_messages_to_keep:] if len(conversation_messages) > num_messages_to_keep else conversation_messages
        
        # Combine: system message + last N pairs
        return system_messages + last_messages
    
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
            "title": self.metadata.get("title", "New Chat"),
            "created_at": self.metadata.get("created_at"),
            "last_updated": self.metadata.get("last_updated"),
            "message_count": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m["role"] == "user"]),
            "assistant_messages": len([m for m in self.conversation_history if m["role"] == "assistant"])
        }
    
    def update_session_title(self, new_title: str):
        """
        Update the session title.
        
        Args:
            new_title: New title for the session
        """
        self.metadata["title"] = new_title
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.save_history()
        logger.info(f"Updated title for session {self.session_id}: {new_title}")
    
    def update_session_title_from_message(self, message: str, max_length: int = 50):
        """
        Auto-generate and update session title from a message.
        
        Args:
            message: Message content to generate title from
            max_length: Maximum length for the title
        """
        if not message:
            return
        
        # Clean up the message
        title = message.strip()
        
        # Truncate if too long
        if len(title) > max_length:
            title = title[:max_length].rsplit(' ', 1)[0] + "..."
        
        self.update_session_title(title)
    
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

