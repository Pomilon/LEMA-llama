"""
Manage conversation state and history.
"""

from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class Message:
    """Single message in conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    metadata: Dict = field(default_factory=dict)

class ConversationManager:
    """Manage conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum number of turns to keep
        """
        self.max_history = max_history
        self.messages: List[Message] = []
    
    def add_user_message(self, content: str):
        """Add user message to history."""
        self.messages.append(Message(role='user', content=content))
        self._trim_history()
    
    def add_assistant_message(self, content: str, **metadata):
        """Add assistant message to history."""
        self.messages.append(Message(
            role='assistant', 
            content=content,
            metadata=metadata
        ))
        self._trim_history()
    
    def get_context(self, include_current: bool = True) -> str:
        """
        Build context string from conversation history.
        
        Args:
            include_current: Whether to include the most recent exchange
        
        Returns:
            Formatted conversation context
        """
        # For now, we just use the immediate question
        # You could extend this to include conversation history
        if not self.messages:
            return ""
        
        # Get last user message
        for msg in reversed(self.messages):
            if msg.role == 'user':
                return msg.content
        
        return ""
    
    def clear(self):
        """Clear conversation history."""
        self.messages.clear()
    
    def _trim_history(self):
        """Keep only max_history recent messages."""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
