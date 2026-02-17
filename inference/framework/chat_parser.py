"""
Parse and validate LEMA custom chat format.
"""

import re
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class LemaResponse:
    """Parsed LEMA response."""
    answer: str
    explanation: str
    confidence: str
    raw_text: str
    is_valid: bool

class ChatParser:
    """Parse LEMA custom chat format."""
    
    # Regex patterns for extracting fields
    LEMA_REPLY_PATTERN = r'\\[LEMA_REPLY\\](.*?)\\[/LEMA_REPLY\\]'
    ANSWER_PATTERN = r'Answer:\s*(.+?)(?=\n|Explanation:|Confidence:|$)'
    EXPLANATION_PATTERN = r'Explanation:\s*(.+?)(?=\n|Confidence:|$)'
    CONFIDENCE_PATTERN = r'Confidence:\s*(High|Medium|Low)'
    
    @classmethod
    def parse_response(cls, text: str) -> LemaResponse:
        """
        Parse a LEMA-formatted response.
        
        Args:
            text: Generated text that should contain [LEMA_REPLY] block
        
        Returns:
            LemaResponse with parsed fields and validation status
        """
        # Extract LEMA_REPLY block
        reply_match = re.search(cls.LEMA_REPLY_PATTERN, text, re.DOTALL)
        
        if not reply_match:
            return LemaResponse(
                answer="",
                explanation="",
                confidence="",
                raw_text=text,
                is_valid=False
            )
        
        reply_content = reply_match.group(1)
        
        # Extract fields
        answer_match = re.search(cls.ANSWER_PATTERN, reply_content, re.DOTALL)
        explanation_match = re.search(cls.EXPLANATION_PATTERN, reply_content, re.DOTALL)
        confidence_match = re.search(cls.CONFIDENCE_PATTERN, reply_content)
        
        answer = answer_match.group(1).strip() if answer_match else ""
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        confidence = confidence_match.group(1).strip() if confidence_match else ""
        
        is_valid = bool(answer and explanation and confidence)
        
        return LemaResponse(
            answer=answer,
            explanation=explanation,
            confidence=confidence,
            raw_text=text,
            is_valid=is_valid
        )
    
    @classmethod
    def format_prompt(cls, user_message: str) -> str:
        """
        Format a user message into the LEMA chat template.
        
        Args:
            user_message: User's question/input
        
        Returns:
            Properly formatted prompt for the model
        """
        return f"""<|system|>
You are a precise assistant trained using LEMA.

<|user|>
{user_message}

<|assistant|>
[LEMA_REPLY]
Answer:"""
