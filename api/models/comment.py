"""
API model for comments.
"""

from pydantic import BaseModel

class Comment(BaseModel):
    """Model representing a comment."""
    content: str
