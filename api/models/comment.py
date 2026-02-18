"""
API model for comments.
"""

from typing import List

from pydantic import BaseModel


class Comment(BaseModel):
    """Model representing a comment."""

    content: str


class CommentList(BaseModel):
    """Model representing a list of comments for summarization."""

    comments: List[str]
