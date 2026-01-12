"""
NLP models and response schemas for analysis.
"""

from pydantic import BaseModel

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""

    label: str
    confidence: float

class AnalysisResponse(BaseModel):
    """Response model for overall analysis."""

    sentiment: SentimentResponse
    is_sensitive: bool
    alert_level: str
