"""
NLP module for sentiment and hate speech analysis using pysentimiento.
"""

from pysentimiento import create_analyzer
from transformers import pipeline

sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
hate_analyzer = create_analyzer(task="hate_speech", lang="es")

danger_analyzer = pipeline(
    "text-classification", model="byandrev/evd", tokenizer="byandrev/evd"
)

danger_analyzer_v2 = pipeline(
    "text-classification", model="byandrev/evd2", tokenizer="byandrev/evd2"
)
