"""
NLP module for sentiment and hate speech analysis using pysentimiento.
"""

from pysentimiento import create_analyzer

sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
hate_analyzer = create_analyzer(task="hate_speech", lang="es")
