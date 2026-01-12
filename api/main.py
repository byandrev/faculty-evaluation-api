"""
Main API module defining the FastAPI application and its endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.nlp import hate_analyzer, sentiment_analyzer
from api.models.comment import Comment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint returning a simple greeting message.
    """
    return {"message": "Hello World"}


@app.post("/comments/")
async def analyze_comment(comment: Comment):
    """
    Endpoint to analyse and store a comment.
    """

    sentiment = sentiment_analyzer.predict(comment.content)
    hate = hate_analyzer.predict(comment.content)

    print(sentiment)
    print(hate)

    return {
      "comment": comment.content,
      "sentiment": sentiment,
      "hate": hate,
      "status": "Comment created successfully"
    }
