"""
Main API module defining the FastAPI application and its endpoints.
"""

import csv
from io import StringIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.models.comment import Comment
from api.nlp import danger_analyzer, hate_analyzer, sentiment_analyzer

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
    danger = danger_analyzer.predict(comment.content)[0]

    return {
        "comment": comment.content,
        "sentiment": sentiment,
        "hate": hate,
        "danger": danger,
        "status": "Comment created successfully",
    }


@app.post("/upload/")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Endpoint to analyze comments from a CSV file.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a CSV file."
        )

    content = await file.read()

    try:
        text_content = content.decode("utf-8")
        csv_file = StringIO(text_content)
        reader = csv.DictReader(csv_file)

        fieldnames = reader.fieldnames

        if not fieldnames:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        comment_field = next(
            (
                name
                for name in fieldnames
                if name.lower() in ["comment", "content", "text", "body"]
            ),
            None,
        )

        if not comment_field:
            comment_field = fieldnames[0]

        results = []

        for row in reader:
            comment_text = row.get(comment_field, "").strip()

            if comment_text:
                sentiment = sentiment_analyzer.predict(comment_text)
                hate = hate_analyzer.predict(comment_text)
                danger = danger_analyzer.predict(comment_text)[0]

                results.append(
                    {
                        "comment": comment_text,
                        "sentiment": sentiment,
                        "hate": hate,
                        "danger": danger,
                    }
                )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid file encoding. Please use UTF-8."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

    return {
        "filename": file.filename,
        "results": results,
        "status": "CSV analyzed successfully",
    }
