"""
Main API module defining the FastAPI application and its endpoints.
"""

import csv
import resource
import time
from io import StringIO

import ollama
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.models.comment import Comment, CommentList
from api.nlp import (
    danger_analyzer,
    danger_analyzer_v2,
    danger_analyzer_v3,
    hate_analyzer,
    sentiment_analyzer,
)
from api.settings import settings

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_metrics() -> dict:
    return {
        "wall_time": time.perf_counter(),
        "cpu_time": time.process_time(),
        "ram_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def build_metrics(metrics_start: dict) -> dict:
    wall_elapsed = time.perf_counter() - metrics_start["wall_time"]
    cpu_elapsed = time.process_time() - metrics_start["cpu_time"]
    ram_delta_kb = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - metrics_start["ram_kb"],
        0,
    )

    return {
        "time_seconds": round(wall_elapsed, 6),
        "cpu_seconds": round(cpu_elapsed, 6),
        "ram_delta_mb": round(ram_delta_kb / 1024, 6),
    }


def get_danger_analyzer(model: str):
    """
    Returns the appropriate danger analyzer based on the model name.
    """

    if model == "evd2":
        return danger_analyzer_v2
    elif model == "evd3":
        return danger_analyzer_v3

    return danger_analyzer


def map_danger_label(label: str, model: str) -> str:
    """
    Maps the danger label to a descriptive value based on the model used.
    """

    if model == "evd":
        mapping = {"LABEL_0": "normal", "LABEL_1": "critico", "LABEL_2": "muy_critico"}
    elif model == "evd2":  # evd2
        mapping = {
            "LABEL_0": "bueno",
            "LABEL_1": "bajo",
            "LABEL_2": "critico",
            "LABEL_3": "muy_critico",
        }
    else:  # evd3
        mapping = {"LABEL_0": "bajo", "LABEL_1": "medio", "LABEL_2": "alto"}

    return mapping.get(label)


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """
    Root endpoint returning a simple greeting message.
    """
    return {"message": "Hello World"}


@app.post("/comments/")
@limiter.limit("30/minute")
async def analyze_comment(
    request: Request,
    comment: Comment,
    model: str = Query(
        default="evd",
        regex="^(evd|evd2|evd3)$",
        description="Danger analysis model to use (evd, evd2 or evd3)",
    ),
):
    """
    Endpoint to analyse and store a comment.
    """

    metrics_start = start_metrics()

    sentiment = sentiment_analyzer.predict(comment.content)
    hate = hate_analyzer.predict(comment.content)

    analyzer = get_danger_analyzer(model)
    danger_label = analyzer.predict(comment.content)[0]
    danger = map_danger_label(danger_label["label"], model)

    return {
        "comment": comment.content,
        "sentiment": sentiment,
        "hate": hate,
        "danger": {"label": danger_label, "description": danger},
        "model_used": model,
        "metrics": build_metrics(metrics_start),
        "status": "Comment created successfully",
    }


@app.post("/upload/")
@limiter.limit("10/minute")
async def analyze_csv(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(
        default="evd",
        regex="^(evd|evd2|evd3)$",
        description="Danger analysis model to use (evd, evd2 or evd3)",
    ),
):
    """
    Endpoint to analyze comments from a CSV file.
    """

    metrics_start = start_metrics()

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

        analyzer = get_danger_analyzer(model)

        for row in reader:
            comment_text = row.get(comment_field, "").strip()

            if comment_text:
                sentiment = sentiment_analyzer.predict(comment_text)
                hate = hate_analyzer.predict(comment_text)
                danger_label = analyzer.predict(comment_text)[0]
                danger = map_danger_label(danger_label["label"], model)

                results.append(
                    {
                        "comment": comment_text,
                        "sentiment": sentiment,
                        "hate": hate,
                        "danger": {"label": danger_label, "description": danger},
                        "model_used": model,
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
        "model_used": model,
        "metrics": build_metrics(metrics_start),
        "status": "CSV analyzed successfully",
    }


@app.post("/summarize/")
@limiter.limit("5/minute")
async def summarize_comments(request: Request, comment_list: CommentList):
    """
    Endpoint to generate an executive summary of comments using Gemma3 via Ollama.

    Receives a list of comments and returns an institutional summary analyzing
    the overall perception, areas for improvement, and any alert comments
    requiring immediate attention.
    """

    if not comment_list.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    comments_text = ", ".join(comment_list.comments)
    prompt = settings.summarize_prompt.format(comments_text)

    try:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        summary = response["message"]["content"]

        return {
            "summary": summary,
            "total_comments": len(comment_list.comments),
            "status": "Summary generated successfully",
        }

    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama service error: {str(e)}. Make sure Ollama is running and gemma3 model is available.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )
