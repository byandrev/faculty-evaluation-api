"""
Main API module defining the FastAPI application and its endpoints.
"""

import csv
from io import StringIO

import ollama
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.models.comment import Comment, CommentList
from api.nlp import danger_analyzer, hate_analyzer, sentiment_analyzer

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)

SUMMARIZE_PROMPT = """Actúa como un analista institucional en evaluación docente universitaria, con experiencia en análisis de retroalimentación estudiantil y detección de riesgos académicos y conductuales.

Objetivo del resumen (OBLIGATORIO)
Generar un resumen ejecutivo narrativo, corto y claro, en lenguaje institucional, que describa: la percepción general del docente, sus principales áreas de mejora, y la existencia de comentarios de alerta que requieren atención inmediata, si los hay.

Este resumen debe poder leerse como una sola conclusión gerencial.

🚨 Regla crítica de seguridad (NO NEGOCIABLE)
Si se detecta al menos un comentario relacionado con: acoso sexual, conducta sexual inapropiada, intimidación, violencia, abuso de poder,

DEBES: mencionarlo explícitamente en el resumen, indicar que requiere atención institucional inmediata, sin citar ni describir el contenido sensible.

✍️ Formato de salida esperado

Redacta un solo párrafo siguiendo estrictamente esta estructura lógica, SOLO DEVUELVE UN PARRAFO :

"El docente presenta comentarios mayormente [positivos / mixtos / negativos] por parte de los estudiantes, destacándose principalmente en [fortalezas generales]. No obstante, se identifican algunas áreas de mejora relacionadas con [áreas principales de mejora]. Adicionalmente, se detectaron comentarios de alerta relacionados con [tipo de alerta], los cuales deben ser tratados de manera inmediata."

📌 Reglas de redacción

- Usa lenguaje institucional, objetivo y profesional.
- No incluyas citas textuales.
- No hagas juicios personales ni sancionatorios.
- No menciones cantidades exactas de comentarios sensibles.
- Si NO existen alertas, el resumen debe cerrar con: "No se identificaron comentarios de alerta que requieran atención inmediata."

Lista de comentarios separados por coma:

{}"""

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


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """
    Root endpoint returning a simple greeting message.
    """
    return {"message": "Hello World"}


@app.post("/comments/")
@limiter.limit("30/minute")
async def analyze_comment(request: Request, comment: Comment):
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
@limiter.limit("10/minute")
async def analyze_csv(request: Request, file: UploadFile = File(...)):
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
    prompt = SUMMARIZE_PROMPT.format(comments_text)

    try:
        response = ollama.chat(
            model="gemma3",
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
