# Directory structure:
# 
# fastapi_service/
# ├── app.py
# ├── models/
# │   └── __init__.py
# ├── requirements.txt
# └── Dockerfile

# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from transformers import pipeline
import uvicorn

# --- Pydantic schemas ---
class Comment(BaseModel):
    text: str = Field(..., example="This is amazing content!")
    likes: Optional[int] = Field(0, ge=0, example=25)
    timestamp: Optional[str] = Field(None, example="2025-06-10T12:00:00Z")
    is_reply: Optional[bool] = Field(False)

class AnalyzeRequest(BaseModel):
    comments: List[Comment]

class SentimentResult(BaseModel):
    label: str
    score: float

class ToxicityResult(BaseModel):
    label: str
    score: float

class CommentAnalysis(BaseModel):
    original_text: str
    likes: int
    timestamp: Optional[str]
    is_reply: bool
    sentiment: SentimentResult
    toxicity: ToxicityResult

class AnalyzeResponse(BaseModel):
    results: List[CommentAnalysis]
    meta: Dict[str, Any]

# --- FastAPI app init ---
app = FastAPI(
    title="OBN Comment Analyzer",
    description="Analyze YouTube comments for sentiment and toxicity",
    version="1.0.0"
)

# --- Load models once at startup ---
@app.on_event("startup")
def load_models():
    global sentiment_pipeline, toxicity_pipeline
    sentiment_pipeline = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")
    toxicity_pipeline = pipeline(
        "text-classification", model="unitary/toxic-bert"
    )


# --- Analysis endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided.")

    analyses = []
    positive, negative, neutral = 0, 0, 0
    for c in request.comments:
        sent = sentiment_pipeline(c.text)[0]
        tox = toxicity_pipeline(c.text)[0]
        # Count for meta
        lbl = sent['label'].lower()
        if lbl == 'positive': positive += 1
        elif lbl == 'negative': negative += 1
        else: neutral += 1

        analyses.append(CommentAnalysis(
            original_text=c.text,
            likes=c.likes or 0,
            timestamp=c.timestamp,
            is_reply=c.is_reply,
            sentiment=SentimentResult(label=sent['label'], score=sent['score']),
            toxicity=ToxicityResult(label=tox['label'], score=tox['score'])
        ))

    total = positive + negative + neutral or 1
    meta = {
        "positive_ratio": positive / total,
        "negative_ratio": negative / total,
        "neutral_ratio": neutral / total,
    }

    return AnalyzeResponse(results=analyses, meta=meta)

# --- Run with: uvicorn app:app --host 0.0.0.0 --port 5000 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)


# Dockerfile
#
# FROM python:3.9-slim
# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .
# EXPOSE 5000
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
