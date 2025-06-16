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
from langdetect import detect
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

class EmotionResult(BaseModel):
    label: str
    score: float

class ToxicityResult(BaseModel):
    label: str
    score: float

class SpamResult(BaseModel):
    label: str
    score: float

class TopicResult(BaseModel):
    label: str
    score: float

class SummarizationResult(BaseModel):
    summary: str

class CommentAnalysis(BaseModel):
    original_text: str
    likes: int
    timestamp: Optional[str]
    is_reply: bool
    sentiment: SentimentResult
    emotion: EmotionResult
    toxicity: ToxicityResult
    # spam: SpamResult
    topic: TopicResult
    # language: str
    # summary: SummarizationResult

class AnalyzeResponse(BaseModel):
    results: List[CommentAnalysis]
    meta: Dict[str, Any]

# --- FastAPI app init ---
app = FastAPI(
    title="OBN Comment Analyzer",
    description="Analyze YouTube comments for multiple NLP insights",
    version="1.1.0"
)

# --- Load models once at startup ---


@app.on_event("startup")
def load_models():
    global sentiment_pipeline, emotion_pipeline, toxicity_pipeline
    global  topic_pipeline
    
    # summarization_pipeline,spam_pipeline

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
     
    )

    emotion_pipeline = pipeline(
        "text-classification",
        model="bhadresh-savani/bert-base-go-emotion",
     
    )

    toxicity_pipeline = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
     
    )

    # spam_pipeline = pipeline(
    #     "text-classification",
    #     model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
     
    # )

    topic_pipeline = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
     
    )

    # summarization_pipeline = pipeline(
    #     "summarization",
    #     model="google/pegasus-xsum",
     
    # )


# --- Analysis endpoint ---
# @app.post("/analyze", response_model=AnalyzeResponse)
# def analyze(request: AnalyzeRequest):
#     if not request.comments:
#         raise HTTPException(status_code=400, detail="No comments provided.")

#     analyses = []
#     counters = {"positive": 0, "negative": 0, "neutral": 0}
#     for c in request.comments:
#         text = c.text
#         sent = sentiment_pipeline(text)[0]
#         emo = emotion_pipeline(text)[0]
#         tox = toxicity_pipeline(text)[0]
#         # spam = spam_pipeline(text)[0]
#         # Define candidate topics for zero-shot
#         topics = ["praise","criticism","question","denial","propaganda","ethnic_sentiment","religious_comment","call_to_action","support_opposition","conspiracy","misinformation","spam","neutral","other"]

#         topic = topic_pipeline(text, candidate_labels=topics)["labels"][0]
#         topic_score = topic_pipeline(text, candidate_labels=topics)["scores"][0]
#         # Language detection
#         # lang = detect(text)
#         # Summarization (limit long text)
#         # summary = summarization_pipeline(text, max_length=20, min_length=3, do_sample=False)[0]["summary_text"]
#         print("*****************")
#         print(sent, emo, tox,  topic, topic_score )
#         print("*****************")
#         # Count sentiment
#         lbl = sent["label"].lower()
#         if lbl in counters:
#             counters[lbl] += 1

#         analyses.append(CommentAnalysis(
#             original_text=text,
#             likes=c.likes or 0,
#             timestamp=c.timestamp,
#             is_reply=c.is_reply,
#             sentiment=SentimentResult(label=sent["label"], score=sent["score"]),
#             emotion=EmotionResult(label=emo["label"], score=emo["score"]),
#             toxicity=ToxicityResult(label=tox["label"], score=tox["score"]),
#             # spam=SpamResult(label=spam["label"], score=spam["score"]),
#             topic=TopicResult(label=topic, score=topic_score),
#             # language=lang,
#             # summary=SummarizationResult(summary=summary)
#         ))

#     total = sum(counters.values()) or 1
#     meta = {f"{k}_ratio": v / total for k, v in counters.items()}

#     return AnalyzeResponse(results=analyses, meta=meta)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided.")
    
    print("--> 1. Received comments for analysis:", len(request.comments))
    comments = request.comments
    texts = [c.text for c in comments]
    print("--> 2. Extracted texts for analysis:", len(texts))
    # 🔁 Batch process all texts
    sentiments =  sentiment_pipeline(texts, batch_size=8)
    emotions =emotion_pipeline(texts, batch_size=6)
    toxicities = toxicity_pipeline(texts, batch_size=4)
    print("sentiment outputs with batch of **8**:", sentiments)
    print("emotion outputs with batch of **6**:", emotions)
    print("toxicity outputs with batch of **4**:", toxicities)

    # Zero-shot topic classification requires specifying candidate labels
    topics = [
        "praise", "criticism", "question", "denial", "propaganda",
        "ethnic_sentiment", "religious_comment", "call_to_action",
        "support_opposition", "conspiracy", "misinformation",
        "spam", "neutral", "other"
    ]
    topic_outputs = topic_pipeline(texts, candidate_labels=topics, batch_size=2)
    print("Topic outputs with batch of **2**:", topic_outputs)

    # Initialize counters
    counters = {"positive": 0, "negative": 0, "neutral": 0}
    analyses = []

    for idx, c in enumerate(comments):
        print(f"Processing comment {idx + 1}/{len(comments)}: {c.text[:50]}...")
        sent = sentiments[idx]
        emo = emotions[idx]
        tox = toxicities[idx]
        topic_label = topic_outputs[idx]["labels"][0]
        topic_score = topic_outputs[idx]["scores"][0]

        # Count sentiment
        lbl = sent["label"].lower()
        if lbl in counters:
            counters[lbl] += 1

        analyses.append(CommentAnalysis(
            original_text=c.text,
            likes=c.likes or 0,
            timestamp=c.timestamp,
            is_reply=c.is_reply,
            sentiment=SentimentResult(label=sent["label"], score=sent["score"]),
            emotion=EmotionResult(label=emo["label"], score=emo["score"]),
            toxicity=ToxicityResult(label=tox["label"], score=tox["score"]),
            topic=TopicResult(label=topic_label, score=topic_score),
        ))

    total = sum(counters.values()) or 1
    meta = {f"{k}_ratio": v / total for k, v in counters.items()}
    print("--> 3. Analysis complete. Total comments analyzed:", len(analyses))
    return AnalyzeResponse(results=analyses, meta=meta)

# --- Run with: uvicorn app:app --host 0.0.0.0 --port 5000 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

