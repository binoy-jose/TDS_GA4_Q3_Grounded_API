from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------
# Load Embedding Model
# -----------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Models
# -----------------------------

class Chunk(BaseModel):
    chunk_id: str
    text: str


class QuestionRequest(BaseModel):
    question: str
    chunks: list[Chunk]


# -----------------------------
# Health Check
# -----------------------------

@app.get("/")
def root():
    return {"message": "Grounded QA API Running"}


# -----------------------------
# Main Endpoint
# -----------------------------

@app.post("/answer")
def answer(req: QuestionRequest):

    # Empty input handling

    if len(req.chunks) == 0 or req.question.strip() == "":
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    # -------------------------
    # Embed question and chunks
    # -------------------------

    question_embedding = embed_model.encode(
        req.question,
        convert_to_tensor=True
    )

    chunk_texts = [c.text for c in req.chunks]

    chunk_embeddings = embed_model.encode(
        chunk_texts,
        convert_to_tensor=True
    )

    similarities = cos_sim(question_embedding, chunk_embeddings)[0].cpu().numpy()

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    
    # -------------------------
    # Unanswerable check
    # -------------------------

    if best_score < 0.35:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": round(min(best_score, 0.30), 2),
            "answerable": False
        }

    # -------------------------
    # Select Top Chunks
    # -------------------------

    top_indices = similarities.argsort()[-2:][::-1]

    citations = []
    context = ""

    for idx, score in zip(top_indices, similarities[top_indices]):

        if score > 0.30:
            context += req.chunks[idx].text + "\n\n"
            citations.append(req.chunks[idx].chunk_id)

    # If nothing passed similarity threshold

    if len(citations) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.20,
            "answerable": False
        }

    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are a retrieval-augmented question answering assistant.

Answer ONLY using the information in the provided context.

If the answer is not directly supported by the context,
reply exactly:

I don't know

Do not use outside knowledge.
Do not guess.
Do not explain.

Context:
{context}

Question:
{req.question}

Answer:
"""

    # -------------------------
    # Ask Groq
    # -------------------------

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    print("LLM Answer:", answer)

    # -------------------------
    # Final Check
    # -------------------------

    lower_answer = answer.lower()

    if (
        "don't know" in lower_answer
        or "dont know" in lower_answer
        or lower_answer == "unknown"
    ):
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.20,
            "answerable": False
        }

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(min(best_score, 0.99), 2),
        "answerable": True
    }