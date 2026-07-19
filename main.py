from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

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

    # -----------------------------
    # Empty input handling
    # -----------------------------

    if not req.question.strip() or len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.20,
            "answerable": False
        }

    # -----------------------------
    # Build Context
    # -----------------------------

    context = ""

    for chunk in req.chunks:
        context += f"{chunk.chunk_id}: {chunk.text}\n\n"

    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = f"""
You are a grounded question answering assistant.

Answer ONLY using the provided context.

If the answer cannot be fully supported by the context, return exactly this JSON:

{{
  "answer": "I don't know",
  "citations": [],
  "answerable": false
}}

If the answer CAN be supported, return ONLY valid JSON like:

{{
  "answer": "...",
  "citations": ["C1","C2"],
  "answerable": true
}}

Rules:

- Never use outside knowledge.
- Never guess.
- Cite ONLY chunk IDs that are provided.
- Do NOT invent chunk IDs.
- Return ONLY JSON.
- No markdown.
- No explanation.

Context:

{context}

Question:

{req.question}
"""

    # -----------------------------
    # Ask Groq
    # -----------------------------

    try:

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

        output = response.choices[0].message.content.strip()

        # Remove markdown if present
        output = output.replace("```json", "").replace("```", "").strip()

        result = json.loads(output)

    except Exception:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.20,
            "answerable": False
        }

    # -----------------------------
    # Validate citations
    # -----------------------------

    valid_ids = {chunk.chunk_id for chunk in req.chunks}

    citations = [
        cid
        for cid in result.get("citations", [])
        if cid in valid_ids
    ]

    answer_text = result.get("answer", "I don't know")
    answerable = result.get("answerable", False)

    # -----------------------------
    # Final Guardrail
    # -----------------------------

    if (
        not answerable
        or answer_text.lower() == "i don't know"
        or len(citations) == 0
    ):
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.20,
            "answerable": False
        }

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": 0.95,
        "answerable": True
    }