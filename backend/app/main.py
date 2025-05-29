from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import json

from app.core.config import settings

# ALLOWED_ORIGINS = ["http://localhost:5173"]
# MODEL_NAME = "llama3.1"

print(settings.allowed_origins)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    topic: str
    question_quantity: int


class ExplainRequest(BaseModel):
    question: str

def chat_with_model(system_prompt: str, user_prompt: str) -> str:
    try:
        response = ollama.chat(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return response.get("message",{}).get("content" , "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")
    

def build_quiz_prompt(topic: str, quantity: int) -> str:
    return (
        f"You are a helpful assistant that generates quiz questions. "
        f"Respond ONLY with a JSON array of {quantity} multiple-choice quiz questions about '{topic}'.\n\n"
        f"Each question must follow this format:\n"
        f"- 'question': a string\n"
        f"- 'options': a list of 4 full-length answer choices (no A, B, C, D)\n"
        f"- 'answer': exactly one of the option strings — copied **verbatim** from 'options'.\n\n"
        f"Rules:\n"
        f"- Do NOT include 'A.', 'B.', etc. in the options.\n"
        f"- The 'answer' field must match one of the options exactly.\n"
        f"- Respond only with the raw JSON. No markdown, explanation, or formatting outside the JSON."
    )


def build_explanation_prompt(question: str) -> str:
    return (
        f"Here is a quiz question:\n"
        f"\"{question}\"\n\n"
        "Explain the main concept this question is testing. Keep the explanation clear and under 3 sentences."
    )

@app.post("/generate")
def generate_quiz(data: QuizRequest):
    system_prompt = build_quiz_prompt(data.topic, data.question_quantity)
    user_prompt = f"Generate {data.question_quantity} multiple-choice questions about {data.topic}"
    content = chat_with_model(system_prompt, user_prompt)

    try:
        quiz_json = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI response was not valid JSON.")

    return {"quiz": quiz_json}



@app.post("/explain")
def explain_answer(data: ExplainRequest):
    system_prompt = "You are a helpful technical assistant. When given a quiz question, your job is to briefly explain the key concept the question is about."
    user_prompt = build_explanation_prompt(data.question)

    content = chat_with_model(system_prompt, user_prompt)

    return {"explanation": content}

