from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import json


ALLOWED_ORIGINS = ["http://localhost:5173"]
MODEL_NAME = "llama3.1"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    topic: str
    questionnum: int


class ExplainRequest(BaseModel):
    question: str


@app.post("/generate")
def generate_quiz(data: QuizRequest):
    system_prompt = (
    f"You are a helpful assistant in that generates quiz questions. "
    f"Respond ONLY with a JSON array of {data.questionnum} multiple-choice quiz questions about '{data.topic}'.\n\n"
    f"Each question must follow this format:\n"
    f"- 'question': a string\n"
    f"- 'options': a list of 4 full-length answer choices (do not label them as A, B, C, D)\n"
    f"- 'answer': exactly one of the option strings — copy it **verbatim** from the 'options' list.\n\n"
    f"Rules:\n"
    f"- Do NOT include 'A.', 'B.', etc. in the options.\n"
    f"- The 'answer' field must match one of the options exactly — no letters like 'A' or 'B'.\n"
    f"- Respond only with the raw JSON. No markdown, explanation, or formatting outside the JSON."
)
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {data.questionnum} multiple-choice questions about {data.topic}"}
            ]
        )
        content = response.get("message", {}).get("content", "").strip()
        quiz_json = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI response was not valid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"quiz": quiz_json}



@app.post("/explain")
def explain_answer(data: ExplainRequest):
    

    system_prompt = (
        "You are a helpful technical assistant. "
        "When given a quiz question, your job is to briefly explain the key concept the question is about."
    )

    user_prompt = (
        f"Here is a quiz question:\n"
        f"\"{data.question}\"\n\n"
        "Explain the main concept this question is testing. Keep the explanation clear and under 3 sentences. "
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        content = response.get("message", {}).get("content", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")

    return {"explanation": content}