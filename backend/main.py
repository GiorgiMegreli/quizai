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

@app.post("/generate")
def generate_quiz(data: QuizRequest):
    system_prompt = (
        f"You are a helpful assistant that generates quiz questions. "
        f"Respond ONLY with a JSON array of {data.questionnum} multiple-choice quiz questions about '{data.topic}'. "
        f"Each question must include:\n"
        f"- 'question': a string\n"
        f"- 'options': a list of 4 possible answers\n"
        f"- 'answer': the correct answer string\n"
        f"Respond only with the JSON. Do not include any explanation or formatting outside the JSON."
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
