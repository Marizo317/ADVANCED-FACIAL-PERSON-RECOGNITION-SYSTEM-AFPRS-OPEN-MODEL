# llm/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from generator import Generator

app = FastAPI(title="AFPRS LLM Service")
gen = Generator()

class DescRequest(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    gaze: Optional[str] = None
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    position: Optional[str] = None
    depth: Optional[str] = None

class DescResponse(BaseModel):
    description: str

@app.post("/generate", response_model=DescResponse)
def generate(req: DescRequest):
    desc = gen.generate(
        req.age, req.gender, req.emotion, req.gaze,
        req.upper_color, req.lower_color, req.position, req.depth
    )
    return DescResponse(description=desc)

@app.get("/health")
def health():
    return {"status": "ok"}
