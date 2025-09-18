from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.review_agent import review_doc
from src.scoring import overall_score

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def submit_form(request: Request, text: str = Form(None), file: UploadFile = None):
    content = ""
    if file:
        content = (await file.read()).decode("utf-8")
    elif text:
        content = text
    else:
        content = "No input provided."

    feedback = review_doc(content)
    score = overall_score(content)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "feedback": feedback,
        "score": score
    })
