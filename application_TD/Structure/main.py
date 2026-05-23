from fastapi import FastAPI
from upload_api import router as upload_router
from QA_api import router as QA_router
import logging

log = logging.getLogger(__name__)

app = FastAPI()
app.include_router(upload_router)
app.include_router(QA_router)

@app.get("/")
def root():
    return {"status": "ok"}
