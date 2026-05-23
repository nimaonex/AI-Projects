from fastapi import FastAPI, UploadFile, File
from req import DOCXLoader

app = FastAPI()

@app.post("/upload-docx")
async def upload_docx(file: UploadFile = File(...)):
    # Step 1: read file into bytes
    file_bytes: bytes = await file.read()

    # Step 2: pass bytes to your loader
    loader = DOCXLoader(file_content=file_bytes)
    paragraphs = loader.load()
    list_completions = [a.page_content for a in paragraphs]
    
    return {"filename": file.filename, "size": len(file_bytes), "content": list_completions}


# with open(r"E:\AI Projects\doc_files\CE1_EN.docx", "rb") as f:
#     doc_bytes = f.read()

# loader = DOCXLoader(file_content=doc_bytes)
# paragraphs = loader.load()
# list_completions = [a.page_content for a in paragraphs]
# print(list_completions)