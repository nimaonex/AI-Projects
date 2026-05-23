from loader_class import DOCXLoader
from fastapi import UploadFile, File, APIRouter



router = APIRouter(prefix = "/upload-docx", tags = ["upload-docx"])

@router.post("/upload-docx")
async def upload_docx(file: UploadFile = File(...)):
    # Step 1: read file into bytes
    file_bytes: bytes = await file.read()

    # Step 2: pass bytes to your loader
    loader = DOCXLoader(file_content=file_bytes)
    paragraphs = loader.load()
    list_completions = [a.page_content for a in paragraphs]
    
    return {"filename": file.filename, "size": len(file_bytes), "content": "\n".join(list_completions)} 
