from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class IDRequest(BaseModel):
    id: str

app = FastAPI()

# === CORS configuration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or list your exact Next.js origins
    allow_credentials=True,
    allow_methods=["*"],            # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],            # allow Content-Type, Authorization, etc.
)
# ============================

@app.post("/process")
async def process_id(payload: IDRequest):
    print("Received ID:", payload.id)
    return {"status": "received", "id": payload.id}
