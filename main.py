# main.py

import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId

from model import generate_fingerprint  # your reconstruction routine
from match import find_best_fingerprint_match

# ── CONFIG ──
MONGO_URI       = "mongodb+srv://admin:admin123@cluster0.do482.mongodb.net/Reconstructify"
DB_NAME         = "Reconstructify"
COLLECTION_NAME = "fingerprints"
SAVE_ROOT       = "saved_files"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ── APP SETUP ──
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class IDRequest(BaseModel):
    id: str

def get_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME], client

@app.post("/process")
async def process_id(payload: IDRequest):
    # 1) Validate and fetch
    try:
        oid = ObjectId(payload.id)
    except:
        raise HTTPException(400, "Invalid ObjectId")

    coll, client = get_collection()
    doc = coll.find_one({"_id": oid})
    client.close()
    if not doc:
        raise HTTPException(404, "Record not found")

    # 2) Write partial file locally
    partial_data = doc.get("partialFile")
    if not partial_data:
        raise HTTPException(500, "No partial fingerprint found")

    subdir = os.path.join(SAVE_ROOT, payload.id)
    os.makedirs(subdir, exist_ok=True)
    partial_name = doc.get("partialFileName", f"{payload.id}_partial.png")
    partial_path = os.path.join(subdir, partial_name)
    with open(partial_path, "wb") as f:
        f.write(partial_data)

    # 3) Generate the reconstructed fingerprint
    generate_fingerprint(partial_path)



    # 4) Read back the generated image
    generated_filename = "reconstructed_fingerprint.jpg"
    print("\nFingerprint Generated!\n")
    if not os.path.exists(generated_filename):
        raise HTTPException(500, f"Expected {generated_filename} not found")

    image_bytes = open(generated_filename, "rb").read()

    # 5) Return the image directly
    return Response(content=image_bytes, media_type="image/jpeg")


from fastapi.responses import Response

@app.get("/match/{id}")
async def get_best_match(id: str):
    # 1) Locate the reconstructed fingerprint
    # recon_path = os.path.join(SAVE_ROOT, id, "reconstructed_fingerprint.png")
    # if not os.path.exists(recon_path):
    #     raise HTTPException(404, detail="Reconstructed fingerprint not found")

    # 2) Run your matching function
    try:
        best_path, best_score = find_best_fingerprint_match(
            "reconstructed_fingerprint.jpg",
            "original_fingerprints"
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Matching failed: {e}")

    if not os.path.exists(best_path):
        raise HTTPException(404, detail="Best‐match image not found")

    # 3) Read & return the image bytes
    data = open(best_path, "rb").read()
    # infer MIME type from extension
    ext = os.path.splitext(best_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return Response(content=data, media_type=mime)
