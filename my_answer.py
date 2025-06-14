import os
import time
import base64
import mimetypes
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

load_dotenv()


# === Init ===
app = FastAPI()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Ensure this is set in your env
print(os.getenv("GEMINI_API_KEY"))
last_request_time = 0
REQUEST_INTERVAL = 1.2  # seconds

# === 1. Describe Image with Gemini ===
def get_image_description(image_path):
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    image_part = {
        "inline_data": {
            "mime_type": mime_type,
            "data": b64_data
        }
    }

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([
            image_part,
            "Describe this image for markdown documentation."
        ])
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error getting image description for {image_path}: {e}")
        return "Image"

# === 2. Embedding Function with Retry & Rate Limit ===
def get_query_embedding(text: str, delay=1.0, retries=3) -> list:
    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-exp-03-07",  # ✅ MUST MATCH .npz model
                content=text,
                task_type="retrieval_query"  # ✅ MUST be retrieval_query for questions
            )
            time.sleep(delay)
            return result['embedding']
        except Exception as e:
            print(f"⚠️ Query embedding failed: {e}")
            time.sleep(delay * (attempt + 1))
    return [0.0] * 3072  # fallback to 3072

# === 3. Load Precomputed Embeddings ===
def load_embeddings():
    data = np.load("corrected_embeddings.npz", allow_pickle=True)
    return data["chunks"], np.vstack(data["embeddings"])

# === 4. Generate LLM Response ===
def generate_llm_response(question: str, context: str):
    system_prompt = """> You are a knowledgeable and concise teaching assistant. Use only the information provided in the context to answer the question in detailed way including bullet points.
> 
> * Format your response using **Markdown**.
> * Use code blocks (` ``` `) for any code or command-line instructions.
> * Use bullet points or numbered lists for clarity where appropriate.
> 
> ⚠️ **Important:** If the context does not contain enough information to answer the question, reply exactly with:
> 
> ```
> I don't know
> ```
> 
> Do not attempt to guess, fabricate, or add external information.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "user", "parts": [f"Context:\n{context}\n\nQuestion: {question}"]}
        ],
        generation_config=GenerationConfig(
        max_output_tokens=512,
        temperature=0.5,
        top_p=0.95,
        top_k=40
    )
    )
    return response.text

# === 5. Main Answer Function ===
def answer(question: str, image: str = None):
    chunks, embeddings = load_embeddings()
    
    if image:
        image_description = get_image_description(f"data:image/jpeg;base64,{image}")
        question += f" {image_description}"

    question_embedding = get_query_embedding(question)

    # Cosine similarity
    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    )

    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    response = generate_llm_response(question, "\n".join(top_chunks))
    return {
        "question": question,
        "response": response,
        "top_chunks": top_chunks
    }

# === 6. FastAPI Endpoint ===
@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        return answer(data.get("question"), data.get("image"))
    except Exception as e:
        return {"error": str(e)}

# === 7. Run Server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
