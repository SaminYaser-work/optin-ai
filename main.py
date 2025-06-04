from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# Enable CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and FAISS index
print("Loading model and index...")
model = SentenceTransformer("BAAI/bge-small-en")
index_path = "template_index.faiss"
metadata_path = "template_metadata.json"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())

if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = []

class Template(BaseModel):
    id: int
    title: str
    type: str
    industries: list[str] = []
    categories: list[str] = []
    goals: list[str] = []

@app.get("/templates/search")
def search_templates(query: str = Query(..., description="User query for the popup they want"), k: int = 5):
    try:
        # Embed the query
        prompt = "Represent this sentence for retrieval: " + query
        q_emb = model.encode(prompt, convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb)

        # Search the FAISS index
        D, I = index.search(np.array([q_emb]), k)

        # Retrieve top-k metadata
        results = [metadata[i] for i in I[0]]
        return {"query": query, "results": results}

    except Exception as e:
        return {"error": str(e)}

@app.post("/templates")
def add_templates(templates: list[Template]):
    try:
        add(templates)
        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

def add(templates: list[Template]):
    all_embeddings = []
    for template in templates:
        text_parts = [
            template.title,
            "Optin Type: " + template.type,
            "goals: " + " ".join(template.goals),
            "categories: " + " ".join(template.categories),
            "industries: " + " ".join(template.industries),
        ]
        full_text = " ".join(text_parts).strip()
        prompt = "Represent this sentence for retrieval: " + full_text
        emb = model.encode(prompt, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        all_embeddings.append(emb)

        metadata.append({
            "id": template.id,
            "title": template.title,
            "type": template.type,
            "categories": template.categories,
            "goals": template.goals,
            "industries": template.industries,
        })

    # Add all to index
    index.add(np.array(all_embeddings))

    # Save
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)