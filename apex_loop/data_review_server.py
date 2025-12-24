
import os
import shutil
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import threading
import asyncio
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Import your existing logic
# Assuming data_cleaning.py is in the same directory
try:
    from apex_loop.data_cleaning import (
        calculate_md5, 
        is_blurry, 
        get_embeddings, 
        find_outliers, 
        find_label_issues_cleanlab,
        BLUR_THRESHOLD
    )
except ImportError:
    # Handle case where run from root
    from data_cleaning import (
        calculate_md5, 
        is_blurry, 
        get_embeddings, 
        find_outliers, 
        find_label_issues_cleanlab,
        BLUR_THRESHOLD
    )

app = FastAPI()

# --- State ---
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
QUARANTINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "quarantine", "reviewed_discarded"))

# In-memory store for candidates
# Structure: { "filepath": { "path": "...", "reasons": ["Blurry"], "score": 0.5, "status": "pending", "label": "class_x" } }
CANDIDATES: Dict[str, Dict] = {}
SCANNING = False
SCAN_PROGRESS = "Idle"

# Embedding cache for Similarity Search
GLOBAL_EMBEDDINGS = None
GLOBAL_PATHS = []
GLOBAL_INDEX = None # NearestNeighbors index

class ActionRequest(BaseModel):
    filepath: str
    action: str  # "keep" or "discard"
    
class BulkActionRequest(BaseModel):
    filepaths: List[str]
    action: str

class MoveRequest(BaseModel):
    filepath: str
    target_class: str

# --- Scanning Logic ---
def get_label_from_path(path: str) -> str:
    """Extracts class label from directory structure (parent folder name)."""
    return os.path.basename(os.path.dirname(path))

def run_scan_thread():
    global SCANNING, SCAN_PROGRESS, CANDIDATES
    SCANNING = True
    SCAN_PROGRESS = "Starting..."
    
    print("Starting background scan...")
    
    try:
        # 1. Duplicates
        SCAN_PROGRESS = "Checking for duplicates..."
        hashes = {}
        for root, _, files in os.walk(DATASET_DIR):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(root, f)
                    h = calculate_md5(full_path)
                    if h in hashes:
                        if full_path not in CANDIDATES:
                            CANDIDATES[full_path] = {
                                "path": full_path, 
                                "reasons": [], 
                                "status": "pending",
                                "label": get_label_from_path(full_path)
                            }
                        CANDIDATES[full_path]["reasons"].append(f"Duplicate of {os.path.basename(hashes[h])}")
                    else:
                        hashes[h] = full_path

        # 2. Blurry
        SCAN_PROGRESS = "Checking for blurry images..."
        for root, _, files in os.walk(DATASET_DIR):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(root, f)
                    is_bad, score = is_blurry(full_path, threshold=BLUR_THRESHOLD)
                    if is_bad:
                        if full_path not in CANDIDATES:
                            CANDIDATES[full_path] = {
                                "path": full_path, 
                                "reasons": [], 
                                "status": "pending",
                                "label": get_label_from_path(full_path)
                            }
                        CANDIDATES[full_path]["reasons"].append(f"Blurry (Score: {score:.2f})")
                        CANDIDATES[full_path]["score"] = float(score)

        # 3. Embeddings & cleanlab
        SCAN_PROGRESS = "Generating embeddings (this may take a while)..."
        emb, paths, labels = get_embeddings(DATASET_DIR)
        
        # Cache for Similarity Search
        global GLOBAL_EMBEDDINGS, GLOBAL_PATHS, GLOBAL_INDEX
        if len(emb) > 0:
            GLOBAL_EMBEDDINGS = emb
            GLOBAL_PATHS = paths
            # Fit NearestNeighbors for fast lookup
            SCAN_PROGRESS = "Indexing for similarity search..."
            nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='euclidean').fit(emb)
            GLOBAL_INDEX = nbrs
        
        if len(emb) > 0:
            SCAN_PROGRESS = "Analyzing outliers..."
            outliers = find_outliers(emb, labels, paths, std_dev_threshold=3.5)
            for path, dist in outliers:
                if path not in CANDIDATES:
                    CANDIDATES[path] = {
                        "path": path, 
                        "reasons": [], 
                        "status": "pending",
                        "label": get_label_from_path(path)
                    }
                CANDIDATES[path]["reasons"].append(f"Outlier (Dist: {dist:.2f})")
            
            SCAN_PROGRESS = "Analyzing label issues (Cleanlab)..."
            mislabeled = find_label_issues_cleanlab(emb, labels, paths, strictness='standard')
            for path in mislabeled:
                if path not in CANDIDATES:
                    CANDIDATES[path] = {
                        "path": path, 
                        "reasons": [], 
                        "status": "pending",
                        "label": get_label_from_path(path)
                    }
                CANDIDATES[path]["reasons"].append("Suspected Mislabeled")
        
        SCAN_PROGRESS = "Done"
    except Exception as e:
        print(f"Scan failed: {e}")
        SCAN_PROGRESS = f"Failed: {str(e)}"
    finally:
        SCANNING = False

# --- Routes ---

@app.get("/api/scan/start")
def start_scan():
    global SCANNING
    if SCANNING:
        return {"status": "Already scanning"}
    thread = threading.Thread(target=run_scan_thread)
    thread.start()
    return {"status": "Started"}

@app.get("/api/scan/status")
def get_status():
    return {
        "scanning": SCANNING,
        "progress": SCAN_PROGRESS,
        "count": len(CANDIDATES)
    }

@app.get("/api/classes")
def get_classes():
    """Returns a list of all available class names."""
    try:
        classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')]
        return sorted(classes)
    except Exception as e:
        return []

@app.get("/api/candidates")
def get_candidates():
    # Return list of pending candidates
    items = []
    for path, data in CANDIDATES.items():
        if data["status"] == "pending":
            items.append({
                "id": path, # Use path as ID
                "filename": os.path.basename(path),
                "reasons": data["reasons"],
                "score": data.get("score", 0),
                "label": data.get("label", get_label_from_path(path))
            })
    return items

@app.get("/api/image")
def get_image(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return HTTPException(status_code=404, detail="Image not found")

@app.get("/api/similar")
def find_similar(path: str):
    """
    Finds top 20 similar images to the given path using cached embeddings.
    """
    global GLOBAL_EMBEDDINGS, GLOBAL_PATHS, GLOBAL_INDEX
    
    if GLOBAL_INDEX is None or path not in GLOBAL_PATHS:
        return {"error": "Embeddings not ready or file not found in index."}
        
    try:
        # Get index of this file
        idx = GLOBAL_PATHS.index(path)
        query_vec = GLOBAL_EMBEDDINGS[idx].reshape(1, -1)
        
        distances, indices = GLOBAL_INDEX.kneighbors(query_vec)
        
        # indices is [[idx, nearest1, nearest2...]]
        
        similar_items = []
        # flatten
        found_indices = indices[0]
        found_dists = distances[0]
        
        for i, neighbor_idx in enumerate(found_indices):
            if neighbor_idx == idx:
                continue # Skip self within reason 
            
            neighbor_path = GLOBAL_PATHS[neighbor_idx]
            
            # Check if this file is already in CANDIDATES
            reasons = []
            if neighbor_path in CANDIDATES:
                reasons = CANDIDATES[neighbor_path]["reasons"]
                
            similar_items.append({
                "id": neighbor_path,
                "filename": os.path.basename(neighbor_path),
                "reasons": reasons,
                "distance": float(found_dists[i]),
                "is_candidate": neighbor_path in CANDIDATES,
                "label": get_label_from_path(neighbor_path)
            })
            
        return similar_items
        
    except ValueError:
        return {"error": "File not in embedding cache."}
    except Exception as e:
        print(f"Similarity search error: {e}")
        return {"error": str(e)}

@app.post("/api/action/move")
def move_file(req: MoveRequest):
    """Moves a file to a different class directory."""
    if not os.path.exists(req.filepath):
         return {"error": "File not found"}
    
    target_dir = os.path.join(DATASET_DIR, req.target_class)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    filename = os.path.basename(req.filepath)
    new_path = os.path.join(target_dir, filename)
    
    # Handle filename collision
    if os.path.exists(new_path):
        base, ext = os.path.splitext(filename)
        new_path = os.path.join(target_dir, f"{base}_moved_{np.random.randint(1000)}{ext}")
        
    try:
        shutil.move(req.filepath, new_path)
        
        # Update CANDIDATES if present
        if req.filepath in CANDIDATES:
            CANDIDATES[req.filepath]["status"] = "moved"
            
        return {"status": "ok", "new_path": new_path}
    except Exception as e:
        print(f"Move failed: {e}")
        return {"error": str(e)}

@app.post("/api/action")
def take_action(req: ActionRequest):
    if req.filepath not in CANDIDATES and req.action == "discard":
         # Allow discarding non-candidates too
         pass
         
    if req.filepath in CANDIDATES:
        CANDIDATES[req.filepath]["status"] = req.action
    
    if req.action == "discard":
        # Move to quarantine
        try:
            rel_path = os.path.relpath(req.filepath, DATASET_DIR)
            dest = os.path.join(QUARANTINE_DIR, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(req.filepath, dest)
        except Exception as e:
            pass
            
    return {"status": "ok"}

@app.post("/api/action/bulk")
def take_action_bulk(req: BulkActionRequest):
    count = 0
    for fp in req.filepaths:
        try:
            # reuse logic
            if fp in CANDIDATES:
                CANDIDATES[fp]["status"] = req.action
            
            if req.action == "discard":
                rel_path = os.path.relpath(fp, DATASET_DIR)
                dest = os.path.join(QUARANTINE_DIR, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.exists(fp):
                    shutil.move(fp, dest)
                    count += 1
        except Exception as e:
            print(f"Failed to move {fp}: {e}")
            
    return {"status": "ok", "processed": count}

# Serve UI
@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Create quarantine dir
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    print(f"Starting server... Go to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
