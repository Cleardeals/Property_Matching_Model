# app/main.py
from fastapi import FastAPI, HTTPException
from app.matcher import PropertyMatcher
from app.routing import run_routing_process
import traceback # <-- ADD THIS IMPORT

app = FastAPI(
    title="Property Matching & Lead Routing API",
    description="An API to manage property similarity matching and lead routing.",
    version="1.0.0"
)

try:
    matcher = PropertyMatcher()
except Exception as e:
    print(f"❌ CRITICAL: Failed to initialize PropertyMatcher. The API will not be functional. Error: {e}")
    matcher = None

@app.get("/")
def read_root():
    return {"status": "Property Matching API is running."}

@app.post("/run_weekly_matching")
def run_weekly_matching():
    if not matcher:
        raise HTTPException(status_code=503, detail="Matcher service is unavailable due to a loading error.")
    
    try:
        result = matcher.create_and_save_mapping()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        # --- THIS IS THE FIX ---
        # Print the full, detailed error to the server console
        print("--- ❌ An unexpected error occurred in /run_weekly_matching ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/run_lead_routing")
def run_lead_routing():
    try:
        result = run_routing_process()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        # Also add the fix here for consistency
        print("--- ❌ An unexpected error occurred in /run_lead_routing ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")