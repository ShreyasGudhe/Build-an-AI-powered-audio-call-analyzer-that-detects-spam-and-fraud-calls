"""
FastAPI Application Startup Script
Runs the fraud detection API server
"""
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn
    from app.api import app
    
    print("\n" + "="*80)
    print("AI FRAUD CALL ANALYZER - API SERVER")
    print("="*80)
    print("\nStarting API server...")
    print("  * API: http://127.0.0.1:8000")
    print("  * Docs: http://127.0.0.1:8000/docs")
    print("  * Frontend: http://localhost:5173")
    print("\nPress CTRL+C to stop\n")
    print("="*80 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
