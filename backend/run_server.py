"""
Quick launcher — start FastAPI server with frontend.
Usage: python run_server.py
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from config import API_HOST, API_PORT

if __name__ == "__main__":
    print(f"\n  Solar Panel Fault Detection API")
    print(f"  Dashboard: http://localhost:{API_PORT}")
    print(f"  API Docs:  http://localhost:{API_PORT}/docs\n")

    uvicorn.run(
        "api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
