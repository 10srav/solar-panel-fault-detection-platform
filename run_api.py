#!/usr/bin/env python
"""Run the API server with SQLite for development."""
import os
import sys

# Set SQLite database URL before importing app
os.environ["DATABASE_URL"] = "sqlite:///./solar_detection.db"

if __name__ == "__main__":
    import uvicorn
    from src.api.database import init_db

    # Initialize database
    init_db()

    # Run server
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
