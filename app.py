"""
Main Application Entry Point
-------------------------

This is the main entry point for the Product Count API.
"""

import uvicorn
from prodcutCount.api.endpoints import app
from prodcutCount.config.settings import Settings

settings = Settings()

if __name__ == "__main__":
    uvicorn.run(
        "prodcutCount.api.endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
