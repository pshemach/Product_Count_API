"""
Product Count API
---------------
Main entry point for the FastAPI application
"""

import os
# Set OpenMP environment variable to avoid runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from prodcutCount.api.endpoints import router as detection_router
from prodcutCount.api.reference_endpoints import router as reference_router

app = FastAPI(
    title="Product Count API",
    description="API for counting products in supermarket images",
    version="0.1.0"
)

# Include routers
app.include_router(detection_router, prefix="/api/v1")
app.include_router(reference_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Product Count API",
        "version": "0.1.0",
        "description": "API for counting products in supermarket images"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8075)