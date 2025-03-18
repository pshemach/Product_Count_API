"""
Reference Image API Endpoints
--------------------------
Handles reference image management endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from pathlib import Path
import io

from prodcutCount.utils.reference_manager import ReferenceManager
from prodcutCount.config.settings import Settings

router = APIRouter()

# Initialize manager with settings
reference_manager = ReferenceManager(Settings().reference_images_dir)

@router.post("/reference/add")
async def add_reference_image(file: UploadFile = File(...), product_name: str = None):
    """Add a new reference image"""
    try:
        # Read and convert image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Add reference image
        new_id = reference_manager.add_reference_image(image, product_name)
        return {"status": "success", "id": new_id, "name": product_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/reference/{image_id}")
async def remove_reference_image(image_id: str):
    """Remove a reference image"""
    try:
        reference_manager.remove_reference_image(image_id)
        return {"status": "success", "message": f"Removed reference image {image_id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/reference/list")
async def list_references():
    """List all reference images"""
    return reference_manager.get_all_references() 