"""
API Endpoints Module
-----------------

This module provides the FastAPI endpoints for the product counting API.
"""

import os
# Set OpenMP environment variable to avoid runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Dict, List
import logging
from pathlib import Path
import numpy as np
import cv2
from ..core.detector import ProductDetector
from ..core.matcher import ProductMatcher
from ..config.settings import Settings

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize settings
settings = Settings()

# Initialize detector and matcher
detector = ProductDetector(
    model_path=settings.model_path,
    confidence_threshold=settings.confidence_threshold
)

matcher = ProductMatcher(
    similarity_threshold=settings.similarity_threshold
)

# Load reference images
@router.on_event("startup")
async def startup_event():
    """Load reference images on startup."""
    try:
        reference_dir = Path(settings.reference_images_dir)
        if not reference_dir.exists():
            logger.warning(f"Reference directory {reference_dir} does not exist")
            return
        
        matcher.add_reference_images(str(reference_dir))
        logger.info("Successfully loaded reference images")
    except Exception as e:
        logger.error(f"Error loading reference images: {str(e)}")

@router.post("/count-products")
async def count_products(
    image: UploadFile = File(...),
    confidence_threshold: float = None,
    similarity_threshold: float = None
) -> Dict:
    """
    Count products in an uploaded image.
    
    Args:
        image (UploadFile): The image file to process
        confidence_threshold (float, optional): Override default detection confidence
        similarity_threshold (float, optional): Override default matching similarity
        
    Returns:
        Dict: Product counts and detection details
    """
    try:
        # Read and validate image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.array(img)
        
        # Detect products
        detections = detector.detect(
            img_array,
            confidence_threshold=confidence_threshold
        )
        
        # Match and count products
        if similarity_threshold is not None:
            matcher.similarity_threshold = similarity_threshold
        
        product_counts, product_matches = matcher.count_products(
            detections,
            img_array
        )
        
        # Prepare response
        response = {
            "product_counts": product_counts,
            "total_products": sum(product_counts.values()),
            "detections": len(detections),
            "matches": len(product_matches),
            "details": [
                {
                    "product": name,
                    "similarity": float(sim),
                    "bbox": list(map(int, bbox))
                }
                for name, sim, bbox in product_matches
            ]
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@router.post("/batch-count-products")
async def batch_count_products(
    images: List[UploadFile] = File(...),
    confidence_threshold: float = None,
    similarity_threshold: float = None
) -> List[Dict]:
    """
    Count products in multiple images.
    
    Args:
        images (List[UploadFile]): List of image files to process
        confidence_threshold (float, optional): Override default detection confidence
        similarity_threshold (float, optional): Override default matching similarity
        
    Returns:
        List[Dict]: List of product counts and detection details for each image
    """
    try:
        # Process all images
        results = []
        for image in images:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')
            img_array = np.array(img)
            
            # Detect products
            detections = detector.detect(
                img_array,
                confidence_threshold=confidence_threshold
            )
            
            # Match and count products
            if similarity_threshold is not None:
                matcher.similarity_threshold = similarity_threshold
            
            product_counts, product_matches = matcher.count_products(
                detections,
                img_array
            )
            
            # Prepare result for this image
            result = {
                "filename": image.filename,
                "product_counts": product_counts,
                "total_products": sum(product_counts.values()),
                "detections": len(detections),
                "matches": len(product_matches),
                "details": [
                    {
                        "product": name,
                        "similarity": float(sim),
                        "bbox": list(map(int, bbox))
                    }
                    for name, sim, bbox in product_matches
                ]
            }
            results.append(result)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        ) 