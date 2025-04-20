"""
Product Detector Module
--------------------

This module provides functionality for detecting products in images using YOLOv9.
"""

from pathlib import Path
import torch
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ProductDetector:
    """A class for detecting products in images using YOLOv9."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the ProductDetector.

        Args:
            model_path (str, optional): Path to YOLO model weights
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = self._load_model(model_path)

        logger.info(f"Initialized ProductDetector on {device}")

    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """Load the YOLO model."""
        try:
            # Check if model_path is provided and exists
            if model_path and Path(model_path).exists():
                logger.info(f"Loading custom model from: {model_path}")
                model = YOLO(model_path)
            else:
                # If model not found, try downloading YOLOv8n
                logger.warning(f"Model not found at {model_path}, using YOLOv8n instead")
                model = YOLO('yolov8n.pt')

            model = model.to(self.device)
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Detect products in an image.

        Args:
            image (PIL.Image): Input image
            confidence_threshold (float, optional): Override default confidence threshold

        Returns:
            List[np.ndarray]: List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        try:
            # Ensure we're working with a PIL Image
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                else:
                    raise TypeError("Image must be a PIL Image or numpy array")

            # Convert PIL Image to numpy array while preserving orientation
            img_array = np.array(image)

            # Run inference
            results = self.model(img_array)[0]

            # Process results
            detections = []
            for r in results.boxes.data.cpu().numpy():
                if r[4] >= confidence_threshold:
                    detections.append(r)

            logger.info(f"Found {len(detections)} products in image")
            return detections

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise

    def detect_batch(
        self,
        images: List[Image.Image],
        confidence_threshold: Optional[float] = None,
        batch_size: int = 16
    ) -> List[List[np.ndarray]]:
        """
        Detect products in a batch of images.

        Args:
            images (List[PIL.Image]): List of input images
            confidence_threshold (float, optional): Override default confidence threshold
            batch_size (int): Batch size for processing

        Returns:
            List[List[np.ndarray]]: List of detections for each image
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        try:
            all_detections = []

            # Process in batches
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_arrays = [np.array(img) for img in batch]

                # Run inference on batch
                results = self.model(batch_arrays)

                # Process results for each image in batch
                for r in results:
                    detections = []
                    for box in r.boxes.data.cpu().numpy():
                        if box[4] >= confidence_threshold:
                            detections.append(box)
                    all_detections.append(detections)

            logger.info(f"Processed {len(images)} images in batches")
            return all_detections

        except Exception as e:
            logger.error(f"Error during batch detection: {str(e)}")
            raise