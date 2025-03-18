"""
Product Matcher Module
-------------------

This module provides functionality for matching products using feature extraction
and similarity search.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import faiss
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ProductMatcher:
    """A class for matching products using feature extraction and similarity search."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "resnet50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the ProductMatcher.

        Args:
            similarity_threshold (float): Threshold for considering a match valid
            model_name (str): Name of the model to use for feature extraction
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.model = self._initialize_model(model_name)
        self.transform = self._get_transforms()
        self.reference_features = None
        self.reference_labels = []
        self.index = None
        
        logger.info(f"Initialized ProductMatcher with {model_name} on {device}")
    
    def _initialize_model(self, model_name: str) -> nn.Module:
        """Initialize the feature extraction model."""
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from an image using the model.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            np.ndarray: Feature vector
        """
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            features = features.squeeze().cpu().numpy()
        return features
    
    def add_reference_images(self, reference_dir: str):
        """
        Add reference images for product matching.
        
        Args:
            reference_dir (str): Directory containing reference images organized by product
        """
        features_list = []
        reference_path = Path(reference_dir)
        
        for product_dir in reference_path.iterdir():
            if product_dir.is_dir():
                product_name = product_dir.name
                for img_path in product_dir.glob("*.jpg"):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        features = self.extract_features(img)
                        features_list.append(features)
                        self.reference_labels.append(product_name)
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {str(e)}")
        
        if not features_list:
            raise ValueError("No valid reference images found")
        
        self.reference_features = np.vstack(features_list)
        
        # Create FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.reference_features.shape[1])
        faiss.normalize_L2(self.reference_features)
        self.index.add(self.reference_features)
        
        logger.info(f"Added {len(self.reference_labels)} reference images")
    
    def match_product(self, crop_image: Image.Image) -> Tuple[str, float]:
        """
        Match a single product image with reference database.
        
        Args:
            crop_image (PIL.Image): Cropped image of a single product
            
        Returns:
            Tuple[str, float]: Product name and similarity score
        """
        if self.index is None:
            raise RuntimeError("No reference images added. Call add_reference_images first.")
        
        features = self.extract_features(crop_image)
        features = features.reshape(1, -1)
        faiss.normalize_L2(features)
        
        # Find nearest neighbor
        similarities, indices = self.index.search(features, 1)
        similarity = similarities[0][0]
        
        if similarity >= self.similarity_threshold:
            return self.reference_labels[indices[0][0]], similarity
        return "unknown", similarity
    
    def count_products(
        self,
        detections: List[np.ndarray],
        image: Image.Image,
        min_confidence: float = 0.5
    ) -> Dict[str, int]:
        """
        Count products from YOLO detections.
        
        Args:
            detections (List[np.ndarray]): List of YOLO detections
            image (PIL.Image): Original image
            min_confidence (float): Minimum confidence for YOLO detections
            
        Returns:
            Dict[str, int]: Dictionary of product counts
        """
        product_counts = {}
        product_matches = []
        
        for bbox in detections:
            if bbox[4] < min_confidence:  # Skip low confidence detections
                continue
                
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
                crop = image.crop((x1, y1, x2, y2))
                product_name, similarity = self.match_product(crop)
                
                if product_name != "unknown":
                    product_counts[product_name] = product_counts.get(product_name, 0) + 1
                    product_matches.append((product_name, similarity, (x1, y1, x2, y2)))
            except Exception as e:
                logger.error(f"Error processing detection: {str(e)}")
        
        return product_counts, product_matches 