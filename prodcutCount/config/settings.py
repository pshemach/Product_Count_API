"""
Settings Module
-------------

This module provides configuration settings for the product counting API.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Configuration settings for the application."""
    
    # Base paths
    base_dir: str = str(Path(__file__).parent.parent.parent)
    
    # Model settings
    model_name: str = "obj_detect_yolov9.pt"  # Default model name
    model_path: Optional[str] = str(Path(base_dir) / "data" / "models" / model_name)  # Path to YOLOv9 model weights
    fallback_model: str = "yolov8n.pt"  # Fallback to YOLOv8n if custom model not found
    confidence_threshold: float = 0.25  # Detection confidence threshold
    similarity_threshold: float = 0.75  # Product matching similarity threshold
    
    # Data paths
    data_dir: str = str(Path(base_dir) / "data")
    reference_images_dir: str = str(Path(data_dir) / "reference_products")
    models_dir: str = str(Path(data_dir) / "models")
    
    # API settings
    api_title: str = "Product Count API"
    api_description: str = "API for counting products in supermarket images"
    api_version: str = "0.1.0"
    
    # Processing settings
    batch_size: int = 16
    max_image_size: int = 1024  # Maximum image dimension
    
    # Device settings
    device: str = "cuda"  # or "cpu"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False 