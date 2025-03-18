"""
Verify Setup Script
----------------

This script verifies that the model and data paths are set up correctly.
"""

from pathlib import Path
import logging
from prodcutCount.config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_paths():
    """Verify that all required paths exist."""
    settings = Settings()
    
    # Check model path
    model_path = Path(settings.model_path)
    if model_path.exists():
        logger.info(f"✅ Found model at: {model_path}")
    else:
        logger.warning(f"❌ Model not found at: {model_path}")
        logger.info(f"Will fall back to: {settings.fallback_model}")
    
    # Check data directories
    data_paths = {
        "Data directory": settings.data_dir,
        "Models directory": settings.models_dir,
        "Reference images": settings.reference_images_dir
    }
    
    for name, path in data_paths.items():
        path = Path(path)
        if path.exists():
            logger.info(f"✅ {name} exists at: {path}")
        else:
            logger.warning(f"❌ {name} not found at: {path}")

if __name__ == "__main__":
    verify_paths() 