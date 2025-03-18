"""
Product Name Mapping Configuration
--------------------------------
Maps folder IDs to product names
"""

from pathlib import Path
import json
from ..config.settings import Settings

def get_mapping_file() -> Path:
    """Get the path to the mapping file"""
    settings = Settings()
    return Path(settings.data_dir) / "config" / "product_mapping.json"

def save_mapping(mapping: dict):
    """Update the mapping file with new values"""
    mapping_file = get_mapping_file()
    # Ensure the config directory exists
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=4)

def load_mapping() -> dict:
    """Load the current mapping from file"""
    mapping_file = get_mapping_file()
    
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            return json.load(f)
    
    # Return empty mapping if file doesn't exist
    return {} 