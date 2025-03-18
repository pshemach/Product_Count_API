"""
Product Name Mapping Configuration
--------------------------------
Maps folder IDs to product names
"""

from pathlib import Path
import json

def get_mapping_file() -> Path:
    """Get the path to the mapping file"""
    return Path(__file__).parent / 'product_mapping.json'

def save_mapping(mapping: dict):
    """Update the mapping file with new values"""
    mapping_file = get_mapping_file()
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