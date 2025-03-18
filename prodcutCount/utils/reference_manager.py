"""
Reference Image Manager
----------------------
Handles reference image storage and mapping
"""

import json
import cv2
from pathlib import Path
import numpy as np
from prodcutCount.config.product_mapping import load_mapping, save_mapping

class ReferenceManager:
    def __init__(self, reference_dir: str):
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.id_to_name = load_mapping()

    def _load_mapping(self) -> dict:
        """Load existing mapping from file"""
        return load_mapping()

    def _save_mapping(self):
        """Save current mapping to file"""
        save_mapping(self.id_to_name)

    def add_reference_image(self, image: np.ndarray, product_name: str) -> str:
        """Add a new reference image and return its ID"""
        # Generate new ID if product doesn't exist, or use existing ID if it does
        existing_id = None
        for id, name in self.id_to_name.items():
            if name == product_name:
                existing_id = id
                break
        
        if existing_id is None:
            # Generate new ID for new product
            new_id = str(max([int(id) for id in self.id_to_name.keys()], default=999) + 1)
        else:
            new_id = existing_id
        
        # Create or use existing directory for this reference
        ref_dir = self.reference_dir / new_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename for the new image
        existing_files = list(ref_dir.glob("*.jpg"))
        next_number = len(existing_files) + 1
        dest_path = ref_dir / f"reference_{next_number}.jpg"
        
        # Save image to reference directory
        cv2.imwrite(str(dest_path), image)

        # Update mapping only if it's a new product
        if existing_id is None:
            self.id_to_name[new_id] = product_name
            save_mapping(self.id_to_name)
        
        return new_id

    def remove_reference_image(self, image_id: str):
        """Remove a reference image and its mapping"""
        if image_id in self.id_to_name:
            # Remove directory and all contents
            ref_dir = self.reference_dir / image_id
            if ref_dir.exists():
                import shutil
                shutil.rmtree(ref_dir)
            
            # Remove from mapping
            del self.id_to_name[image_id]
            save_mapping(self.id_to_name)

    def get_all_references(self) -> dict:
        """Get all reference image mappings"""
        return self.id_to_name

    def initialize_with_mapping(self, mapping: dict):
        """Initialize with an existing mapping"""
        self.id_to_name = mapping
        save_mapping(self.id_to_name) 