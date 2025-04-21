"""
Product Detection Demo
--------------------
Streamlit interface for product detection and reference management
"""

import os
# Set OpenMP environment variable to avoid runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import io

from prodcutCount.core.detector import ProductDetector
from prodcutCount.core.matcher import ProductMatcher
from prodcutCount.config.settings import Settings
from prodcutCount.utils.reference_manager import ReferenceManager
from prodcutCount.config.product_mapping import load_mapping

# Disable PyTorch's default multithreading to avoid conflicts
torch.set_num_threads(1)

# Page config
st.set_page_config(
    page_title="Product Count Demo",
    page_icon="🛒",
    layout="wide"
)

def prepare_reference_directory(settings: Settings, ref_manager: ReferenceManager) -> str:
    """Prepare reference directory with proper structure for matcher"""
    # Create a temporary directory for organized references
    temp_ref_dir = Path(settings.reference_images_dir) / "temp_organized"
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    temp_ref_dir.mkdir(parents=True, exist_ok=True)

    # Get current mappings
    ref_dir = Path(settings.reference_images_dir)

    try:
        # Organize images by product name
        for ref_id, product_name in ref_manager.get_all_references().items():
            # Clean product name for directory (replace spaces with underscores)
            safe_product_name = product_name.replace(" ", "_")

            # Create product directory
            product_dir = temp_ref_dir / safe_product_name
            product_dir.mkdir(exist_ok=True)

            # Source directory for this reference
            src_dir = ref_dir / ref_id
            if src_dir.exists() and src_dir.is_dir():
                # Copy all jpg files from the source directory
                for img_file in src_dir.glob("*.jpg"):
                    dest_file = product_dir / img_file.name
                    shutil.copy2(str(img_file), str(dest_file))
            else:
                # Check for single file in reference directory
                src_file = ref_dir / f"{ref_id}.jpg"
                if src_file.exists():
                    dest_file = product_dir / f"{ref_id}.jpg"
                    shutil.copy2(str(src_file), str(dest_file))

        return str(temp_ref_dir)

    except Exception as e:
        if temp_ref_dir.exists():
            shutil.rmtree(temp_ref_dir)
        raise Exception(f"Error preparing reference directory: {str(e)}")

def draw_detections(image: np.ndarray, product_matches: list) -> np.ndarray:
    """Draw detection boxes and labels on the image"""
    img = image.copy()

    for name, sim, (x1, y1, x2, y2) in product_matches:
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label
        label = f"{name}: {sim:.2f}"

        # Get label size
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            (0, 255, 0),
            -1
        )

        # Draw label text
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

    return img

@st.cache_resource
def initialize_app():
    """Initialize detector, matcher and reference manager"""
    settings = Settings()

    with st.spinner('Loading detection model...'):
        detector = ProductDetector(
            model_path=settings.model_path,
            confidence_threshold=settings.confidence_threshold
        )

    with st.spinner('Loading matching model...'):
        matcher = ProductMatcher(
            similarity_threshold=settings.similarity_threshold
        )

    ref_manager = ReferenceManager(settings.reference_images_dir)

    # Prepare reference directory and load references
    with st.spinner('Loading reference images...'):
        try:
            # Clean up any existing temp directory
            temp_dir = Path(settings.reference_images_dir) / "temp_organized"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Prepare and load references
            ref_dir = prepare_reference_directory(settings, ref_manager)
            if not Path(ref_dir).exists():
                raise ValueError("Reference directory not created properly")

            matcher.add_reference_images(ref_dir)

            # Clean up temporary directory
            shutil.rmtree(ref_dir)

        except Exception as e:
            st.error(f"Error loading reference images: {str(e)}")
            raise  # Re-raise to prevent continuing with uninitialized matcher

    return detector, matcher, ref_manager

def main():
    st.title("🛒 Product Count Demo")
    st.markdown("""
    This demo allows you to:
    - Upload and manage reference product images
    - Detect and count products in images
    - Match products with references
    """)

    # Initialize components
    detector, matcher, ref_manager = initialize_app()

    # Sidebar for settings and reference management
    with st.sidebar:
        st.header("Settings")

        # Detection settings
        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=detector.confidence_threshold,
            step=0.05,
            help="Minimum confidence threshold for object detection"
        )

        # Matching settings
        similarity_threshold = st.slider(
            "Matching Similarity",
            min_value=0.0,
            max_value=1.0,
            value=matcher.similarity_threshold,
            step=0.05,
            help="Minimum similarity threshold for product matching"
        )

        st.divider()
        st.header("Reference Image Management")

        uploaded_ref = st.file_uploader(
            "Upload Reference Image",
            type=["jpg", "jpeg", "png"],
            key="ref_upload"
        )

        product_name = st.text_input(
            "Product Name",
            key="product_name",
            help="Enter the name for the reference product"
        )

        if st.button("Add Reference Image") and uploaded_ref and product_name:
            # Read file bytes directly
            file_bytes = uploaded_ref.getvalue()

            # Display the reference image as-is
            st.image(file_bytes, caption=f"Reference image for {product_name}", width=200)

            # Use PIL to open the image for processing
            pil_image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

            # Convert to OpenCV format for the reference manager
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            try:
                new_id = ref_manager.add_reference_image(image, product_name)
                st.success(f"Added reference image for {product_name}")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error adding reference image: {str(e)}")

        st.divider()
        st.subheader("Current References")

        references = ref_manager.get_all_references()
        for ref_id, name in references.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{name} (ID: {ref_id})")
            with col2:
                if st.button("Remove", key=f"remove_{ref_id}"):
                    ref_manager.remove_reference_image(ref_id)
                    st.cache_resource.clear()
                    st.rerun()

    # Main content area
    uploaded_file = st.file_uploader(
        "Choose an image to analyze",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing products to detect"
    )

    if uploaded_file is not None:
        try:
            # Display original and processed images
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                # Read the file content directly as bytes
                file_bytes = uploaded_file.getvalue()

                # Display the image directly from bytes to avoid any processing that might cause rotation
                st.image(file_bytes, use_container_width=True)

                # Open image with PIL for processing (not for display)
                image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

            # Process image
            with st.spinner('Processing image...'):
                detections = detector.detect(
                    image,
                    confidence_threshold=confidence_threshold
                )

                matcher.similarity_threshold = similarity_threshold
                product_counts, product_matches = matcher.count_products(
                    detections,
                    image,
                    min_confidence=confidence_threshold
                )

                with col2:
                    st.subheader("Detected Products")
                    # Use the original image for detection results
                    result_image = draw_detections(np.array(image), product_matches)
                    # Convert back to bytes for consistent display
                    result_pil = Image.fromarray(result_image)
                    result_bytes = io.BytesIO()
                    result_pil.save(result_bytes, format='JPEG')
                    # Display the result image directly from bytes
                    st.image(result_bytes.getvalue(), use_container_width=True)

            # Display results
            st.subheader("Results")

            # Metrics
            # col3, col4, col5 = st.columns(3)
            col4, col5 = st.columns(2)
            # with col3:
            #     st.metric("Total Products", sum(product_counts.values()))
            with col4:
                st.metric("Total Detections", len(detections))
            with col5:
                st.metric("Matched Products", len(product_matches))

            # Product breakdown
            if product_counts:
                st.subheader("Product Breakdown")

                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                products = list(product_counts.keys())
                counts = list(product_counts.values())

                ax.bar(products, counts)
                ax.set_ylabel("Count")
                ax.set_title("Product Counts")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                st.pyplot(fig)

                # Detailed table
                st.dataframe(
                    {
                        "Product": products,
                        "Count": counts
                    },
                    use_container_width=True
                )
            else:
                st.info("No products detected in the image.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
