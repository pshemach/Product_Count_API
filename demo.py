"""
Streamlit Demo Application
-----------------------

A user-friendly interface for the Product Count API using Streamlit.
"""

import os
# Set OpenMP environment variable to avoid runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from prodcutCount.core.detector import ProductDetector
from prodcutCount.core.matcher import ProductMatcher
from prodcutCount.config.settings import Settings

# Disable PyTorch's default multithreading to avoid conflicts
torch.set_num_threads(1)

# Page config
st.set_page_config(
    page_title="Product Count Demo",
    page_icon="🛒",
    layout="wide"
)

# Initialize settings and models
@st.cache_resource(show_spinner=True)
def load_models():
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
        
        # Load reference images if available
        reference_dir = Path(settings.reference_images_dir)
        if reference_dir.exists():
            matcher.add_reference_images(str(reference_dir))
    
    return detector, matcher

def draw_detections(image: np.ndarray, product_matches: list) -> np.ndarray:
    """Draw detection boxes and labels on the image."""
    img = image.copy()
    
    # Draw each detection
    for name, sim, (x1, y1, x2, y2) in product_matches:
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{name}: {sim:.2f}"
        
        # Get label size for background
        (label_w, label_h), baseline = cv2.getTextSize(
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
            0.5,
            (0, 0, 0),
            2
        )
    
    return img

# Title and description
st.title("🛒 Product Count Demo")
st.markdown("""
This demo allows you to upload an image of products and get:
- Product detection using YOLOv9
- Product matching with reference images
- Product counts and details
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence threshold for object detection"
    )

    similarity_threshold = st.slider(
        "Matching Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Minimum similarity threshold for product matching"
    )

# Load models with error handling
try:
    with st.spinner('Initializing models...'):
        detector, matcher = load_models()
        matcher.similarity_threshold = similarity_threshold
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing products to analyze"
)

if uploaded_file is not None:
    try:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
        
        # Process image
        with st.spinner('Processing image...'):
            # Detect products
            detections = detector.detect(
                image,
                confidence_threshold=confidence_threshold
            )
            
            # Match and count products
            product_counts, product_matches = matcher.count_products(
                detections,
                image,
                min_confidence=confidence_threshold
            )
            
            # Visualize results
            with col2:
                st.subheader("Detected Products")
                
                # Convert image and draw detections
                img_array = np.array(image)
                result_image = draw_detections(img_array, product_matches)
                
                # Display result
                st.image(result_image, use_container_width=True)
                
                # Display product details in an expandable section
                with st.expander("View Product Details", expanded=False):
                    for name, sim, (x1, y1, x2, y2) in product_matches:
                        st.markdown(f"- **{name}** (Confidence: {sim:.2f})")
        
        # Display results
        st.subheader("Results")
        
        # Product counts in columns
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric(
                "Total Products",
                sum(product_counts.values()),
                help="Total number of products counted"
            )
        
        with col4:
            st.metric(
                "Total Detections",
                len(detections),
                help="Number of objects detected"
            )
        
        with col5:
            st.metric(
                "Matched Products",
                len(product_matches),
                help="Number of products successfully matched"
            )
        
        # Product breakdown
        st.subheader("Product Breakdown")
        if product_counts:
            # Create bar chart
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
