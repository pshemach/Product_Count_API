# Product Count API

A computer vision-based API for counting products in supermarket images using YOLOv9 object detection and feature matching.

## Features

- Product detection using YOLOv9
- Feature-based product matching using ResNet50 and FAISS
- Fast and efficient similarity search
- RESTful API endpoints for product counting
- Visualization tools for debugging

## Project Structure

```
Product_Count_API/
├── data/                      # Data directory
│   ├── reference_products/    # Reference product images
│   ├── test_images/          # Test images
│   └── models/               # Trained models
├── prodcutCount/             # Main package directory
│   ├── __init__.py
│   ├── api/                  # API related code
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── models.py
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── matcher.py
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── visualization.py
│   └── config/              # Configuration files
│       ├── __init__.py
│       └── settings.py
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_matcher.py
├── research/               # Research and experiments
│   └── product_matching_test.ipynb
├── docs/                  # Documentation
│   ├── api.md
│   └── setup.md
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup file
├── .gitignore           # Git ignore file
└── LICENSE              # License file
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Product_Count_API.git
cd Product_Count_API
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -e .
```

## Usage

1. Start the API server:

```bash
uvicorn prodcutCount.api.endpoints:app --reload
```

2. Send a POST request with an image:

```python
import requests

url = "http://localhost:8000/api/v1/count-products"
files = {"image": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Development

1. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Run tests:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
