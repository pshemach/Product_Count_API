from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prodcutCount",
    version="0.1.0",
    author="pasindu",
    author_email="your.email@example.com",
    description="A computer vision-based API for counting products in supermarket images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Product_Count_API",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.115.11",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.26.4",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "faiss-cpu>=1.10.0",
        "ultralytics>=8.3.12",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
    },
)
