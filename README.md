---
title: Amazon E-commerce Visual Assistant
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: amazon_app.py
pinned: false
---

# Amazon E-commerce Visual Assistant

A multimodal AI assistant that helps users search and explore Amazon products through natural language and image-based interactions.

## Features

- Text and image-based product search
- Product comparisons and recommendations
- Visual product recognition
- Detailed product information retrieval
- Price analysis and comparison

## Technologies Used

- FashionCLIP for visual understanding
- Mistral-7B Language Model for text generation
- FAISS for efficient similarity search
- Streamlit for the user interface

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/wisdom196473/amazon-multimodal-product-assistant.git
cd amazon-multimodal-product-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run amazon_app.py
```

## Project Structure

- `amazon_app.py`: Main Streamlit application
- `model.py`: Core AI model implementations
- `requirements.txt`: Project dependencies

## Future Directions

- [ ] Fine-Tune FashionClip embedding model based on the specific domain data
- [ ] Fine-Tune large language model to improve its generalization capabilities
- [ ] Develop feedback loops for continuous improvement
