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

A multimodal AI assistant leveraging the Amazon Product Dataset 2020 to provide comprehensive product search and recommendations through natural language and image-based interactions[1].

## Project Overview

This conversational AI system combines advanced language and vision models to enhance e-commerce customer support, enabling accurate, context-aware responses to product-related queries[1].

## Project Structure

- `amazon_app.py`: Main Streamlit application
- `model.py`: Core AI model implementations
- `Vision_AI.ipynb`: EDA, Embedding Model, LLM
- `requirements.txt`: Project dependencies

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

## Technical Architecture

### Data Processing & Storage
- Standardized text fields and normalized numeric attributes
- Enhanced metadata indices for categories, price ranges, keywords, brands
- Validated image quality and managed duplicates
- Structured data storage in Parquet format[1]

### Model Components
- **Vision-Language Integration**: FashionCLIP for multimodal embedding generation
- **Vector Search**: FAISS with hybrid retrieval combining embedding similarity and metadata filtering
- **Language Model**: Mistral-7B with 4-bit quantization
- **RAG Framework**: Context-enhanced response generation[1]

### Performance Metrics
- Recall@1: 0.6385
- Recall@10: 0.9008
- Precision@1: 0.6385
- NDCG@10: 0.7725[1]

## Implementation Details

### Core Features
- Text and image-based product search
- Product comparisons and recommendations
- Visual product recognition
- Detailed product information retrieval
- Price analysis and comparison[1]

### Technologies Used
- FashionCLIP for visual understanding
- Mistral-7B Language Model (4-bit quantized)
- FAISS for similarity search
- Google Vertex AI for vector storage
- Streamlit for user interface[1]

## Challenges & Solutions

### Technical Challenges Addressed
- Image processing with varying quality
- GPU memory optimization
- Efficient embedding storage
- Query response accuracy[1]

### Implemented Solutions
- Robust image validation pipeline
- 4-bit model quantization
- Optimized batch processing
- Enhanced metadata enrichment[1]

## Future Directions

- [ ] Fine-Tune FashionClip embedding model based on the specific domain data
- [ ] Fine-Tune large language model to improve its generalization capabilities
- [ ] Develop feedback loops for continuous improvement