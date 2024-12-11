# Standard libraries
import streamlit as st
import os
import io
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Deep learning frameworks
import torch
from torch.cuda.amp import autocast
import open_clip

# Hugging Face
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)
from huggingface_hub import hf_hub_download, login
from langchain.prompts import PromptTemplate

# Vector database
import faiss

# Type hints
from typing import Dict, List, Tuple, Optional, Union

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model: Optional[PreTrainedModel] = None
clip_preprocess: Optional[callable] = None
clip_tokenizer: Optional[PreTrainedTokenizer] = None
llm_tokenizer: Optional[PreTrainedTokenizer] = None
llm_model: Optional[PreTrainedModel] = None
product_df: Optional[pd.DataFrame] = None
metadata: Dict = {}
embeddings_df: Optional[pd.DataFrame] = None
text_faiss: Optional[object] = None
image_faiss: Optional[object] = None

def initialize_models() -> bool:
    global clip_model, clip_preprocess, clip_tokenizer, llm_tokenizer, llm_model, device
    
    try:
        print(f"Initializing models on device: {device}")
        
        # Initialize CLIP model with error handling
        try:
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                'hf-hub:Marqo/marqo-fashionCLIP'
            )
            # Use to_empty() first, then move to device
            clip_model = clip_model.to_empty(device=device)
            clip_model = clip_model.to(device)
            clip_model.eval()
            clip_tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')
            print("CLIP model initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {str(e)}")

        # Initialize LLM with optimized settings
        try:
            model_name = "mistralai/Mistral-7B-v0.1"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Get token from Streamlit secrets
            hf_token = st.secrets["HUGGINGFACE_TOKEN"]

            llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                truncation_side="left",
                token=hf_token  # Add token here
            )
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                token=hf_token  # Add token here
            )
            llm_model.eval()
            print("LLM initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

        return True

    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {str(e)}")

# Data loading
def load_data() -> bool:
    """
    Load and initialize all required data with enhanced metadata support and error handling.
    
    Returns:
        bool: True if data loading successful, raises RuntimeError otherwise
    """
    global product_df, metadata, embeddings_df, text_faiss, image_faiss

    try:
        print("Loading product data...")
        # Load cleaned product data
        try:
            cleaned_data_path = hf_hub_download(
                repo_id="chen196473/amazon_product_2020_cleaned",
                filename="amazon_cleaned.parquet",
                repo_type="dataset"
            )
            product_df = pd.read_parquet(cleaned_data_path)
            
            # Add validation columns
            product_df['Has_Valid_Image'] = product_df['Processed Image'].notna()
            product_df['Image_Status'] = product_df['Has_Valid_Image'].map({
                True: 'valid',
                False: 'invalid'
            })
            print("Product data loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load product data: {str(e)}")

        # Load enhanced metadata
        print("Loading metadata...")
        try:
            metadata = {}
            metadata_files = [
                'base_metadata.json',
                'category_index.json',
                'price_range_index.json',
                'keyword_index.json',
                'brand_index.json',
                'product_name_index.json'
            ]
            
            for file in metadata_files:
                file_path = hf_hub_download(
                    repo_id="chen196473/amazon_product_2020_metadata",
                    filename=file,
                    repo_type="dataset"
                )
                with open(file_path, 'r') as f:
                    index_name = file.replace('.json', '')
                    data = json.load(f)
                    
                    if index_name == 'base_metadata':
                        data = {item['Uniq_Id']: item for item in data}
                        for item in data.values():
                            if 'Keywords' in item:
                                item['Keywords'] = set(item['Keywords'])
                    
                    metadata[index_name] = data
            print("Metadata loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {str(e)}")

        # Load embeddings
        print("Loading embeddings...")
        try:
            text_embeddings_dict, image_embeddings_dict = load_embeddings_from_huggingface(
                "chen196473/amazon_vector_database"
            )
            
            # Create embeddings DataFrame
            embeddings_df = pd.DataFrame({
                'text_embeddings': list(text_embeddings_dict.values()),
                'image_embeddings': list(image_embeddings_dict.values()),
                'Uniq_Id': list(text_embeddings_dict.keys())
            })

            # Merge with product data
            product_df = product_df.merge(
                embeddings_df,
                left_on='Uniq Id',
                right_on='Uniq_Id',
                how='inner'
            )
            print("Embeddings loaded and merged successfully")
            
            # Create FAISS indexes
            print("Creating FAISS indexes...")
            try:
                create_faiss_indexes(text_embeddings_dict, image_embeddings_dict)
                print("FAISS indexes created successfully")
                
                # Verify FAISS indexes are properly initialized and contain data
                if text_faiss is None or image_faiss is None:
                    raise RuntimeError("FAISS indexes were not properly initialized")
                
                # Test a simple query to verify indexes are working
                test_query = "test"
                tokens = clip_tokenizer(test_query).to(device)
                with torch.no_grad():
                    text_embedding = clip_model.encode_text(tokens)
                    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                    text_embedding = text_embedding.cpu().numpy()
                
                # Verify search works
                test_results = text_faiss.search(text_embedding[0], k=1)
                if not test_results:
                    raise RuntimeError("FAISS indexes are empty")
                    
                print("FAISS indexes verified successfully")
                
            except Exception as e:
                raise RuntimeError(f"Failed to create or verify FAISS indexes: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings: {str(e)}")

        # Validate required columns
        required_columns = [
            'Uniq Id', 'Product Name', 'Category', 'Selling Price',
            'Model Number', 'Image', 'Normalized Description'
        ]
        missing_cols = set(required_columns) - set(product_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add enhanced metadata fields
        if 'Search_Text' not in product_df.columns:
            product_df['Search_Text'] = product_df.apply(
                lambda x: metadata['base_metadata'].get(x['Uniq Id'], {}).get('Search_Text', ''),
                axis=1
            )

        # Final verification of loaded data
        if product_df is None or product_df.empty:
            raise RuntimeError("Product DataFrame is empty or not initialized")
        
        if not metadata:
            raise RuntimeError("Metadata dictionary is empty")
        
        if embeddings_df is None or embeddings_df.empty:
            raise RuntimeError("Embeddings DataFrame is empty or not initialized")

        print("Data loading completed successfully")
        return True

    except Exception as e:
        # Clean up any partially loaded data
        product_df = None
        metadata = {}
        embeddings_df = None
        text_faiss = None
        image_faiss = None
        raise RuntimeError(f"Data loading failed: {str(e)}")

def load_embeddings_from_huggingface(repo_id: str) -> Tuple[Dict, Dict]:
    """
    Load embeddings from Hugging Face repository with enhanced error handling.
    
    Args:
        repo_id (str): Hugging Face repository ID
        
    Returns:
        Tuple[Dict, Dict]: Dictionaries containing text and image embeddings
    """
    print("Loading embeddings from Hugging Face...")
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename="embeddings.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(file_path)
        
        # Extract embedding columns
        text_cols = [col for col in df.columns if col.startswith('text_embedding_')]
        image_cols = [col for col in df.columns if col.startswith('image_embedding_')]
        
        # Create embedding dictionaries
        text_embeddings_dict = {
            row['Uniq_Id']: row[text_cols].values.astype(np.float32) 
            for _, row in df.iterrows()
        }
        image_embeddings_dict = {
            row['Uniq_Id']: row[image_cols].values.astype(np.float32) 
            for _, row in df.iterrows()
        }
        
        print(f"Successfully loaded {len(text_embeddings_dict)} embeddings")
        return text_embeddings_dict, image_embeddings_dict
    
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from Hugging Face: {str(e)}")

# FAISS index creation
class MultiModalFAISSIndex:
    def __init__(self, dimension, index_type='L2'):
        import faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension) if index_type == 'L2' else faiss.IndexFlatIP(dimension)
        self.id_to_metadata = {}
        
    def add_embeddings(self, embeddings, metadata_list):
        import numpy as np
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        for i, metadata in enumerate(metadata_list):
            self.id_to_metadata[i] = metadata
            
    def search(self, query_embedding, k=5):
        import numpy as np
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx in self.id_to_metadata:
                results.append(self.id_to_metadata[idx])
        return results

def create_faiss_indexes(text_embeddings_dict, image_embeddings_dict):
    """Create FAISS indexes with error handling"""
    global text_faiss, image_faiss
    
    try:
        # Get embedding dimension
        text_dim = next(iter(text_embeddings_dict.values())).shape[0]
        image_dim = next(iter(image_embeddings_dict.values())).shape[0]
        
        # Create indexes
        text_faiss = MultiModalFAISSIndex(text_dim)
        image_faiss = MultiModalFAISSIndex(image_dim)
        
        # Prepare text embeddings and metadata
        text_embeddings = []
        text_metadata = []
        for text_id, embedding in text_embeddings_dict.items():
            if text_id in product_df['Uniq Id'].values:
                product = product_df[product_df['Uniq Id'] == text_id].iloc[0]
                text_embeddings.append(embedding)
                text_metadata.append({
                    'id': text_id,
                    'description': product['Normalized Description'],
                    'product_name': product['Product Name']
                })
        
        # Add text embeddings
        if text_embeddings:
            text_faiss.add_embeddings(text_embeddings, text_metadata)
        
        # Prepare image embeddings and metadata
        image_embeddings = []
        image_metadata = []
        for image_id, embedding in image_embeddings_dict.items():
            if image_id in product_df['Uniq Id'].values:
                product = product_df[product_df['Uniq Id'] == image_id].iloc[0]
                image_embeddings.append(embedding)
                image_metadata.append({
                    'id': image_id,
                    'image_url': product['Image'],
                    'product_name': product['Product Name']
                })
        
        # Add image embeddings
        if image_embeddings:
            image_faiss.add_embeddings(image_embeddings, image_metadata)
            
        return True
        
    except Exception as e:
        raise RuntimeError(f"Failed to create FAISS indexes: {str(e)}")

def get_few_shot_product_comparison_template():
    return """Compare these specific products based on their actual features and specifications:

Example 1:
Question: Compare iPhone 13 and Samsung Galaxy S21
Answer: The iPhone 13 features a 6.1-inch Super Retina XDR display and dual 12MP cameras, while the Galaxy S21 has a 6.2-inch Dynamic AMOLED display and triple camera setup. Both phones offer 5G connectivity, but the iPhone uses A15 Bionic chip while S21 uses Snapdragon 888.

Example 2:
Question: Compare Amazon Echo Dot and Google Nest Mini
Answer: The Amazon Echo Dot features Alexa voice assistant and a 1.6-inch speaker, while the Google Nest Mini comes with Google Assistant and a 40mm driver. Both devices offer smart home control and music playback, but differ in their ecosystem integration.

Current Question: {query}
Context: {context}

Guidelines:
- Only compare the specific products mentioned in the query
- Focus on actual product features and specifications
- Keep response to 2-3 clear sentences
- Ensure factual accuracy based on the context provided

Answer:"""

def get_zero_shot_product_template():
    return """You are a product information specialist. Describe only the specific product's actual features based on the provided context.

Context: {context}

Question: {query}

Guidelines:
- Only describe the specific product mentioned in the query
- Focus on actual features and specifications from the context
- Keep response to 2-3 factual sentences
- Ensure information accuracy

Answer:"""

def get_zero_shot_image_template():
    return """Analyze this product image and provide a concise description:

Product Information:
{context}

Guidelines:
- Describe the main product features and intended use
- Highlight key specifications and materials
- Keep response to 2-3 sentences
- Focus on practical information

Answer:"""

# Image processing functions
def process_image(image):
    try:
        if isinstance(image, str):
            response = requests.get(image)
            image = Image.open(io.BytesIO(response.content))
        
        processed_image = clip_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(processed_image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to fetch image from URL: {url}, Status Code: {response.status_code}")

# Context retrieval and enhancement
def filter_by_metadata(query, metadata_index):
    relevant_products = set()
    
    # Check category index
    if 'category_index' in metadata_index:
        categories = metadata_index['category_index']
        for category in categories:
            if any(term.lower() in category.lower() for term in query.split()):
                relevant_products.update(categories[category])
    
    # Check product name index
    if 'product_name_index' in metadata_index:
        product_names = metadata_index['product_name_index']
        for term in query.split():
            if term.lower() in product_names:
                relevant_products.update(product_names[term.lower()])
    
    # Check price ranges
    price_terms = {'cheap', 'expensive', 'price', 'cost', 'affordable'}
    if any(term in query.lower() for term in price_terms) and 'price_range_index' in metadata_index:
        price_ranges = metadata_index['price_range_index']
        for price_range in price_ranges:
            relevant_products.update(price_ranges[price_range])
    
    return relevant_products if relevant_products else None

def enhance_context_with_metadata(product, metadata_index):
    """Enhanced context building using new metadata structure"""
    # Access base_metadata using product ID directly since it's now a dictionary
    base_metadata = metadata_index['base_metadata'].get(product['Uniq Id'])
    
    if base_metadata:
        # Get keywords and search text from enhanced metadata
        keywords = base_metadata.get('Keywords', [])
        search_text = base_metadata.get('Search_Text', '')
        
        # Build enhanced description
        description = []
        description.append(f"Product Name: {base_metadata['Product_Name']}")
        description.append(f"Category: {base_metadata['Category']}")
        description.append(f"Price: ${base_metadata['Selling_Price']:.2f}")
        
        # Add key features from normalized description
        if 'Normalized_Description' in base_metadata:
            features = []
            for feature in base_metadata['Normalized_Description'].split('|'):
                if ':' in feature:
                    key, value = feature.split(':', 1)
                    if not any(skip in key.lower() for skip in 
                        ['uniq id', 'product url', 'specifications', 'asin']):
                        features.append(f"{key.strip()}: {value.strip()}")
            if features:
                description.append("Key Features:")
                description.extend(features[:3])
        
        # Add relevant keywords
        if keywords:
            description.append("Related Terms: " + ", ".join(list(keywords)[:5]))
        
        return "\n".join(description)
    
    return None

def retrieve_context(query, image=None, top_k=5):
    """Enhanced context retrieval using both FAISS and metadata"""
    # Initialize context lists
    similar_items = []
    context = []
    
    if image is not None:
        # Process image query
        image_embedding = process_image(image)
        image_embedding = image_embedding.reshape(1, -1)
        similar_items = image_faiss.search(image_embedding[0], k=top_k)
    else:
        # Process text query with enhanced metadata filtering
        relevant_products = filter_by_metadata(query, metadata)
        
        tokens = clip_tokenizer(query).to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy()
        
        # Get FAISS results
        similar_items = text_faiss.search(text_embedding[0], k=top_k*2)  # Get more results for filtering
        
        # Filter results using metadata if available
        if relevant_products:
            similar_items = [item for item in similar_items if item['id'] in relevant_products][:top_k]
    
    # Build enhanced context
    for item in similar_items:
        product = product_df[product_df['Uniq Id'] == item['id']].iloc[0]
        enhanced_context = enhance_context_with_metadata(product, metadata)
        if enhanced_context:
            context.append(enhanced_context)
    
    return "\n\n".join(context), similar_items

def display_product_images(similar_items, max_images=1):
    displayed_images = []
    
    for item in similar_items[:max_images]:
        try:
            # Get image URL from product data
            image_url = item['Image'] if isinstance(item, pd.Series) else item.get('Image')
            if not image_url:
                continue
                
            # Handle multiple image URLs
            image_urls = image_url.split('|')
            image_url = image_urls[0]  # Take first image
            
            # Load image
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            
            # Get product details
            product_name = item['Product Name'] if isinstance(item, pd.Series) else item.get('product_name')
            price = item['Selling Price'] if isinstance(item, pd.Series) else item.get('price', 0)
            
            # Add to displayed images
            displayed_images.append({
                'image': img,
                'product_name': product_name,
                'price': float(price)
            })
            
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue
    
    return displayed_images

def classify_query(query):
    """Classify the type of query to determine the retrieval strategy."""
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ['compare', 'difference between']):
        return 'comparison'
    elif any(keyword in query_lower for keyword in ['show', 'picture', 'image', 'photo']):
        return 'image_search'
    else:
        return 'product_info'

def boost_category_relevance(query, product, similarity_score):
    query_terms = set(query.lower().split())
    category_terms = set(product['Category'].lower().split())
    category_overlap = len(query_terms & category_terms)
    category_boost = 1 + (category_overlap * 0.2)  # 20% boost per matching term
    return similarity_score * category_boost

def hybrid_retrieval(query, top_k=5):
    query_type = classify_query(query)
    
    tokens = clip_tokenizer(query).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy()

    # First get text matches
    text_results = text_faiss.search(text_embedding[0], k=top_k*2)
    
    if query_type == 'image_search':
        image_results = []
        for item in text_results:
            # Get original product with embeddings intact
            product = product_df[product_df['Uniq Id'] == item['id']].iloc[0]
            # Get image embeddings from embeddings_df instead
            image_embedding = embeddings_df[embeddings_df['Uniq_Id'] == item['id']]['image_embeddings'].iloc[0]
            similarity = np.dot(text_embedding.flatten(), image_embedding.flatten())
            boosted_similarity = boost_category_relevance(query, product, similarity)
            image_results.append((product, boosted_similarity))
        
        image_results.sort(key=lambda x: x[1], reverse=True)
        results = [item for item, _ in image_results[:top_k]]
    else:
        results = [product_df[product_df['Uniq Id'] == item['id']].iloc[0] for item in text_results[:top_k]]

    return results, query_type

def fallback_text_search(query, top_k=10):
    relevant_products = filter_by_metadata(query, metadata)
    if not relevant_products:
        # Check brand index specifically
        if 'brand_index' in metadata:
            query_terms = query.lower().split()
            for term in query_terms:
                if term in metadata['brand_index']:
                    relevant_products = set(metadata['brand_index'][term])
                    break
    
    if relevant_products:
        results = [product_df[product_df['Uniq Id'] == pid].iloc[0] for pid in list(relevant_products)[:top_k]]
    else:
        query_lower = query.lower()
        results = product_df[
            (product_df['Product Name'].str.lower().str.contains(query_lower)) |
            (product_df['Category'].str.lower().str.contains(query_lower)) |
            (product_df['Normalized Description'].str.lower().str.contains(query_lower))
        ].head(top_k)

    return results

def generate_rag_response(query, context, image=None):
    """Enhanced RAG response generation"""
    # Select template based on query type and metadata
    if "compare" in query.lower() or "difference between" in query.lower() or "vs." in query.lower():
        template = get_few_shot_product_comparison_template()
    elif image is not None:
        template = get_zero_shot_image_template()
    else:
        template = get_zero_shot_product_template()
    
    # Create enhanced prompt with metadata context
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "context"]
    )
    
    # Configure generation parameters
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=300,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.2,
        early_stopping=True,
        truncation=True,
        padding=True
    )
    
    # Generate and clean response
    formatted_prompt = prompt.format(query=query, context=context)
    response = pipe(formatted_prompt)[0]['generated_text']
    
    # Clean response
    for section in ["Answer:", "Question:", "Guidelines:", "Context:"]:
        if section in response:
            response = response.split(section)[-1].strip()
    
    return response

def chatbot(query, image_input=None):
    """
    Main chatbot function to handle queries and provide responses.
    """
    if image_input is not None:
        try:
            # Convert URL to image if needed
            if isinstance(image_input, str):
                image_input = load_image_from_url(image_input)
            elif not isinstance(image_input, Image.Image):
                raise ValueError("Invalid image input type")
            
            # Get context and generate response
            context, _ = retrieve_context(query, image_input)
            if not context:
                return "No relevant products found for this image."
            response = generate_rag_response(query, context, image_input)
            return response
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return f"Failed to process image: {str(e)}"
    else:
        try:
            print(f"Processing query: {query}")
            if text_faiss is None or image_faiss is None:
                return "Search indexes not initialized. Please try again."
                
            results, query_type = hybrid_retrieval(query)
            print(f"Query type: {query_type}")

            if not results and query_type == 'image_search':
                print("No relevant images found. Falling back to text search.")
                results = fallback_text_search(query)

            if not results:
                return "No relevant products found."

            context = "\n\n".join([enhance_context_with_metadata(item, metadata) for item in results])
            response = generate_rag_response(query, context)

            if query_type == 'image_search':
                print("\nFound matching products:")
                displayed_images = display_product_images(results)
                
                # Always return a dictionary with both text and images for image search queries
                return {
                    'text': response,
                    'images': displayed_images
                }

            return response
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing request: {str(e)}"

def cleanup_resources():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
