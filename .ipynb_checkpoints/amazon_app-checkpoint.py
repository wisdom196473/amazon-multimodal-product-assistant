import streamlit as st

# Configure page
st.set_page_config(
    page_title="E-commerce Visual Assistant",
    page_icon="🛍️",
    layout="wide"
)

from streamlit_chat import message
import torch
from PIL import Image
import requests
from io import BytesIO
from model import initialize_models, load_data, chatbot, cleanup_resources

# Helper functions
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

def initialize_assistant():
    if not st.session_state.models_loaded:
        with st.spinner("Loading models and data..."):
            initialize_models()
            load_data()
            st.session_state.models_loaded = True
        st.success("Assistant is ready!")

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"], caption="Uploaded Image", width=200)
            if "display_images" in message:
                # Since we only have one image, we don't need multiple columns
                img_data = message["display_images"][0]  # Get the first (and only) image
                st.image(
                    img_data['image'],
                    caption=f"{img_data['product_name']}\nPrice: ${img_data['price']:.2f}",
                    width=350  # Adjusted width for single image display
                )

def handle_user_input(prompt, uploaded_image):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Processing your request..."):
        try:
            response = chatbot(prompt, image_input=uploaded_image)
            
            if isinstance(response, dict):
                assistant_message = {
                    "role": "assistant",
                    "content": response['text']
                }
                if 'images' in response and response['images']:
                    assistant_message["display_images"] = response['images']
                st.session_state.messages.append(assistant_message)
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}"
            })
    
    st.rerun()

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8edf2 100%);
            padding: 20px;
            border-radius: 15px;
        }
        
        /* Header styling */
        .stTitle {
            color: #1e3d59;
            font-size: 2.5rem !important;
            text-align: center;
            padding: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e3d59 0%, #2b5876 100%);
        }
        
        /* Chat container styling */
        .stChatMessage {
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
            padding: 15px;
        }
        
        /* Input box styling */
        .stTextInput > div > div > input {
            border-radius: 20px;
            border: 2px solid #1e3d59;
            padding: 10px 20px;
        }
        
        /* Radio button styling */
        .stRadio > label {
            background-color: white;
            padding: 10px 20px;
            border-radius: 10px;
            margin: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #1e3d59 0%, #2b5876 100%);
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        }
        
        /* Footer styling */
        footer {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Main title with enhanced styling
st.markdown("<h1 class='stTitle'>🛍️ Amazon E-commerce Visual Assistant</h1>", unsafe_allow_html=True)

# Sidebar configuration with enhanced styling
with st.sidebar:
    st.title("Assistant Features")
    
    st.markdown("### 🤖 How It Works")
    st.markdown("""
    This AI-powered shopping assistant combines:
    
    **🧠 Advanced Technologies**
    - FashionCLIP Visual AI
    - Mistral-7B Language Model
    - Multimodal Understanding
    
    **💫 Capabilities**
    - Product Search & Recognition
    - Visual Analysis
    - Detailed Comparisons
    - Price Analysis
    """)
    
    st.markdown("---")
    
    st.markdown("### 👥 Development Team")
    team_members = {
        "Yu-Chih (Wisdom) Chen",
        "Feier Xu",
        "Yanchen Dong",
        "Kitae Kim"
    }
    
    for name in team_members:
        st.markdown(f"**{name}**")
    
    st.markdown("---")
    
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
def main():
    # Initialize assistant
    initialize_assistant()
    
    # Chat container
    chat_container = st.container()
    
    # User input section at the bottom
    input_container = st.container()
    
    with input_container:
        # Chat input
        prompt = st.chat_input("What would you like to know?")
        
        # Input options below chat input
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            input_option = st.radio(
                "Input Method:",
                ("Text Only", "Upload Image", "Image URL"),
                key="input_method"
            )
        
        # Handle different input methods
        uploaded_image = None
        if input_option == "Upload Image":
            with col2:
                uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    uploaded_image = Image.open(uploaded_file)
                    st.image(uploaded_image, caption="Uploaded Image", width=200)
        
        elif input_option == "Image URL":
            with col2:
                image_url = st.text_input("Enter image URL")
                if image_url:
                    uploaded_image = load_image_from_url(image_url)
                    if uploaded_image:
                        st.image(uploaded_image, caption="Image from URL", width=200)
    
    # Display chat history
    with chat_container:
        display_chat_history()
    
    # Handle user input and generate response
    if prompt:
        handle_user_input(prompt, uploaded_image)

    # Footer
    st.markdown("""
    <footer>
        <h3>💡 Tips for Best Results</h3>
        <p>Be specific in your questions for more accurate responses!</p>
        <p>Try asking about product features, comparisons, or prices.</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_resources()
