#!/usr/bin/env python3
"""
Streamlit UI for Fashion Similarity Analysis
"""

import streamlit as st
import os
import glob
from pathlib import Path
import pandas as pd
from PIL import Image

from vector_db import VectorDatabase
from embeddings import generate_image_embeddings

# Page configuration
st.set_page_config(
    page_title="Fashion Similarity Analysis",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_databases():
    """Load vector databases (cached for performance)."""
    dress_db = None
    full_db = None
    
    try:
        if os.path.exists('./analysis_dress_db'):
            dress_db = VectorDatabase('./analysis_dress_db', 'dress_embeddings')
        if os.path.exists('./analysis_full_db'):
            full_db = VectorDatabase('./analysis_full_db', 'full_image_embeddings')
    except Exception as e:
        st.error(f"Error loading databases: {e}")
    
    return dress_db, full_db

@st.cache_data
def load_sample_images():
    """Load available sample images (cached)."""
    return sorted(glob.glob('sample_images/output/*.png'))


def format_results(raw_results, search_type):
    """Format raw database results for display."""
    if not raw_results['ids'][0]:
        return []
    
    formatted = []
    for doc_id, metadata, distance in zip(
        raw_results['ids'][0],
        raw_results['metadatas'][0], 
        raw_results['distances'][0]
    ):
        similarity = 1 - distance
        
        # For dress-only results, use the segmented dress image if available
        if search_type == "Dress-Only" and metadata.get('extracted_path'):
            display_image_path = metadata.get('extracted_path', '')
        else:
            # For full-image results, use the original image
            display_image_path = metadata.get('image_path', '')
        
        formatted.append({
            'id': doc_id,
            'image_name': metadata['image_name'],
            'similarity': similarity,
            'distance': distance,
            'type': metadata.get('type', 'unknown'),
            'image_path': metadata.get('image_path', ''),  # Keep original for reference
            'display_image_path': display_image_path,  # Path to display (segmented or full)
        })
    
    return formatted

def query_similarity(image_path, dress_db, full_db, search_type="both", top_k=5):
    """Query databases for similar images."""
    try:
        # Generate embedding for query image
        query_embedding = generate_image_embeddings(image_path, output_embedding_length=1024)
        
        results = {}
        
        # Query dress database
        if search_type in ["both", "dress"] and dress_db:
            dress_results = dress_db.query_similar(query_embedding, n_results=top_k)
            results['dress_results'] = format_results(dress_results, "Dress-Only")
        
        # Query full image database  
        if search_type in ["both", "full"] and full_db:
            full_results = full_db.query_similar(query_embedding, n_results=top_k)
            results['full_results'] = format_results(full_results, "Full-Image")
        
        return results
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return {}

def display_image_with_info(img_path, caption="", width=150):
    """Display image with information."""
    try:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            st.image(img, caption=caption, width=width)
        else:
            st.write(f"❌ Image not found: {img_path}")
    except Exception as e:
        st.write(f"❌ Error loading image: {e}")

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("👗 Fashion Similarity Analysis")
    st.markdown("**Compare dress-only vs full-image similarity search**")
    
    # Load databases
    dress_db, full_db = load_databases()
    
    # Check database status
    col1, col2 = st.columns(2)
    with col1:
        if dress_db:
            dress_stats = dress_db.get_stats()
            st.success(f"✅ Dress DB: {dress_stats['total_embeddings']} embeddings")
        else:
            st.error("❌ Dress database not found")
    
    with col2:
        if full_db:
            full_stats = full_db.get_stats()
            st.success(f"✅ Full Image DB: {full_stats['total_embeddings']} embeddings")
        else:
            st.error("❌ Full image database not found")
    
    if not dress_db and not full_db:
        st.warning("⚠️ No databases found. Run `python3 fashion_similarity_analysis.py` first.")
        return
    
    st.divider()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Query Configuration")
    
    # Load sample images
    sample_images = load_sample_images()
    
    # Search configuration
    search_type = st.sidebar.selectbox(
        "Search Type:",
        options=["both", "dress", "full"],
        format_func=lambda x: {
            "both": "👗🖼️ Both (Dress + Full)",
            "dress": "👗 Dress-Only", 
            "full": "🖼️ Full-Image"
        }[x]
    )
    
    top_k = st.sidebar.slider("Number of Results:", min_value=1, max_value=10, value=5)
    
    # Query method selection
    query_method = st.sidebar.radio(
        "Query Method:",
        options=["Select from Samples", "Upload Image"]
    )
    
    query_image_path = None
    query_image_name = None
    
    if query_method == "Select from Samples":
        if sample_images:
            # Image selection
            image_names = [Path(img).stem for img in sample_images]
            selected_idx = st.sidebar.selectbox(
                "Select Query Image:",
                options=range(len(image_names)),
                format_func=lambda i: f"{i+1}. {image_names[i]}"
            )
            
            query_image_path = sample_images[selected_idx]
            query_image_name = image_names[selected_idx]
            
            # Display selected query image in sidebar
            st.sidebar.markdown("**Selected Query Image:**")
            try:
                query_img = Image.open(query_image_path)
                st.sidebar.image(query_img, caption=query_image_name, width=200)
            except Exception as e:
                st.sidebar.error(f"Could not load image: {e}")
        else:
            st.sidebar.error("No sample images found")
    
    else:  # Upload Image
        uploaded_file = st.sidebar.file_uploader(
            "Upload Query Image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to search for similar fashion items"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            query_image_path = temp_path
            query_image_name = uploaded_file.name
            
            # Display uploaded query image in sidebar
            st.sidebar.markdown("**Uploaded Query Image:**")
            try:
                query_img = Image.open(query_image_path)
                st.sidebar.image(query_img, caption=query_image_name, width=200)
            except Exception as e:
                st.sidebar.error(f"Could not load uploaded image: {e}")
    
    # Query execution
    if query_image_path and st.sidebar.button("🔍 Search Similar Images", type="primary"):
        
        # Display query image
        st.subheader(f"🎯 Query Image: {query_image_name}")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            display_image_with_info(query_image_path, f"Query: {query_image_name}", width=200)
        
        st.divider()
        
        # Perform search
        with st.spinner("Searching for similar images..."):
            results = query_similarity(query_image_path, dress_db, full_db, search_type, top_k)
        
        if results:
            # Display results
            if search_type == "both" and 'dress_results' in results and 'full_results' in results:
                # Side-by-side comparison
                st.subheader("🔄 Side-by-Side Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 👗 Dress-Only Results")
                    for i, result in enumerate(results['dress_results'][:top_k], 1):
                        with st.expander(f"#{i} {result['image_name']} (Similarity: {result['similarity']:.3f})"):
                            # Show segmented dress image for dress-only results
                            img_path = result['display_image_path']
                            display_image_with_info(img_path, f"Rank {i} (Segmented Dress)", width=150)
                            st.write(f"**Similarity:** {result['similarity']:.3f}")
                            st.write(f"**Distance:** {result['distance']:.3f}")
                
                with col2:
                    st.markdown("### 🖼️ Full-Image Results")
                    for i, result in enumerate(results['full_results'][:top_k], 1):
                        with st.expander(f"#{i} {result['image_name']} (Similarity: {result['similarity']:.3f})"):
                            # Show full original image for full-image results
                            img_path = result['display_image_path']
                            display_image_with_info(img_path, f"Rank {i} (Full Image)", width=150)
                            st.write(f"**Similarity:** {result['similarity']:.3f}")
                            st.write(f"**Distance:** {result['distance']:.3f}")
                
                # Comparison table
                st.subheader("📊 Comparison Table")
                comparison_data = []
                max_results = max(len(results['dress_results']), len(results['full_results']))
                
                for i in range(min(top_k, max_results)):
                    row = {"Rank": i + 1}
                    
                    if i < len(results['dress_results']):
                        dress_result = results['dress_results'][i]
                        row["Dress-Only"] = dress_result['image_name']
                        row["Dress Similarity"] = f"{dress_result['similarity']:.3f}"
                    else:
                        row["Dress-Only"] = "---"
                        row["Dress Similarity"] = "---"
                    
                    if i < len(results['full_results']):
                        full_result = results['full_results'][i]
                        row["Full-Image"] = full_result['image_name']
                        row["Full Similarity"] = f"{full_result['similarity']:.3f}"
                    else:
                        row["Full-Image"] = "---"
                        row["Full Similarity"] = "---"
                    
                    comparison_data.append(row)
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
            else:
                # Single search type results
                if 'dress_results' in results:
                    st.subheader("👗 Dress-Only Similarity Results")
                    st.info("💡 Showing segmented dress images for dress-only similarity")
                    display_results_grid(results['dress_results'][:top_k])
                
                if 'full_results' in results:
                    st.subheader("🖼️ Full-Image Similarity Results") 
                    st.info("💡 Showing full original images for full-image similarity")
                    display_results_grid(results['full_results'][:top_k])
        
        # Clean up temporary file
        if query_method == "Upload Image" and query_image_path and os.path.exists(query_image_path):
            os.remove(query_image_path)

def display_results_grid(results):
    """Display results in a grid format."""
    cols = st.columns(min(3, len(results)))  # Max 3 columns
    
    for i, result in enumerate(results):
        with cols[i % 3]:
            st.markdown(f"**#{i+1} {result['image_name']}**")
            # Use display_image_path (segmented for dress-only, full for full-image)
            img_path = result['display_image_path']
            display_image_with_info(img_path, width=150)
            st.write(f"**Similarity:** {result['similarity']:.3f}")
            st.write(f"**Distance:** {result['distance']:.3f}")
            st.write("---")


if __name__ == "__main__":
    main()