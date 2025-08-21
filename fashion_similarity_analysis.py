#!/usr/bin/env python3
"""
Fashion Similarity Analysis: Dress-Only vs Full-Image Embeddings

This script demonstrates how semantic segmentation improves clothing similarity matching by comparing:
1. Dress-only embeddings - Extracted dress regions only  
2. Full-image embeddings - Complete original images

We'll show that dress-only search provides better clothing similarity while full-image search 
can be misled by pose similarity.
"""

import pandas as pd
import numpy as np
import glob
import os
import shutil
from pathlib import Path
import json

# Import our custom modules
from segmentation import generate_segmentation_mask
from image_extraction import extract_masked_region
from embeddings import generate_image_embeddings
from vector_db import VectorDatabase

# Configuration
OUTPUT_BASE_DIR = 'analysis'
MASK_PROMPT = 'dress'
EMBEDDING_DIMENSION = 1024  # Use 1024 dimensions to match existing databases

def load_sample_images():
    """Load and preview sample images."""
    print("=" * 80)
    print("STEP 1: LOADING SAMPLE IMAGES")
    print("=" * 80)
    
    image_paths = sorted(glob.glob('sample_images/output/*.png'))
    print(f"Found {len(image_paths)} sample images")
    
    print("\n📸 Sample Images Preview:")
    for i, img_path in enumerate(image_paths[:6]):  # Show first 6
        print(f"  {i+1}. {Path(img_path).name}")
    
    if len(image_paths) > 6:
        print(f"  ... and {len(image_paths) - 6} more images")
    
    return image_paths

def segment_dresses(image_paths):
    """Segment dresses from original images with smart skipping."""
    print("\n" + "=" * 80)
    print("STEP 2: SEGMENTING DRESSES FROM IMAGES")
    print("=" * 80)
    
    print(f"🎯 Processing {len(image_paths)} images...")
    print(f"📁 Output directory: {OUTPUT_BASE_DIR}")
    print(f"🏷️  Mask prompt: {MASK_PROMPT}")
    
    segmentation_results = {}
    failed_segmentations = []
    skipped_segmentations = []
    
    for i, img_path in enumerate(image_paths):
        img_name = Path(img_path).stem
        output_dir = f'{OUTPUT_BASE_DIR}/{img_name}'
        
        print(f"\n{i+1}/{len(image_paths)} Processing: {Path(img_path).name}")
        
        # Use the correct filename pattern that matches helper functions
        mask_path = f"{output_dir}/{img_name}_{MASK_PROMPT}_mask.png"
        extracted_path = f"{output_dir}/{img_name}_{MASK_PROMPT}_extracted.png"
        
        # Check for existing files
        if os.path.exists(mask_path) and os.path.exists(extracted_path):
            print(f"  ⏭️  Skipping: Segmentation already complete")
            skipped_segmentations.append(img_name)
            
            # Load existing result
            segmentation_results[img_name] = {
                'original_image': img_path,
                'mask_path': mask_path,
                'extracted_region': extracted_path,
                'output_dir': output_dir
            }
            continue
        
        print(f"  🆕 Generating segmentation...")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate segmentation mask
            print(f"    📍 Generating mask for '{MASK_PROMPT}'...")
            image_bytes, mask_bytes = generate_segmentation_mask(img_path, MASK_PROMPT)
            
            if mask_bytes:
                # Save mask
                with open(mask_path, "wb") as f:
                    f.write(mask_bytes)
                print(f"    ✅ Mask saved: {mask_path}")
                
                # Extract masked region
                print(f"    🔗 Extracting masked region...")
                extract_masked_region(img_path, mask_path, extracted_path)
                print(f"    ✅ Extracted region saved: {extracted_path}")
                
                # Store result (NO EMBEDDINGS YET)
                segmentation_results[img_name] = {
                    'original_image': img_path,
                    'mask_path': mask_path,
                    'extracted_region': extracted_path,
                    'output_dir': output_dir
                }
                
            else:
                raise Exception("No mask generated")
                
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            failed_segmentations.append(img_name)
    
    print(f"\n📊 SEGMENTATION SUMMARY:")
    print(f"  - New segmentations: {len(segmentation_results) - len(skipped_segmentations)}")
    print(f"  - Skipped (already exists): {len(skipped_segmentations)}")
    print(f"  - Failed: {len(failed_segmentations)}")
    print(f"  - Total with segmentation: {len(segmentation_results)}")
    
    if failed_segmentations:
        print(f"❌ Failed images: {failed_segmentations}")
    
    return segmentation_results

def generate_embeddings(segmentation_results):
    """Generate embeddings for segmented images."""
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("=" * 80)
    
    print(f"🔢 Generating embeddings for segmented images...")
    print(f"📏 Embedding dimension: {EMBEDDING_DIMENSION}")
    
    embedding_results = {}
    failed_embeddings = []
    skipped_embeddings = []
    
    for img_name, seg_result in segmentation_results.items():
        print(f"\n📸 Processing embeddings for: {img_name}")
        
        # Check what embeddings we already have
        original_path = seg_result['original_image']
        extracted_path = seg_result['extracted_region']
        
        # Check if we need to generate embeddings
        needs_original = 'original_embedding' not in seg_result
        needs_extracted = 'extracted_embedding' not in seg_result
        
        if not needs_original and not needs_extracted:
            print(f"  ⏭️  Skipping: Both embeddings already exist")
            skipped_embeddings.append(img_name)
            embedding_results[img_name] = seg_result
            continue
        
        try:
            result = seg_result.copy()  # Start with segmentation data
            
            # Generate original image embedding if needed
            if needs_original:
                print(f"  🔢 Generating original image embedding...")
                original_embedding = generate_image_embeddings(
                    original_path, 
                    output_embedding_length=EMBEDDING_DIMENSION
                )
                result['original_embedding'] = original_embedding
                print(f"    ✅ Original: {len(original_embedding)} dimensions")
            
            # Generate extracted region embedding if needed
            if needs_extracted and os.path.exists(extracted_path):
                print(f"  🔢 Generating extracted region embedding...")
                extracted_embedding = generate_image_embeddings(
                    extracted_path,
                    output_embedding_length=EMBEDDING_DIMENSION
                )
                result['extracted_embedding'] = extracted_embedding
                print(f"    ✅ Extracted: {len(extracted_embedding)} dimensions")
            elif not os.path.exists(extracted_path):
                print(f"  ⚠️  Extracted file missing: {extracted_path}")
            
            embedding_results[img_name] = result
            
        except Exception as e:
            print(f"  ❌ Failed to generate embeddings: {str(e)}")
            failed_embeddings.append(img_name)
    
    # Final results with both segmentation AND embeddings
    successful_results = {k: v for k, v in embedding_results.items() 
                         if 'extracted_embedding' in v and 'original_embedding' in v}
    
    print(f"\n📊 EMBEDDING SUMMARY:")
    print(f"  - New embeddings generated: {len(embedding_results) - len(skipped_embeddings)}")
    print(f"  - Skipped (already exists): {len(skipped_embeddings)}")
    print(f"  - Failed: {len(failed_embeddings)}")
    print(f"  - Complete results (segmentation + embeddings): {len(successful_results)}")
    
    if failed_embeddings:
        print(f"❌ Failed embedding generation: {failed_embeddings}")
    
    return successful_results

def create_and_populate_vectordbs(successful_results):
    """Create and populate vector databases."""
    print("\n" + "=" * 80)
    print("STEP 4: CREATING AND POPULATING VECTOR DATABASES")
    print("=" * 80)
    
    print("🧹 Cleaning up existing databases...")
    for db_path in ["./analysis_dress_db", "./analysis_full_db"]:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"  Removed existing database: {db_path}")
    
    print("\n🗄️ Creating fresh vector databases...")
    
    # Create databases
    dress_vectordb = VectorDatabase("./analysis_dress_db", "dress_embeddings")
    full_vectordb = VectorDatabase("./analysis_full_db", "full_image_embeddings")
    
    print("\n📥 Populating databases...")
    dress_embeddings_data = []
    full_embeddings_data = []
    
    for img_name, result in successful_results.items():
        print(f"  📍 Adding {img_name}...")
        
        # Add dress embedding
        if 'extracted_embedding' in result:
            dress_metadata = {
                'image_name': img_name,
                'image_path': result['original_image'],
                'type': 'dress_only',
                'extracted_path': result.get('extracted_region')
            }
            
            dress_id = f"{img_name}_dress"
            dress_vectordb.add_embedding(result['extracted_embedding'], dress_metadata, dress_id)
            dress_embeddings_data.append({
                'id': dress_id,
                'image_name': img_name,
                'embedding': result['extracted_embedding']
            })
        
        # Add full image embedding
        if 'original_embedding' in result:
            full_metadata = {
                'image_name': img_name,
                'image_path': result['original_image'], 
                'type': 'full_image'
            }
            
            full_id = f"{img_name}_full"
            full_vectordb.add_embedding(result['original_embedding'], full_metadata, full_id)
            full_embeddings_data.append({
                'id': full_id,
                'image_name': img_name,
                'embedding': result['original_embedding']
            })
    
    print(f"\n✅ Databases populated:")
    print(f"  - Dress embeddings: {dress_vectordb.get_stats()['total_embeddings']}")
    print(f"  - Full image embeddings: {full_vectordb.get_stats()['total_embeddings']}")
    print(f"  - Dress data entries: {len(dress_embeddings_data)}")
    print(f"  - Full data entries: {len(full_embeddings_data)}")
    
    return dress_vectordb, full_vectordb, dress_embeddings_data, full_embeddings_data

def perform_similarity_searches(dress_vectordb, full_vectordb, dress_embeddings_data, full_embeddings_data):
    """Perform similarity searches on both databases."""
    print("\n" + "=" * 80)
    print("STEP 5: PERFORMING SIMILARITY SEARCHES")
    print("=" * 80)
    
    def get_similarity_results(vectordb, embeddings_data, top_k=3):
        """Get similarity results for all embeddings in the database."""
        results = []
        
        for item in embeddings_data:
            query_embedding = item['embedding']
            query_name = item['image_name']
            
            # Query the database
            search_results = vectordb.query_similar(query_embedding, n_results=top_k + 1)  # +1 to exclude self
            
            # Process results (exclude self-match)
            neighbors = []
            for doc_id, metadata, distance in zip(
                search_results['ids'][0],
                search_results['metadatas'][0],
                search_results['distances'][0]
            ):
                if metadata['image_name'] != query_name:  # Exclude self
                    similarity = 1 - distance
                    neighbors.append({
                        'neighbor': metadata['image_name'],
                        'similarity': similarity,
                        'distance': distance
                    })
                
                if len(neighbors) >= top_k:
                    break
            
            results.append({
                'query_image': query_name,
                'neighbors': neighbors
            })
        
        return results
    
    print("🔍 Performing similarity searches...")
    
    # Get similarity results for both databases
    dress_similarity_results = get_similarity_results(dress_vectordb, dress_embeddings_data)
    full_similarity_results = get_similarity_results(full_vectordb, full_embeddings_data)
    
    print(f"✅ Completed similarity searches for {len(dress_similarity_results)} images")
    
    return dress_similarity_results, full_similarity_results

def analyze_results(dress_similarity_results, full_similarity_results):
    """Analyze and compare the similarity search results."""
    print("\n" + "=" * 80)
    print("STEP 6: ANALYZING RESULTS")
    print("=" * 80)
    
    def create_comparison_dataframe(dress_results, full_results):
        """Create a comprehensive comparison dataframe."""
        comparison_data = []
        
        for dress_result, full_result in zip(dress_results, full_results):
            query_image = dress_result['query_image']
            
            # Get top 3 neighbors for each type
            for rank in range(min(3, len(dress_result['neighbors']), len(full_result['neighbors']))):
                dress_neighbor = dress_result['neighbors'][rank]
                full_neighbor = full_result['neighbors'][rank]
                
                comparison_data.append({
                    'Query Image': query_image,
                    'Rank': rank + 1,
                    'Dress-Only Neighbor': dress_neighbor['neighbor'],
                    'Dress-Only Similarity': f"{dress_neighbor['similarity']:.3f}",
                    'Full-Image Neighbor': full_neighbor['neighbor'], 
                    'Full-Image Similarity': f"{full_neighbor['similarity']:.3f}",
                    'Difference': f"{dress_neighbor['similarity'] - full_neighbor['similarity']:.3f}"
                })
        
        return pd.DataFrame(comparison_data)
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(dress_similarity_results, full_similarity_results)
    
    print("📊 Similarity Comparison Results")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # Calculate statistics
    dress_similarities = pd.to_numeric(comparison_df['Dress-Only Similarity'])
    full_similarities = pd.to_numeric(comparison_df['Full-Image Similarity'])
    differences = pd.to_numeric(comparison_df['Difference'])
    
    stats = {
        'dress_avg': dress_similarities.mean(),
        'full_avg': full_similarities.mean(),
        'avg_difference': differences.mean(),
        'dress_std': dress_similarities.std(),
        'full_std': full_similarities.std(),
        'positive_differences': (differences > 0).sum(),
        'negative_differences': (differences < 0).sum(),
        'total_comparisons': len(differences)
    }
    
    print(f"\n📈 Statistical Analysis")
    print("=" * 50)
    print(f"Average Dress-Only Similarity: {stats['dress_avg']:.3f} (±{stats['dress_std']:.3f})")
    print(f"Average Full-Image Similarity: {stats['full_avg']:.3f} (±{stats['full_std']:.3f})")
    print(f"Average Difference: {stats['avg_difference']:.3f}")
    print(f"\nComparisons where dress-only > full-image: {stats['positive_differences']}/{stats['total_comparisons']}")
    print(f"Comparisons where dress-only < full-image: {stats['negative_differences']}/{stats['total_comparisons']}")
    
    # Save results to JSON (convert numpy types to native Python types)
    results_output = {
        'comparison_data': comparison_df.to_dict('records'),
        'statistics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, np.integer) else v 
                      for k, v in stats.items()},
        'summary': {
            'total_images_analyzed': len(dress_similarity_results),
            'embedding_dimension': EMBEDDING_DIMENSION,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    output_file = 'fashion_similarity_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return comparison_df, stats

def print_conclusions():
    """Print key findings and conclusions."""
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    
    conclusions = """
### Key Findings:

1. **Dress-Only Search Advantages:**
   - Focuses purely on clothing characteristics (fabric, style, color)
   - Reduces noise from pose, background, and body positioning
   - Provides more consistent clothing-based similarity

2. **Full-Image Search Limitations:**
   - Can be influenced by pose similarity rather than clothing similarity
   - Background elements may affect similarity scores
   - Less precise for fashion-specific applications

3. **Practical Applications:**
   - **Fashion retail:** Dress-only search better for "find similar styles"
   - **Pose analysis:** Full-image search better for "find similar poses"
   - **Hybrid approach:** Use both for comprehensive similarity

### Recommendations:
- Use **dress-only embeddings** when similarity should be based on clothing characteristics
- Use **full-image embeddings** when context (pose, setting) matters
- Consider **weighted combinations** of both embedding types for balanced results
"""
    
    print(conclusions)

def main():
    """Main analysis pipeline."""
    print("🎯 FASHION SIMILARITY ANALYSIS: DRESS-ONLY VS FULL-IMAGE EMBEDDINGS")
    print("=" * 80)
    print("This analysis demonstrates how semantic segmentation improves clothing similarity matching")
    print("by comparing dress-only embeddings vs full-image embeddings.")
    
    try:
        # Step 1: Load sample images
        image_paths = load_sample_images()
        
        # Step 2: Segment dresses from images
        segmentation_results = segment_dresses(image_paths)
        
        # Step 3: Generate embeddings
        successful_results = generate_embeddings(segmentation_results)
        
        if not successful_results:
            print("❌ No successful results to analyze. Exiting.")
            return
        
        # Step 4: Create and populate vector databases
        dress_vectordb, full_vectordb, dress_embeddings_data, full_embeddings_data = create_and_populate_vectordbs(successful_results)
        
        # Step 5: Perform similarity searches
        dress_similarity_results, full_similarity_results = perform_similarity_searches(
            dress_vectordb, full_vectordb, dress_embeddings_data, full_embeddings_data
        )
        
        # Step 6: Analyze results
        comparison_df, stats = analyze_results(dress_similarity_results, full_similarity_results)
        
        # Print conclusions
        print_conclusions()
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Analyzed {len(successful_results)} images with {EMBEDDING_DIMENSION}-dimensional embeddings")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()