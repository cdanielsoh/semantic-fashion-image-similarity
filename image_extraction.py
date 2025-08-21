"""
Image extraction module for applying masks to original images.
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path

def extract_masked_region(original_image_path, mask_image_path, output_path=None):
    """
    Extract the region from the original image using the segmentation mask.
    
    Args:
        original_image_path (str): Path to the original image
        mask_image_path (str): Path to the segmentation mask
        output_path (str): Optional path to save the extracted region
        
    Returns:
        PIL.Image: The extracted region with white background
    """
    
    # Open original image and mask
    original = Image.open(original_image_path).convert("RGB")
    mask = Image.open(mask_image_path).convert("L")  # Convert to grayscale
    
    # Resize mask to match original image if needed
    if original.size != mask.size:
        mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    original_array = np.array(original)
    mask_array = np.array(mask)
    
    # Create RGB result with white background
    result_array = np.full_like(original_array, 255)  # Start with white background
    
    # Apply inverted mask (black areas in mask = keep original, white areas = white background)
    # Create boolean mask where black pixels (0) are True
    keep_mask = mask_array < 128  # Black areas become True
    
    # Apply the mask to all RGB channels
    result_array[keep_mask] = original_array[keep_mask]
    
    # Create result image
    extracted = Image.fromarray(result_array, 'RGB')
    
    # Save if output path provided
    if output_path:
        extracted.save(output_path)
        print(f"Extracted region saved to: {output_path}")
    
    return extracted

def extract_and_save_masked_region(original_image_path, mask_image_path, output_dir="extracted_regions"):
    """
    Extract masked region and save it with a meaningful filename.
    
    Args:
        original_image_path (str): Path to original image
        mask_image_path (str): Path to mask image
        output_dir (str): Directory to save extracted region
        
    Returns:
        str: Path to the saved extracted region
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    mask_name = Path(mask_image_path).stem
    output_filename = f"{mask_name}_extracted.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Extract and save
    extracted = extract_masked_region(original_image_path, mask_image_path, output_path)
    
    return output_path