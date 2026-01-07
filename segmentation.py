"""
Semantic segmentation module using Amazon Titan Image Generator.
"""

import base64
import json
import logging
import boto3
import os
from botocore.exceptions import ClientError

class ImageError(Exception):
    """Custom exception for errors returned by Amazon Titan Image Generator G1"""
    def __init__(self, message):
        self.message = message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_segmentation_mask(image_path, mask_prompt, model_id='amazon.titan-image-generator-v2:0', region='us-west-2'):
    """
    Generate segmentation mask using Amazon Titan Image Generator G1 with inpainting.
    
    Args:
        image_path (str): Path to the input image
        mask_prompt (str): Text prompt describing what to segment (e.g., "person", "clothing", "background")
        model_id (str): The Titan model ID to use
        region (str): AWS region
        
    Returns:
        tuple: (generated_image_bytes, mask_bytes)
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read and encode input image
    with open(image_path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode('utf8')
    
    # Create request body with returnMask=True to get the segmentation mask
    body = json.dumps({
        "taskType": "INPAINTING", 
        "inPaintingParams": {
            "text": f"segment and highlight {mask_prompt}",
            "negativeText": "bad quality, low res, blurry",
            "image": input_image,
            "maskPrompt": mask_prompt,
            "returnMask": True
        },
        "imageGenerationConfig": {
            "quality": "standard",
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    })
    
    logger.info(f"Generating segmentation mask for '{mask_prompt}' using Amazon Titan Image Generator G1")
    
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=region)
    
    try:
        response = bedrock.invoke_model(
            body=body, 
            modelId=model_id, 
            accept="application/json", 
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Check for errors
        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise ImageError(f"Image generation error: {finish_reason}")
        
        # Extract images
        images = response_body.get("images", [])
        if not images:
            raise ImageError("No images returned from the model")
        
        # Get the generated image
        base64_image = images[0]
        image_bytes = base64.b64decode(base64_image.encode('ascii'))
        
        # Get the mask from maskImage key
        mask_bytes = None
        mask_image = response_body.get("maskImage")
        if mask_image:
            mask_bytes = base64.b64decode(mask_image.encode('ascii'))
        
        logger.info("Successfully generated segmentation mask")
        return image_bytes, mask_bytes
        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        raise ImageError(f"Client error: {message}")
