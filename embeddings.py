"""
Image embedding module using Amazon Titan Multimodal Embeddings.
"""

import base64
import json
import logging
import boto3
import os
from botocore.exceptions import ClientError

class EmbedError(Exception):
    """Custom exception for errors returned by Amazon Titan Multimodal Embeddings G1"""
    def __init__(self, message):
        self.message = message

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_image_embeddings(image_path, model_id='amazon.titan-embed-image-v1', output_embedding_length=1024, region='us-west-2'):
    """
    Generate embeddings for an image using Amazon Titan Multimodal Embeddings G1.
    
    Args:
        image_path (str): Path to the image file
        model_id (str): The Titan embedding model ID
        output_embedding_length (int): Length of output embeddings (256, 384, 1024)
        region (str): AWS region
        
    Returns:
        list: The embedding vector
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read and encode image
    with open(image_path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode('utf8')
    
    # Create request body
    body = json.dumps({
        "inputImage": input_image,
        "embeddingConfig": {
            "outputEmbeddingLength": output_embedding_length
        }
    })
    
    logger.info(f"Generating embeddings for {os.path.basename(image_path)} with length {output_embedding_length}")
    
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=region)
    
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        
        # Check for errors
        finish_reason = response_body.get("message")
        if finish_reason is not None:
            raise EmbedError(f"Embeddings generation error: {finish_reason}")
        
        embedding = response_body.get('embedding')
        if not embedding:
            raise EmbedError("No embedding returned from the model")
        
        logger.info(f"Successfully generated {len(embedding)}-dimensional embedding")
        return embedding
        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        raise EmbedError(f"Client error: {message}")

def save_embeddings(embeddings_dict, output_path):
    """
    Save embeddings dictionary to JSON file.
    
    Args:
        embeddings_dict (dict): Dictionary with image info and embeddings
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"Embeddings saved to: {output_path}")