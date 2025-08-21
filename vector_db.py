"""
Vector database module using ChromaDB for storing and querying embeddings.
"""

import chromadb
from pathlib import Path

class VectorDatabase:
    """Local vector database using ChromaDB for storing and querying image embeddings."""
    
    def __init__(self, db_path="./chroma_db", collection_name="image_embeddings"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            db_path (str): Path to store the database
            collection_name (str): Name of the collection
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"Initialized vector database at {db_path}")
        print(f"Collection '{collection_name}' ready with {self.collection.count()} existing vectors")
    
    def add_embedding(self, embedding, metadata, doc_id=None):
        """
        Add an embedding to the database.
        
        Args:
            embedding (list): The embedding vector
            metadata (dict): Metadata about the image/embedding
            doc_id (str): Optional unique ID, will generate if not provided
            
        Returns:
            str: The document ID
        """
        if doc_id is None:
            # Generate ID from metadata
            image_name = Path(metadata.get('image_path', 'unknown')).stem
            prompt = metadata.get('mask_prompt', 'unknown')
            doc_id = f"{image_name}_{prompt}_{len(embedding)}d"
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        print(f"Added embedding for {doc_id} (dimension: {len(embedding)})")
        return doc_id
    
    def query_similar(self, query_embedding, n_results=5, include_distances=True):
        """
        Find similar embeddings in the database.
        
        Args:
            query_embedding (list): The query embedding vector
            n_results (int): Number of results to return
            include_distances (bool): Whether to include similarity distances
            
        Returns:
            dict: Query results with IDs, metadata, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'distances'] if include_distances else ['metadatas']
        )
        
        return results
    
    def get_stats(self):
        """Get database statistics."""
        count = self.collection.count()
        return {
            "total_embeddings": count,
            "collection_name": self.collection_name,
            "db_path": self.db_path
        }
    
    def list_all(self):
        """List all embeddings in the database."""
        results = self.collection.get(include=['metadatas'])
        return results
    
    def reset_collection(self):
        """Delete and recreate the collection (removes all data)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection '{self.collection_name}'")
        except Exception:
            print(f"Collection '{self.collection_name}' doesn't exist or already deleted")
        
        # Recreate collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection '{self.collection_name}'")