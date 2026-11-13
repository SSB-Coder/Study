"""
39. Vector Databases & HNSW (Hierarchical Navigable Small World)
Deep Dive Implementation

Standard databases use SQL to find exact keyword matches.
Vector databases (Pinecone, FAISS, Milvus) store neural network embeddings
and search by *meaning* (Cosine Similarity). 

If you have 1 Billion documents (RAG), doing a flat math comparison against 
every single document takes too long. Vector Databases use the HNSW algorithm 
to search a massive graph in milliseconds.
"""

import numpy as np
import math

# ---------------------------------------------------------------------------
# 1. THE EMBEDDING MATH (Cosine Similarity)
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a, vec_b):
    """
    Calculates the angle between two high-dimensional vectors.
    1.0 means perfectly identical meaning. 
    0.0 means completely unrelated.
    -1.0 means opposite meaning.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

# ---------------------------------------------------------------------------
# 2. HNSW MOCK GRAPH (Hierarchical Navigable Small World)
# ---------------------------------------------------------------------------
# HNSW works like an interstate highway system. 
# Top Layer (Layer 2): Only major cities. You quickly find the closest city.
# Middle Layer (Layer 1): Towns. You drop down and find the closest town.
# Bottom Layer (Layer 0): Every single house. You navigate to the exact street.

class SimplifiedHNSWNode:
    def __init__(self, id, vector, text):
        self.id = id
        self.vector = vector
        self.text = text
        self.connections = [] # Other nodes this node is linked to

def greedy_graph_search(entry_node, query_vector):
    """
    Simulates navigating the "Small World" graph.
    Instead of checking all 1 million nodes, we start at the entry node,
    and greedily walk towards neighbors that have higher cosine similarity
    to our query vector. We stop when no neighbors are closer.
    """
    current_node = entry_node
    current_similarity = cosine_similarity(query_vector, current_node.vector)
    
    path_taken = [current_node.id]
    
    while True:
        best_neighbor = None
        best_neighbor_sim = current_similarity
        
        # Look at all connected neighbors
        for neighbor in current_node.connections:
            sim = cosine_similarity(query_vector, neighbor.vector)
            if sim > best_neighbor_sim:
                best_neighbor = neighbor
                best_neighbor_sim = sim
                
        # If we found a neighbor that is closer to our query, walk to it
        if best_neighbor is not None:
            current_node = best_neighbor
            current_similarity = best_neighbor_sim
            path_taken.append(current_node.id)
        else:
            # We reached a local maximum (the closest node in this neighborhood)
            break
            
    return current_node, current_similarity, path_taken

# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Vector Database Setup ---")
    # Simulate an LLM generating 1536-dimensional embeddings for documents
    np.random.seed(42)
    doc_1 = np.random.randn(128) # "The dog chased the cat."
    doc_2 = np.random.randn(128) # "Felines are hunted by canines."
    doc_3 = np.random.randn(128) # "The stock market crashed today."
    
    # Manually force doc_1 and doc_2 to be highly similar (meaning overlap)
    doc_2 = doc_1 + (np.random.randn(128) * 0.2)
    
    print(f"Similarity (Dog vs Feline): {cosine_similarity(doc_1, doc_2):.4f} (High!)")
    print(f"Similarity (Dog vs Stocks): {cosine_similarity(doc_1, doc_3):.4f} (Low!)")
    
    print("\n--- 2. HNSW Graph Search (RAG Retrieval) ---")
    # Create nodes
    node_A = SimplifiedHNSWNode("A_Dog", doc_1, "The dog chased the cat.")
    node_B = SimplifiedHNSWNode("B_Feline", doc_2, "Felines are hunted by canines.")
    node_C = SimplifiedHNSWNode("C_Stocks", doc_3, "The stock market crashed today.")
    
    # Build the graph (Link them together)
    node_C.connections.append(node_A)
    node_A.connections.append(node_B)
    
    # Let's say a user queries: "Tell me about puppies chasing kitties"
    # The query vector is very close to doc_1 and doc_2
    query_vector = doc_1 + (np.random.randn(128) * 0.1)
    
    # Start the search at a random entry point (The Stock Market node)
    # HNSW will quickly route us away from finance and towards animals without 
    # needing to scan a billion unrelated finance documents.
    found_node, sim, path = greedy_graph_search(entry_node=node_C, query_vector=query_vector)
    
    print(f"Entry Point: {node_C.id}")
    print(f"Path Navigated: {path}")
    print(f"Final Retrieved Document: '{found_node.text}'")
    print(f"Final Similarity Score: {sim:.4f}")
