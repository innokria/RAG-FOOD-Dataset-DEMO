import os
import torch
from safetensors.torch import load_file
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    model_name = "nvidia/llama-nemotron-embed-vl-1b-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"--- Loading Model Weights ({device})...")
    _embedding_model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device
    ).eval()
    print("--- Model Loaded Successfully!")
    return _embedding_model

# make a helper function to change the model's modality
def prepare_processor(modality , embedding_model):
    """Prepare the model for different modality.

    Args:
        modality (str): can be 'text' , 'image' or 'image_text'
        embedding_model (_type_): embedding model

    Returns:
        _type_: (modality , embedding_model)
    """
    # Set max number of tokens (p_max_length) based on modality
    if modality == "image":
        p_max_length = 2048
    elif modality == "image_text":
        p_max_length = 10240
    elif modality == "text":
        p_max_length = 8192
    embedding_model.processor.p_max_length = p_max_length
    # Image specific settings(only matter if image is present)
    # Sets max number of tiles an image can be split. Each tile consumes 256 tokens.
    embedding_model.processor.max_input_tiles = 6
    # Enables an extra tile with the full image at lower resolution
    embedding_model.processor.use_thumbnail = True
    return modality , embedding_model


class RecipeRetriever:
    def __init__(self , 
        dataset = None , 
        embedding_path = "data/embedding_tensors/all_recipes_image_text_embeddings.safetensors"
    ):
        # load the dataset from local imagefolder directory
        self.dataset = dataset
        
        if self.dataset is None:
            raise ValueError("RecipeRetriever was initialized without a dataset. Please provide one.")
        # load the embeddings
        self.target_embeddings = load_file(embedding_path)["image_text_embeddings"]
        if self.target_embeddings is None:
            raise ValueError("RecipeRetriever was initialized without embedding vectors. Please provide one.")
    
    def _l2_normalize(self , x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def match_query(self, text_query=None, image_query=None, top_k=6):
        # 1. ENSURE THE MODEL IS LOADED (via the safe global helper)
        embedding_model = get_embedding_model()
        
        # set modality based on query
        if text_query and image_query:
            modality = "image_text"
        elif text_query:
            modality = "text"
        elif image_query:
            modality = "image"
        else:
            raise ValueError("At least one query (text or image) must be provided.")
        
        modality, embedding_model = prepare_processor(
            modality=modality,
            embedding_model=embedding_model
        )
        
        # create query embedding
        with torch.inference_mode():
            if modality == "text":
                query_embedding = embedding_model.encode_queries([text_query])
            elif modality == "image":
                query_embedding = embedding_model.encode_documents(images=[image_query])
            elif modality == "image_text":
                query_embedding = embedding_model.encode_documents(
                    texts=[text_query],
                    images=[image_query]
                )
        
        device = self.target_embeddings.device
        query_embedding = query_embedding.to(device)
        
        # Similarity score
        scores_matrix = self._l2_normalize(query_embedding) @ self._l2_normalize(self.target_embeddings).T
        scores_flat = scores_matrix.flatten()
        
        # Get Top K
        k_val = min(top_k, scores_flat.size(0))
        top_scores, top_indices = torch.topk(scores_flat, k=k_val)
        
        # Map back to Dataset
        results = []
        for i in range(k_val):
            idx = top_indices[i].item()
            recipe_data = self.dataset[idx]
            results.append({
                "rank": i + 1,
                "score": top_scores[i].item(),
                "recipe": recipe_data
            })
        
        return results

