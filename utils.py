import os
import torch
import requests
import io
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel

# global model cache(need to load once)
_embedding_model = None

def get_embedding_model():
    """
    Load the Llama Nemotron Embed VL model with the required revision.
    Uses flash_attention_2 if available, otherwise falls back.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    model_name = "nvidia/llama-nemotron-embed-vl-1b-v2"
    revision = "062ffaa1e3d24a8a50bd6a7ac7b8e54103e1f01d"
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # float16 for GPU, float32 for CPU (Space Free Tier in our Huggingface)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Choose attention implementation
    try:
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
    except:
        attn_impl = "eager"
    
    print(f"Loading model on {device}...")
    
    _embedding_model = AutoModel.from_pretrained(
        model_name,
        revision = revision,
        dtype = dtype,
        trust_remote_code = True,
        attn_implementation = attn_impl,
        device_map = device
    ).eval()
    return _embedding_model


def prepare_processor(modality, embedding_model):
    """
    Configure the processor's max length and image tiling based on modality.
    Modality can be 'text', 'image', or 'image_text'.
    """
    if modality == "image":
        p_max_length = 2048
    elif modality == "image_text":
        p_max_length = 10240
    else:  # text
        p_max_length = 8192

    embedding_model.processor.p_max_length = p_max_length
    embedding_model.processor.max_input_tiles = 6
    embedding_model.processor.use_thumbnail = True
    return modality, embedding_model


class RecipeRetriever:
    """
    Retrieves recipes from a pre‑embedded dataset using the Nemotron embedding model.
    Expects the dataset saved with `save_to_disk` and the embeddings as a safetensors file.
    """

    def __init__(
        self,
        hf_dataset_repo="tiptoghosh/your-dataset-name", 
        embedding_path="data/all_recipes_image_text_embeddings.safetensors"
    ):
        self.repo = hf_dataset_repo
        
        # Load local embeddings (only ~30-50MB, very fast)
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Missing embeddings at {embedding_path}")
        
        emb_dict = load_file(embedding_path)
        self.target_embeddings = emb_dict["image_text_embeddings"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_embeddings = self.target_embeddings.to(self.device)
        
        # We don't load the dataset locally anymore!
        self.dataset = None
    
    def _fetch_row_via_api(self, idx):
        """Fetches a specific row from the HF Dataset Server without downloading the whole dataset."""
        url = f"https://datasets-server.huggingface.co/rows?dataset={self.repo}&split=train&offset={idx}&length=1"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            row_content = data["rows"][0]["row"]
            
            # Convert the image from the API format (usually a URL or dict) to PIL
            # If the image is a dict with 'src', we fetch it
            img_data = row_content["image"]
            if isinstance(img_data, dict) and "src" in img_data:
                img_response = requests.get(img_data["src"])
                row_content["image"] = Image.open(io.BytesIO(img_response.content))
            
            return row_content
        except Exception as e:
            print(f"Error fetching row {idx}: {e}")
            return None
        
        
    def _l2_normalize(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def match_query(self, text_query=None, image_query=None, top_k=6):
        """
        Given a text query, an image query, or both, return the top‑k matching recipes.
        Returns a list of dicts: [{"rank": int, "score": float, "recipe": dict}, ...]
        """
        if not text_query and image_query is None:
            raise ValueError("At least one query (text or image) must be provided.")

        # Determine modality
        if text_query and image_query is not None:
            modality = "image_text"
        elif text_query:
            modality = "text"
        else:
            modality = "image"

        # Get the embedding model and prepare processor
        model = get_embedding_model()
        modality, model = prepare_processor(modality, model)

        # Encode the query
        with torch.inference_mode():
            if modality == "text":
                query_embedding = model.encode_queries([text_query])
            elif modality == "image":
                query_embedding = model.encode_documents(images=[image_query])
            else:  # image_text
                query_embedding = model.encode_documents(
                    texts=[text_query], images=[image_query]
                )

        # Move query to same device as target embeddings
        query_embedding = query_embedding.to(self.device)

        # Compute cosine similarity
        query_norm = self._l2_normalize(query_embedding)
        target_norm = self._l2_normalize(self.target_embeddings)
        scores_flat = (query_norm @ target_norm.T).flatten()

        # Top‑k
        k_val = min(top_k, scores_flat.size(0))
        top_scores, top_indices = torch.topk(scores_flat, k=k_val)

        # Build results
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