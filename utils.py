import os
import torch
from PIL import Image
from safetensors.torch import load_file
from datasets import load_from_disk
from transformers import AutoModel, AutoProcessor

# ----------------------------------------------------------------------
# Global model cache (load once)
# ----------------------------------------------------------------------
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

    # Choose attention implementation based on availability
    try:
        attn_impl = "flash_attention_2"
        _ = torch.backends.cuda.flash_sdp_enabled()  # just a check
    except:
        attn_impl = "eager"

    _embedding_model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map="auto" if torch.cuda.is_available() else "cpu"
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
    elif modality == "text":
        p_max_length = 8192
    else:
        raise ValueError(f"Unknown modality: {modality}")

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
        dataset_path="data/dataset/",
        embedding_path="data/embedding_tensors/all_recipes_image_text_embeddings.safetensors"
    ):
        # Load dataset from disk
        self.dataset = load_from_disk(dataset_path)

        # Load precomputed image_text embeddings
        emb_dict = load_file(embedding_path)
        self.target_embeddings = emb_dict["image_text_embeddings"]

        # Move to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_embeddings = self.target_embeddings.to(self.device)

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