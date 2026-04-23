import gradio as gr
from utils import RecipeRetriever
import warnings
warnings.filterwarnings('ignore' , category = FutureWarning)

import torch
from transformers import AutoModel , AutoProcessor
from datasets import load_dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

dataset = load_dataset(
    "tiptoghosh/food-recipes-15k" , split = "train"
)

print("Initializing OmniChef-Nexus Retriever...")
retriever = RecipeRetriever(dataset = dataset)
print("Retriever Ready!")


def run_omni_search(text_query, image_query):
    if not text_query and image_query is None:
        return "Please provide at least a text description or an image."
    
    raw_results = retriever.match_query(
        text_query=text_query, 
        image_query=image_query, 
        top_k=6
    )
    
    gallery_items = []
    recipe_details = ""
    
    for res in raw_results:
        recipe = res['recipe']
        score = res['score']
        
        # Add to Gallery
        gallery_items.append((recipe['image'], f"Match Score: {score:.2f}"))
        recipe_details += f"{recipe['markdown']}\n"
        
    return gallery_items, recipe_details


# Gradio 6.0 expects theme in launch(), but can still be passed to Blocks for backward compatibility.
with gr.Blocks(theme = gr.themes.Soft()) as demo:
    gr.Markdown("# 🍳 OmniChef‑Nexus: Multimodal Recipe RAG")
    gr.Markdown(
        "Powered by **NVIDIA Llama Nemotron Embed VL**. Search by describing a dish or uploading ingredient photos."
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Describe what you want to cook...",
                placeholder="e.g., 'A healthy chicken salad with avocado'"
            )
            image_input = gr.Image(
                label="Or upload a photo of ingredients",
                type="pil"
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                search_btn = gr.Button("Find Recipes", variant="primary")

        with gr.Column(scale=2):
            result_gallery = gr.Gallery(
                label="Top Matches",
                columns=3,
                height="auto",
                object_fit="contain"
            )

    gr.Markdown("---")
    recipe_output = gr.Markdown("Search results will appear here...")

    # Set up interactions
    search_btn.click(
        fn=run_omni_search,
        inputs=[text_input, image_input],
        outputs=[result_gallery, recipe_output]
    )

    # Allow 'Enter' key to trigger search
    text_input.submit(
        fn=run_omni_search,
        inputs=[text_input, image_input],
        outputs=[result_gallery, recipe_output]
    )

    clear_btn.click(
        lambda: [None, None, [], "Search results will appear here..."],
        outputs=[text_input, image_input, result_gallery, recipe_output]
    )

    gr.Examples(
        examples=[
            ["A refreshing summer salad with berries", None],
            [None, "sample/example_ingredients.jpg"]
        ],
        inputs=[text_input, image_input]
    )


if __name__ == "__main__":
    demo.launch(theme = gr.themes.Soft())  