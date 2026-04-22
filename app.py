import gradio as gr
from utils import RecipeRetriever
import os 

print("Initializing OmniChef-Nexus Retriever...")
retriever = RecipeRetriever()
print("Retriever Ready!")


def run_omni_search(text_query, image_query):
    """
    Gradio callback: takes text and optional image, returns gallery and markdown.
    """
    # Handle empty queries
    if not text_query and image_query is None:
        return [], "Please provide at least a text description or an image."

    # Perform search
    try:
        raw_results = retriever.match_query(
            text_query=text_query,
            image_query=image_query,
            top_k=6
        )
    except Exception as e:
        return [], f"**Error during search:** {str(e)}"

    # Build gallery and markdown
    gallery_items = []
    recipe_details = ""

    for res in raw_results:
        recipe = res["recipe"]
        score = res["score"]

        # Gallery entry: (PIL image, caption)
        gallery_items.append((recipe["image"], f"Match Score: {score:.3f}"))

        # Markdown details
        recipe_details += f"### {res['rank']}. {recipe['name']}\n"
        recipe_details += f"**Match Score:** {score:.3f}\n\n"
        recipe_details += f"**Ingredients:** {recipe['ingredients']}\n\n"
        # Use the pre‑formatted markdown recipe
        recipe_details += f"{recipe['markdown_recipe']}\n"
        recipe_details += "---\n"

    if not gallery_items:
        return [], "No matching recipes found."

    return gallery_items, recipe_details


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍳 OmniChef‑Nexus: Multimodal Recipe RAG")
    gr.Markdown(
        "Powered by **NVIDIA Llama Nemotron Embed VL**. Search by describing a dish or uploading ingredient photos."
    )

    with gr.Row():
        with gr.Column(scale = 1):
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
    demo.launch()