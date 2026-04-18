import gradio as gr
from utils import RecipeRetriever

# ----------------------------------------------------------------------
# Initialize the retriever once (caches model and dataset)
# ----------------------------------------------------------------------
print("Loading dataset and embeddings...")
retriever = RecipeRetriever()
print("Ready!")


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


# ----------------------------------------------------------------------
# Build Gradio interface
# ----------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍳 OmniChef‑Nexus: Multimodal Recipe RAG")
    gr.Markdown(
        "Search for recipes using **text**, an **image**, or **both**!\n"
        "The system uses NVIDIA's Llama Nemotron Embed VL model to find the best matches."
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
            search_btn = gr.Button("Find Recipes", variant="primary")

        with gr.Column(scale=2):
            result_gallery = gr.Gallery(
                label="Top Matches",
                columns=3,
                height="auto",
                object_fit="contain"
            )

    gr.Markdown("## Detailed Instructions")
    recipe_output = gr.Markdown("Search results will appear here...")

    search_btn.click(
        fn=run_omni_search,
        inputs=[text_input, image_input],
        outputs=[result_gallery, recipe_output]
    )

# ----------------------------------------------------------------------
# Launch (compatible with Hugging Face Spaces)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)