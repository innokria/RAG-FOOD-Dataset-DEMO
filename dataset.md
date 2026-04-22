# Food.com Multimodal Recipe Dataset (15K)

> **Cross-linking Note**
> - **Arriving from GitHub?** → Download the dataset on Hugging Face: [tiptoghosh/food-recipes-15k](https://huggingface.co/datasets/tiptoghosh/food-recipes-15k)
> - **Arriving from Hugging Face?** → Explore the source code on GitHub: [OmniChef-Nexus](https://github.com/Tipto-Ghosh/OmniChef-Nexus)

A curated, high-quality multimodal recipe dataset engineered for training and evaluating vision-language models (VLMs) and multimodal retrieval-augmented generation (RAG) systems. Each sample pairs a **rendered recipe card image** (PNG, 300 DPI) with a **structured Markdown representation** of the same recipe, enabling cross-modal retrieval, VLM fine-tuning, and document understanding research.

---

## Dataset Summary

| Property | Value |
|:---|:---|
| **Total samples** | ~15,000 |
| **Modalities per sample** | 2 — PNG recipe card image + Markdown text |
| **Image format** | PNG, 300 DPI, A4 aspect ratio |
| **Source dataset** | Food.com Recipes and User Interactions (Kaggle) |
| **Raw recipe pool** | ~231,637 recipes |
| **License** | See source dataset license |

---

## Intended Use Cases

This dataset was designed to support the following downstream research and engineering tasks:

- **Multimodal RAG** — Visual and text-based recipe retrieval using vision-language embedding models (e.g., `nvidia/llama-nemotron-embed-vl-1b-v2`)
- **VLM fine-tuning** — Structured document understanding, ingredient extraction, and nutritional reasoning
- **Cross-modal retrieval** — Ingredient-based, nutrition-aware, and image-based recipe search
- **Document layout understanding** — Recipe cards follow a consistent yet visually diverse template across 10 colour schemes and 4 chart styles

---

## Data Sources

Two raw files from the [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data) Kaggle dataset were used as input:

| File | Description | Raw Size |
|:---|:---|:---|
| `RAW_recipes.csv` | Recipe metadata, ingredients, steps, nutrition, tags | ~231,637 rows |
| `RAW_interactions.csv` | User ratings and free-text reviews per recipe | Several hundred thousand rows |

---

## Dataset Construction Pipeline

The dataset was built using a six-stage reproducible pipeline.

### Stage 1 — Data Loading and Exploratory Analysis

Both raw CSVs were loaded via pandas. A quality audit was performed covering missing value counts, duplicate recipe detection, and distribution analysis of `n_steps` and `n_ingredients`. Coverage was measured across a range of filtering thresholds (5, 8, 10, 12, 15, 18, 20, 25) to inform downstream cutoff selection. All rows containing null values were dropped prior to further processing.

### Stage 2 — Interaction Signal Aggregation

The interactions table was aggregated at the recipe level. For each recipe, the mean star rating across all reviewers was computed, individual review texts were collected into a list (nulls excluded), and the total number of distinct user ratings was counted. This aggregated table was left-joined onto the recipes DataFrame on `recipe_id`. The original `id` column was subsequently dropped, with `recipe_id` serving as the canonical identifier for all downstream stages.

### Stage 3 — Filtering and Sampling

The full merged corpus of ~231,637 recipes was reduced to the final ~15,000 subset using the following criteria applied in sequence:

| Step | Criterion | Rationale |
|:---|:---|:---|
| 1 | Sort by `rating` descending | Prioritises higher-quality, well-reviewed recipes |
| 2 | `n_ingredients ≤ 15` | Ensures ingredient lists render cleanly on a single card |
| 3 | `n_steps ≤ 15` | Prevents step overflow across card pages |
| 4 | `num_ratings > 10` | Requires a minimum review signal to filter low-confidence recipes |
| 5 | Drop remaining null rows | Final cleanliness pass before checkpoint serialisation |

Two columns unused downstream (`contributor_id`, `submitted`) were dropped at this stage. The filtered subset was saved as an intermediate CSV checkpoint.

### Stage 4 — Review Text Cleaning

Raw review strings from the interactions table contained HTML entities, encoding artifacts, and short noise strings. Each string was normalised in the following order: HTML entity decoding, non-ASCII character stripping, whitespace and newline collapsing, and filtering of strings shorter than 5 characters. Cleaned reviews were stored back into the DataFrame as Python lists. Up to 5 reviews were used for the Markdown text modality; up to 10 (sampled deterministically by row index) were used for the recipe card image.

### Stage 5 — Markdown Text Modality Generation

Each row was serialised into a structured Markdown string constituting the text modality of the sample. The schema covers the following fields in a fixed order:

- Recipe title, recipe ID, cook time
- Star rating and total review count
- Description paragraph
- Comma-separated ingredient list
- Numbered preparation steps
- Compact nutrition summary: Calories, Total Fat, Sugar, Sodium, Protein, Saturated Fat
  *(Note: the 7th nutrition element, Total Carbs, is intentionally omitted from the compact summary)*
- Top 5 cleaned user reviews

This Markdown string is stored in the `markdown` column and is intended to be embedded directly alongside the image for multimodal retrieval.

### Stage 6 — Recipe Card Image Modality Generation

Each recipe was rendered as a styled single-page A4 PDF using a custom multi-module Python pipeline (`config.py`, `utils.py`, `charts.py`, `builder.py`, `pipeline.py`). The PDF was subsequently rasterised to PNG at 300 DPI using PyMuPDF (no Poppler dependency required). Images are stored at `data/output/images/recipe_id_<ID>.png`.

**Visual variation system:** To maximise visual diversity across the 15,000 cards, each recipe card is assigned a unique visual style determined by its row index. The system combines 10 colour schemes with 4 nutrition chart types, yielding 40 base style combinations. Assignments are seeded by row index, ensuring full reproducibility across pipeline runs.

The 10 colour schemes are: `WarmSpice`, `FreshHerb`, `OceanBreeze`, `MidnightChef`, `BerryFusion`, `SunsetGrill`, `SlateCuisine`, `TropicalBowl`, `RusticBakery`, and `CrimsonFeast`.

The 4 nutrition chart variants are: vertical bar chart, horizontal bar chart, radar chart, and table-only (no chart).

**Card layout:** Each card contains the following sections — header (title, cook time, star rating), ingredient list, numbered preparation steps, nutrition panel (with the assigned chart variant), and a review section (up to 10 cleaned reviews). If a required field is missing or unparseable, that section is omitted rather than rendered as a placeholder.

**Multi-page overflow handling:** If a card exceeded one page, the pipeline automatically reduced the review count and, if necessary, applied progressive font scaling until the card fit within a single page or a minimum scale threshold was reached. Only the first page was exported to PNG.

**Failed renders:** 12 recipe IDs failed during image generation and were excluded from the final dataset. Additional rows where the expected PNG was not found on disk were also excluded, producing the final ~15,000 sample count.

---

## Dataset Schema

| Column | Type | Description |
|:---|:---|:---|
| `name` | `string` | Recipe name (title-cased) |
| `recipe_id` | `int64` | Unique recipe identifier |
| `minutes` | `int64` | Total preparation and cook time in minutes |
| `description` | `string` | Recipe description text |
| `tags` | `list[string]` | Descriptive tags (up to 8 per card) |
| `steps` | `list[string]` | Ordered preparation steps (numbered, sentence-cased) |
| `n_steps` | `int64` | Number of preparation steps (≤ 15) |
| `ingredients` | `list[string]` | List of ingredient strings |
| `n_ingredients` | `int64` | Number of ingredients (≤ 15) |
| `nutrition` | `struct` | Nutritional values as a labelled struct of `float32` fields |
| `nutrition.calories` | `float32` | Energy in kilocalories |
| `nutrition.total_fat_pdv` | `float32` | Total fat as % Daily Value |
| `nutrition.sugar_pdv` | `float32` | Sugar as % Daily Value |
| `nutrition.sodium_pdv` | `float32` | Sodium as % Daily Value |
| `nutrition.protein_pdv` | `float32` | Protein as % Daily Value |
| `nutrition.saturated_fat_pdv` | `float32` | Saturated fat as % Daily Value |
| `nutrition.carbs_pdv` | `float32` | Total carbohydrates as % Daily Value |
| `rating` | `float32` | Mean star rating across all reviewers (1.0–5.0) |
| `num_ratings` | `float32` | Total number of user ratings (> 10 for all samples) |
| `markdown` | `string` | Full structured Markdown recipe (text modality) |
| `image` | `Image` | Rendered recipe card PNG at 300 DPI (image modality) |

---

## Nutrition Values — Important Note

Nutritional values are sourced directly from the original Food.com dataset and are expressed as **Percent Daily Values (PDV)** rather than absolute gram weights, with the exception of Calories (reported in kilocalories). PDV values represent the percentage of the recommended daily intake contributed by one serving of the recipe, based on a 2,000-calorie reference diet.

---

## Data Quality Summary

| Metric | Value |
|:---|:---|
| Raw recipe pool | ~231,637 |
| After quality filtering (`n_steps ≤ 15`, `n_ingredients ≤ 15`, `num_ratings > 10`) | ~15,700 |
| Excluded — failed image renders | 12 |
| Excluded — image file missing on disk | Varies |
| **Final dataset size** | **~15,000 samples** |
| Modalities per sample | 2 (PNG image + Markdown text) |
| Minimum ratings per recipe | > 10 |
| Image resolution | 300 DPI (A4) |

---

## Recommended Embedding Model

This dataset was designed for use with [`nvidia/llama-nemotron-embed-vl-1b-v2`](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2), a 1B-parameter vision-language embedding model producing 2048-dimensional vectors from both image and text inputs. Recipe card images are intended to be embedded directly as document images, while Markdown strings can be embedded as text or incorporated into a combined multimodal input.

---

## Repository Structure

```
data/
├── all csv files/
│   ├── RAW_recipes.csv              # Original Kaggle recipes file
│   ├── RAW_interactions.csv         # Original Kaggle interactions file
│   └── recipes_15k_samples.csv      # Intermediate filtered CSV checkpoint
└── output/
    └── images/
        └── recipe_id_<ID>.png       # Rendered recipe card images (300 DPI PNG)
```

---

## Dependencies

| Library | Purpose |
|:---|:---|
| `pandas` | Data loading, merging, filtering, and serialisation |
| `ast` | Safe parsing of stringified Python lists from CSV columns |
| `re`, `html` | Review text cleaning and HTML entity decoding |
| `reportlab` | PDF layout generation for recipe cards |
| `PyMuPDF (fitz)` | PDF-to-PNG rasterisation at 300 DPI (no Poppler required) |
| `Pillow (PIL)` | Image loading, colour conversion, and JPEG encoding |
| `numpy` | Numerical type handling and NaN detection |
| `datasets` | Hugging Face Dataset construction and Hub upload |
| `tqdm` | Progress tracking across the generation pipeline |

---

## Citation

If you use this dataset in your research, please cite **both** the original Food.com source dataset and this curated multimodal version.

### Original Source — Food.com Recipes and User Interactions

```bibtex
@inproceedings{majumder2019generating,
  title     = {Generating Personalized Recipes from Historical User Preferences},
  author    = {Majumder, Bodhisattwa Prasad and Li, Shuyang and Ni, Jianmo and McAuley, Julian},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
               and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year      = {2019}
}
```

### This Dataset

```bibtex
@misc{ghosh2025food15k,
  author       = {Tipto Ghosh},
  title        = {Food.com Multimodal Recipe Dataset (15K)},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/tiptoghosh/food-recipes-15k}},
  note         = {Curated multimodal recipe dataset with rendered recipe card images and
                  structured Markdown text representations.
                  Project repository: \url{https://github.com/Tipto-Ghosh/OmniChef-Nexus}.
                  Author profile: \url{https://scholar.google.com/citations?user=tiptoghosh}}
}
```