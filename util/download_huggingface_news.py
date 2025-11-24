"""
 This script downloads news articles from HuggingFace's news API.
 It will download a balanced article dataset which include 10 documents
 from 5 different categories. 
"""
import os
from collections import defaultdict

from datasets import load_dataset

print('Downloading HuggingFace news dataset...')

ds = load_dataset("heegyu/news-category-dataset")

data = ds["train"]

# Choose 5 categories
TARGET_CATEGORIES = ["POLITICS",
                     "ENTERTAINMENT",
                     "SPORTS",
                     "TECH",
                     "TRAVEL"]

NUM_CATEGORIES = 10  # 10 documents with 5 topics = 50 docs
os.makedirs("../data", exist_ok=True)

selected_categories = defaultdict(int)
document_index = 1  # doc count

for item in data:
    category = item["category"]
    text = item["short_description"]

    if category in TARGET_CATEGORIES:
        if selected_categories[category] < NUM_CATEGORIES:
            with open(f"data/doc_{document_index}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            selected_categories[category] += 1
            document_index += 1
            print(f"Saved {category}, ({selected_categories[category]}/{NUM_CATEGORIES})")

        # Stop at the number of categories defined in NUM_CATEGORIES
        if sum(selected_categories.values()) == NUM_CATEGORIES * len(TARGET_CATEGORIES):
            break

print("\nDONE: Huggingface dataset created in ./data/")
print("Category counts:", dict(selected_categories))

