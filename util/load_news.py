"""
 This script downloads news articles from HuggingFace's news API.
 It will download a balanced article dataset which include 10 documents
 from 5 different categories. 
"""
import os
import random
from datasets import load_dataset

"""
Download a balanced set of CNN/DailyMail news articles.
We select 50 documents: 10 per topic.
"""

import os
from datasets import load_dataset
from collections import defaultdict

print("Downloading CNN/DailyMail dataset...")
ds = load_dataset("cnn_dailymail", "3.0.0")

articles = ds["train"]  # large dataset

# Define topics using keyword search inside article text
TOPIC_KEYWORDS = {
    "POLITICS": ["president", "congress", "government", "election"],
    "BUSINESS": ["company", "market", "economy", "trade"],
    "SPORTS": ["game", "team", "tournament", "athlete"],
    "TECH": ["technology", "software", "internet", "device"],
    "WORLD": ["international", "global", "foreign", "country"]
}

NUM_PER_TOPIC = 10  # 10 per category → 50 total

os.makedirs("data", exist_ok=True)

selected = defaultdict(int)
doc_index = 1

print("Filtering and saving articles...")

for item in articles:
    text = item["article"].strip().replace("\n", " ")

    # Check which topic this article belongs to
    for topic, keywords in TOPIC_KEYWORDS.items():
        if selected[topic] >= NUM_PER_TOPIC:
            continue

        if any(kw.lower() in text.lower() for kw in keywords):
            # save article
            with open(f"data/doc_{doc_index}.txt", "w", encoding="utf-8") as f:
                f.write(text)

            selected[topic] += 1
            doc_index += 1

            print(f"Saved {topic} ({selected[topic]}/{NUM_PER_TOPIC})")

            break  # avoid double-counting an article

    # Stop when all 50 docs collected
    if sum(selected.values()) == NUM_PER_TOPIC * len(TOPIC_KEYWORDS):
        break

print("\nDONE — Saved 50 CNN news articles into ./data/")
print("Category counts:", dict(selected))
