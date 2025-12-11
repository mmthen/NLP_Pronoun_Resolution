import os
import requests
from langdetect import detect

# Map of topic → list of Gutenberg IDs
corpus_ids = {
    "literature": [1342, 1260, 1400, 768, 98, 514, 33, 84, 345, 110],
    "philosophy": [4363, 1998, 1497, 3800, 2680, 59, 3207, 147, 11224, 34901],
    "science": [1228, 5001, 944, 15491, 29398, 20417, 14474, 27338, 47993, 33537],
    "psychology": [28422, 620, 63100, 67120, 43986, 14005, 15396, 19144, 40155, 21607],
    "adventure": [120, 2701, 215, 3748, 164, 103, 2166, 236, 139, 1268]
}

base_url_template = "https://www.gutenberg.org/files/{id}/{id}-0.txt"


# create output directories
root_dir = "data/raw"
os.makedirs(root_dir, exist_ok=True)

def clean_text(text):
    """Remove Project Gutenberg license header + footer"""
    start = text.find("*** START")
    if start == -1:
        start = 0
    else:
        start = text.find("\n", start) + 1

    end = text.find("*** END")
    if end == -1:
        end = len(text)

    cleaned = text[start:end]
    return cleaned.strip()

def try_download(gid):
    """Try several common Gutenberg filename patterns for a given ID."""
    url_patterns = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]

    for url in url_patterns:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception:
            continue

    return None


for topic, ids in corpus_ids.items():
    topic_path = os.path.join(root_dir, topic)
    os.makedirs(topic_path, exist_ok=True)

    for gid in ids:
        print(f"Downloading {gid} into {topic} ...")
        raw_text = try_download(gid)

        if raw_text is None:
            print("FAILED:", gid)
            continue

        text = clean_text(raw_text)

        try:
            lang = detect(text[:5000])  # detect language using first 5k chars
        except Exception:
            lang = "unknown"

        if lang != "en":
            print(f"Skipping {gid} (language detected = {lang})")
            continue

        out_file = os.path.join(topic_path, f"{gid}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)

print("DONE — all documents downloaded + cleaned!")
