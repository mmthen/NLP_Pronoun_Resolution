

"""
Load documents from a CSV file.
"""
import os


def load_documents(data_dir="util/data"):
    documents = []
    filenames = []

    for f_name in sorted(os.listdir(data_dir)):
        if f_name.endswith(".txt"):
            with open(os.path.join(data_dir, f_name), "r", encoding="utf-8") as f:
                documents.append(f.read())
                filenames.append(f_name)

    print(f"Loaded {len(documents)} documents from {data_dir}.")
    return documents, filenames


