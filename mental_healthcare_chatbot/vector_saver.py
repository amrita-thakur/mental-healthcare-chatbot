from dotenv import load_dotenv

from document_utils import load_and_split_documents
from db import save_to_faiss
from llm import recursive_embed_cluster_summarize

load_dotenv(override=True)

# Load and split documents
docs_texts = load_and_split_documents()

# Build tree
leaf_texts = docs_texts
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

# Initialize all_texts with leaf_texts
all_texts = leaf_texts.copy()

# Iterate through the results to extract summaries from each level and add them to all_texts
for level in sorted(results.keys()):
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)

save_to_faiss(all_texts)

