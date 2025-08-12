# # build_index.py
# import os, glob, pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from tqdm import tqdm

# DATA_DIR = "data"
# INDEX_DIR = "index"
# INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
# META_PATH = os.path.join(INDEX_DIR, "index_meta.pkl")

# EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # mendukung Bahasa Indonesia
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 100

# os.makedirs(INDEX_DIR, exist_ok=True)
# os.makedirs(DATA_DIR, exist_ok=True)

# def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     text = text.replace("\r\n", "\n")
#     chunks = []
#     start = 0
#     L = len(text)
#     while start < L:
#         end = min(start + size, L)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         start += size - overlap
#     return chunks

# # 1. collect files
# all_chunks = []
# metas = []
# files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
# if not files:
#     print("Tidak ada file .txt di folder 'data/'. Tambahkan file lalu jalankan lagi.")
#     exit(0)

# for filepath in files:
#     name = os.path.basename(filepath)
#     with open(filepath, "r", encoding="utf-8") as f:
#         txt = f.read()
#     chunks = chunk_text(txt)
#     for i, ch in enumerate(chunks):
#         metas.append({"source": name, "chunk_id": i, "text": ch})
#         all_chunks.append(ch)

# print(f"Total file: {len(files)}, total chunk: {len(all_chunks)}")

# # 2. embeddings
# print("Memuat model embedding:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)

# batch_size = 64
# embs = []
# for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
#     batch = all_chunks[i:i+batch_size]
#     e = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
#     embs.append(e)
# embeddings = np.vstack(embs).astype("float32")

# # 3. normalisasi untuk cosine
# norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
# norms[norms==0] = 1e-8
# embeddings = embeddings / norms

# # 4. buat index FAISS (inner product on normalized vectors => cosine)
# dim = embeddings.shape[1]
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings)
# faiss.write_index(index, INDEX_PATH)

# # 5. simpan metadata
# with open(META_PATH, "wb") as f:
#     pickle.dump(metas, f)

# print("Index tersimpan di:", INDEX_PATH)
# print("Metadata tersimpan di:", META_PATH)

import os, glob, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

DATA_DIR = "data"
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "index_meta.pkl")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # mendukung Bahasa Indonesia
CHUNK_SIZE = 150       # chunk lebih kecil supaya embedding lebih fokus
CHUNK_OVERLAP = 30     # overlap untuk konteks lanjutan

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks

# 1. collect files
all_chunks = []
metas = []
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
if not files:
    print("Tidak ada file .txt di folder 'data/'. Tambahkan file lalu jalankan lagi.")
    exit(0)

for filepath in files:
    name = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        txt = f.read()
    chunks = chunk_text(txt)
    for i, ch in enumerate(chunks):
        metas.append({"source": name, "chunk_id": i, "text": ch})
        all_chunks.append(ch)

print(f"Total file: {len(files)}, total chunk: {len(all_chunks)}")

# 2. embeddings
print("Memuat model embedding:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

batch_size = 64
embs = []
for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
    batch = all_chunks[i:i+batch_size]
    e = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embs.append(e)
embeddings = np.vstack(embs).astype("float32")

# 3. normalisasi untuk cosine
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms==0] = 1e-8
embeddings = embeddings / norms

# 4. buat index FAISS (inner product on normalized vectors => cosine similarity)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# 5. simpan metadata
with open(META_PATH, "wb") as f:
    pickle.dump(metas, f)

print("Index tersimpan di:", INDEX_PATH)
print("Metadata tersimpan di:", META_PATH)
