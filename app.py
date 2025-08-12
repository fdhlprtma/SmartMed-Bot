# # app.py
# import os, pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from llama_cpp import Llama

# # ---------- CONFIG ----------
# INDEX_PATH = "index/index.faiss"
# META_PATH = "index/index_meta.pkl"
# EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_PATH = "models/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"   # sesuaikan nama model
# TOP_K = 4
# SCORE_THRESHOLD = 0.3    # Turunkan threshold ke 0.3 supaya konteks relevan tetap dipakai
# MAX_TOKENS = 200
# # ----------------------------

# if not os.path.exists(INDEX_PATH):
#     print("Index tidak ditemukan. Jalankan: python build_index.py")
#     exit(1)
# if not os.path.exists(MODEL_PATH):
#     print(f"Model tidak ditemukan di {MODEL_PATH}. Masukkan model .gguf ke folder models/ atau jalankan download_model.py")
#     exit(1)

# print("Memuat embedding model:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)

# print("Memuat index FAISS...")
# index = faiss.read_index(INDEX_PATH)
# with open(META_PATH, "rb") as f:
#     metas = pickle.load(f)

# print("Memuat model LLM (llama-cpp)...")
# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

# def get_top_docs(question, k=TOP_K):
#     q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
#     q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
#     scores, ids = index.search(q_emb, k)
#     results = []
#     for sc, idx in zip(scores[0], ids[0]):
#         if idx != -1:
#             results.append((metas[idx]['text'], float(sc)))
#     return results

# def answer(question):
#     top = get_top_docs(question, TOP_K)
#     if not top:
#         return "Tidak ada data."

#     # Ambil dokumen yang skor >= threshold, jika tidak ada ambil dokumen dengan skor tertinggi
#     chosen = [(txt, sc) for txt, sc in top if sc >= SCORE_THRESHOLD]
#     if not chosen:
#         chosen = [top[0]]

#     print("DEBUG: Top docs for '{}':".format(question))
#     for txt, sc in chosen:
#         print(f"Score: {sc:.3f}, Text snippet: {txt[:60]}...")

#     context = "\n\n---\n\n".join([f"[score={sc:.3f}]\n{txt}" for txt, sc in chosen])

#     system_instr = (
#         "Kamu adalah asisten yang *hanya* boleh menjawab berdasarkan konteks yang diberikan. "
#         "Jika jawaban tidak ada di konteks, tulis persis: 'Tidak ada data.' "
#         "Jangan tambahkan informasi dari luar konteks."
#     )

#     prompt = (
#         f"{system_instr}\n\nKonteks:\n{context}\n\nPertanyaan: {question}\n\n"
#         "Jawab singkat dan hanya berdasarkan konteks. Jika tidak ada, tulis 'Tidak ada data.' "
#         "dan Jawab singkat, hanya gunakan kata-kata yang ada di konteks, tanpa tambahan apapun."
#     )

#     resp = llm.create_completion(
#         prompt=prompt,
#         max_tokens=MAX_TOKENS,
#         temperature=0.0
#     )
#     text = resp["choices"][0]["text"].strip()
#     return text if text else "Tidak ada data."

# if __name__ == "__main__":
#     print("AI siap. Ketik 'exit' untuk keluar.")
#     while True:
#         q = input("\nTanya: ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         print("Mencari konteks...")
#         jawaban = answer(q)
#         print("\nJawab:", jawaban)


# # app.py
# import os
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from llama_cpp import Llama

# # ---------- CONFIG ----------
# INDEX_PATH = "index/index.faiss"
# META_PATH = "index/index_meta.pkl"
# EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_PATH = "models/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"  # sesuaikan modelmu
# TOP_K = 4
# SCORE_THRESHOLD = 0.3  # turunkan threshold biar dapat konteks lebih banyak
# MAX_TOKENS = 200
# # ----------------------------

# if not os.path.exists(INDEX_PATH):
#     print("Index tidak ditemukan. Jalankan: python build_index.py")
#     exit(1)

# if not os.path.exists(MODEL_PATH):
#     print(f"Model tidak ditemukan di {MODEL_PATH}. Masukkan model .gguf ke folder models/ atau jalankan download_model.py")
#     exit(1)

# print("Memuat embedding model:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)

# print("Memuat index FAISS...")
# index = faiss.read_index(INDEX_PATH)

# with open(META_PATH, "rb") as f:
#     metas = pickle.load(f)

# print("Memuat model LLM (llama-cpp)...")
# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

# def get_top_docs(question, k=TOP_K):
#     q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
#     q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
#     scores, ids = index.search(q_emb, k)
#     results = []

#     print(f"DEBUG: Top docs for '{question}':")
#     for sc, idx in zip(scores[0], ids[0]):
#         if idx == -1:
#             continue
#         text = metas[idx]['text']
#         print(f"Score: {sc:.3f}, Text snippet: {text[:50]}...")
#         if sc >= SCORE_THRESHOLD:
#             results.append(text)

#     return results

# def answer(question):
#     contexts = get_top_docs(question)
#     if not contexts:
#         return "Tidak ada data."

#     # Gabungkan semua konteks relevan jadi satu teks (tanpa score)
#     combined_context = "\n\n---\n\n".join(contexts)

#     # Prompt yang super ketat: minta jawab HANYA dari teks konteks persis
#     prompt = (
#         "Kamu adalah asisten yang hanya boleh menjawab PERSIS dengan menyalin kalimat dari konteks berikut. "
#         "JIKA jawaban untuk pertanyaan TIDAK ADA di konteks, balas dengan 'Tidak ada data.' "
#         "JANGAN menambahkan informasi, opini, atau kalimat lain selain yang ada di konteks. "
#         "Jawab singkat dan langsung, tanpa mengulang pertanyaan atau konteks.\n\n"
#         f"Konteks:\n{combined_context}\n\n"
#         f"Pertanyaan: {question}\n\n"
#         "Jawaban:"
#     )

#     response = llm.create_completion(
#         prompt=prompt,
#         max_tokens=MAX_TOKENS,
#         temperature=0.0,
#         stop=["\n\n"]
#     )

#     text = response["choices"][0]["text"].strip()

#     # Bersihkan output dari artefak jika perlu (opsional)
#     if not text:
#         return "Tidak ada data."
#     # Jika model "ngarang" dan jawabannya diluar konteks, fallback:
#     if text.lower() in ["", "tidak ada data.", "no data."]:
#         return "Tidak ada data."
#     return text

# if __name__ == "__main__":
#     print("AI siap. Ketik 'exit' untuk keluar.")
#     while True:
#         q = input("\nTanya: ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         print("Mencari konteks...")
#         jawaban = answer(q)
#         print(f"\nTanya: {q}")
#         print(f"Jawab: {jawaban}")

# import os, pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from llama_cpp import Llama

# # ---------- CONFIG ----------
# INDEX_PATH = "index/index.faiss"
# META_PATH = "index/index_meta.pkl"
# EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_PATH = "models/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"
# TOP_K = 4
# SCORE_THRESHOLD = 0.3   # Turunkan threshold
# MAX_TOKENS = 200
# # ----------------------------

# if not os.path.exists(INDEX_PATH):
#     print("Index tidak ditemukan. Jalankan: python build_index.py")
#     exit(1)
# if not os.path.exists(MODEL_PATH):
#     print(f"Model tidak ditemukan di {MODEL_PATH}. Masukkan model .gguf ke folder models/ atau jalankan download_model.py")
#     exit(1)

# print("Memuat embedding model:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)

# print("Memuat index FAISS...")
# index = faiss.read_index(INDEX_PATH)
# with open(META_PATH, "rb") as f:
#     metas = pickle.load(f)

# print("Memuat model LLM (llama-cpp)...")
# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

# def get_top_docs(question, k=TOP_K):
#     q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
#     q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
#     scores, ids = index.search(q_emb, k)
#     results = []
#     for sc, idx in zip(scores[0], ids[0]):
#         if idx != -1:
#             results.append((metas[idx]['text'], float(sc)))
#     return results

# def answer(question):
#     top = get_top_docs(question, TOP_K)
#     if not top:
#         return "Tidak ada data."

#     print("DEBUG: Top docs:")
#     for i, (txt, sc) in enumerate(top, 1):
#         print(f"{i}. Score: {sc:.3f} - Text snippet: {txt[:80]}...")

#     if top[0][1] < SCORE_THRESHOLD:
#         return "Tidak ada data."

#     chosen = [(txt, sc) for txt, sc in top if sc >= SCORE_THRESHOLD] or [top[0]]
#     context = "\n\n---\n\n".join([f"[score={sc:.3f}]\n{txt}" for txt, sc in chosen])

#     system_instr = (
#     "Jawablah PERSIS hanya berdasarkan konteks berikut.\n"
#     "Jika informasi tidak ditemukan di konteks, tulis 'Tidak ada data.' saja.\n"
#     "Jangan menambahkan apa pun di luar konteks.\n"
#     "Jawaban harus singkat dan jelas."
#     )

#     prompt = f"""{system_instr}

# Konteks:
# {context}

# Pertanyaan:
# {question}

# Jawab:"""

#     resp = llm.create_completion(
#         prompt=prompt,
#         max_tokens=MAX_TOKENS,
#         temperature=0.0,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#     text = resp["choices"][0]["text"].strip()
#     return text if text else "Tidak ada data."


# if __name__ == "__main__":
#     print("AI siap. Ketik 'exit' untuk keluar.")
#     while True:
#         q = input("\nTanya: ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         print("Mencari konteks...")
#         jawaban = answer(q)
#         print("\nJawab:", jawaban)


# import os
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from llama_cpp import Llama

# # CONFIG
# INDEX_PATH = "index/index.faiss"
# META_PATH = "index/index_meta.pkl"
# EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_PATH = "models/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"
# TOP_K = 4
# SCORE_THRESHOLD = 0.6
# MAX_TOKENS = 80

# if not os.path.exists(INDEX_PATH):
#     print("Index tidak ditemukan. Jalankan: python build_index.py")
#     exit(1)
# if not os.path.exists(MODEL_PATH):
#     print(f"Model tidak ditemukan di {MODEL_PATH}. Masukkan model .gguf ke folder models/")
#     exit(1)

# print("Memuat embedding model:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)

# print("Memuat index FAISS...")
# index = faiss.read_index(INDEX_PATH)
# with open(META_PATH, "rb") as f:
#     metas = pickle.load(f)

# print("Memuat model LLM (llama-cpp)...")
# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

# def get_top_docs(question, k=TOP_K):
#     q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
#     q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9
#     scores, ids = index.search(q_emb, k)
#     results = []
#     for sc, idx in zip(scores[0], ids[0]):
#         if idx != -1:
#             results.append((metas[idx]['text'], float(sc)))
#     return results

# def answer(question):
#     top = get_top_docs(question, TOP_K)
#     if not top or top[0][1] < SCORE_THRESHOLD:
#         return "Tidak ada data."

#     chosen = [(txt, sc) for txt, sc in top if sc >= SCORE_THRESHOLD]
#     if not chosen:
#         return "Tidak ada data."

#     # Ambil maksimal 2 konteks terbaik saja
#     chosen = chosen[:2]
#     context = "\n\n---\n\n".join([txt for txt, _ in chosen])

#     system_instr = (
#         "Jawablah hanya berdasar konteks berikut.\n"
#         "Jika tidak ada jawabannya dalam konteks, jawab 'Tidak ada data.'\n"
#         "Jangan tambahkan informasi lain, jawab singkat dan jelas."
#     )

#     prompt = f"""{system_instr}

# Konteks:
# {context}

# Pertanyaan:
# {question}

# Jawab singkat:"""

#     resp = llm.create_completion(
#         prompt=prompt,
#         max_tokens=MAX_TOKENS,
#         temperature=0.0,
#     )
#     text = resp["choices"][0]["text"].strip()
#     if not text:
#         return "Tidak ada data."
#     return text

# if __name__ == "__main__":
#     print("AI siap. Ketik 'exit' untuk keluar.")
#     while True:
#         q = input("\nTanya: ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         print("Mencari konteks...")
#         jawaban = answer(q)
#         print(f"\nJawab: {jawaban}")



# import os
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from llama_cpp import Llama

# # Config
# INDEX_PATH = "index/index.faiss"
# META_PATH = "index/index_meta.pkl"
# EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_PATH = "models/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"
# TOP_K = 2
# SCORE_THRESHOLD = 0.6
# MAX_TOKENS = 150

# # Load resources
# if not os.path.exists(INDEX_PATH):
#     print("Index tidak ditemukan. Jalankan build_index.py terlebih dahulu.")
#     exit(1)
# if not os.path.exists(MODEL_PATH):
#     print(f"Model tidak ditemukan di {MODEL_PATH}. Pastikan model sudah ada.")
#     exit(1)

# print("Memuat embedding model:", EMBED_MODEL)
# embedder = SentenceTransformer(EMBED_MODEL)

# print("Memuat index FAISS...")
# index = faiss.read_index(INDEX_PATH)
# with open(META_PATH, "rb") as f:
#     metas = pickle.load(f)

# print("Memuat model LLM (llama-cpp)...")
# llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

# def get_top_docs(question, k=TOP_K):
#     q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
#     q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9
#     scores, ids = index.search(q_emb, k)
#     results = []
#     for sc, idx in zip(scores[0], ids[0]):
#         if idx == -1:
#             continue
#         results.append((metas[idx]['text'], float(sc)))
#     return results

# def answer(question):
#     top_docs = get_top_docs(question, TOP_K)
#     print(f"DEBUG: Top docs for '{question}':")
#     for i, (text, score) in enumerate(top_docs, 1):
#         print(f"{i}. Score: {score:.3f} - Text snippet: {text[:50]}...")

#     filtered_docs = [(text, score) for text, score in top_docs if score >= SCORE_THRESHOLD]

#     if not filtered_docs:
#         # turunkan threshold saat tidak ada data supaya cek kemungkinan lain
#         filtered_docs = [(text, score) for text, score in top_docs if score >= 0.3]
#         if not filtered_docs:
#             return "Tidak ada data."

#     context = "\n\n---\n\n".join([text for text, _ in filtered_docs])

#     prompt = f"""Jawablah hanya berdasarkan konteks berikut.
# Jika jawaban tidak ada dalam konteks, jawab 'Tidak ada data.'
# Jangan menambahkan informasi lain, jawab singkat dan jelas. namun jika dalam hal kalimat menyapa contohnya "Hai", maka jawablah tidak usah 
# menggunakan kata 'Tidak ada data.' dan langsung menjawab pertanyaan.

# Konteks :
# {context}

# Pertanyaan:
# {question}

# Jawaban singkat:"""

#     response = llm.create_completion(
#         prompt=prompt,
#         max_tokens=MAX_TOKENS,
#         temperature=0.0,
#         stop=["\n\n"]
#     )
#     text = response['choices'][0]['text'].strip()

#     if not text:
#         return "Tidak ada data."
#     return text


# if __name__ == "__main__":
#     print("AI siap. Ketik 'exit' untuk keluar.")
#     while True:
#         q = input("\nTanya: ").strip()
#         if q.lower() in ("exit", "quit"):
#             break
#         print("Mencari konteks...")
#         jawaban = answer(q)
#         print(f"\nJawab: {jawaban}")




import os
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama
from datetime import datetime

# ============= CONFIGURASI =============
INDEX_PATH = "index/index.faiss"
META_PATH = "index/index_meta.pkl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = "models/deepseek-coder-1.3b-instruct-Q4_K_M.gguf"
TOP_K = 5  # Ambil lebih banyak dokumen
SCORE_THRESHOLD = 0.45  # Threshold lebih rendah
MAX_TOKENS = 250  # Lebih banyak token
SAFETY_FILTER = True
SAFETY_RESPONSE = "Maaf, saya tidak dapat memberikan saran medis spesifik. Silakan konsultasikan dengan dokter."

# ============= INISIALISASI =============
class HealthAssistant:
    def __init__(self):
        self.embedder = self._load_embedder()
        self.index, self.metas = self._load_index()
        self.llm = self._load_llm()
        self.session_history = []
        self.conversation_context = []
        self.debug_mode = False  # Mode debug
        
    def _load_embedder(self):
        print("Memuat embedding model...")
        return SentenceTransformer(EMBED_MODEL)
    
    def _load_index(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError("Index tidak ditemukan. Harap jalankan build_index.py terlebih dahulu.")
        
        print("Memuat index FAISS...")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metas = pickle.load(f)
        return index, metas
    
    def _load_llm(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")
        
        print("Memuat model LLM...")
        return Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=40,
            verbose=False
        )
    
    # ============= CORE FUNCTIONALITY =============
    def _get_top_docs(self, question, k=TOP_K):
        q_emb = self.embedder.encode([question], convert_to_numpy=True).astype("float32")
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9
        scores, ids = self.index.search(q_emb, k)
        
        results = []
        for sc, idx in zip(scores[0], ids[0]):
            if idx == -1: 
                continue
            results.append((self.metas[idx]['text'], float(sc)))
        
        # Debug: tampilkan dokumen yang ditemukan
        if self.debug_mode:
            print(f"\n[DEBUG] Dokumen untuk: '{question}':")
            for i, (text, score) in enumerate(results):
                print(f"  Dokumen {i+1} (skor: {score:.2f}): {text[:100]}...")
        
        return results

    def _is_greeting(self, text):
        greetings = ["halo", "hai", "hi", "hei", "hallo"]
        return any(greet in text.lower() for greet in greetings)

    def _safety_check(self, question):
        """Filter pertanyaan yang berbahaya"""
        red_flags = [
            "operasi", "resep obat", "dosis", "suntik", 
            "diagnosa", "pengobatan", "penyakit", "gejala"
        ]
        return any(flag in question.lower() for flag in red_flags)

    def _extract_source(self, text):
        """Ekstrak sumber dari teks jika ada"""
        source_match = re.search(r"\[sumber: (.+?)\]", text)
        return source_match.group(1) if source_match else None

    def _clean_response(self, text):
        """Bersihkan respons dari pola tidak diinginkan"""
        # Hapus prefix yang tidak perlu
        for prefix in ["Jawaban:", "Asisten:", "AI:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Hapus pola template
        text = re.sub(r"###\s*[A-Z]+:.*", "", text)
        
        # Hapus whitespace berlebihan
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # ============= MAIN INTERFACE =============
    def answer(self, question):
        # 1. Handle greetings
        if self._is_greeting(question):
            return "Hai! Saya Asisten Informasi Kesehatan. Ada yang bisa saya bantu?"
        
        # 2. Safety check pada input
        if SAFETY_FILTER and self._safety_check(question):
            return SAFETY_RESPONSE
        
        # 3. Retrieve relevant documents
        top_docs = self._get_top_docs(question)
        
        # 4. Jika tidak ada dokumen relevan
        if not top_docs:
            return "Maaf, saya belum menemukan informasi tentang itu."
        
        # 5. Bangun context dari dokumen terbaik
        context = ""
        references = []
        for i, (text, score) in enumerate(top_docs):
            # Gunakan dokumen dengan skor memadai
            if score >= SCORE_THRESHOLD:
                # Simpan sumber untuk referensi
                if source := self._extract_source(text):
                    references.append(f"{i+1}. {source}")
                
                # Tambahkan ke konteks
                context += f"{text}\n\n"
        
        # 6. Jika tidak ada konteks yang memenuhi threshold
        if not context:
            return "Maaf, informasi yang tersedia tidak cukup relevan."
        
        # 7. Bangun prompt dengan instruksi ketat
        prompt = f"""### INSTRUKSI:
Anda adalah asisten informasi kesehatan. JAWABLAH PERTANYAAN HANYA BERDASARKAN INFORMASI BERIKUT.
JANGAN MEMBUAT JAWABAN JIKA TIDAK ADA DALAM INFORMASI.
JANGAN MENAMBAHKAN INFORMASI BARU.
GUNAKAN BAHASA INDONESIA SEDERHANA.

### INFORMASI:
{context}

### PERTANYAAN:
{question}

### JAWABAN:
"""
        
        # 8. Generate response
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.2,  # Rendah untuk mengurangi halusinasi
            top_p=0.5,
            stop=["\n\n", "###", "PERTANYAAN", "INFORMASI"]
        )
        
        answer_text = response['choices'][0]['text'].strip()
        answer_text = self._clean_response(answer_text)
        
        # 9. Tambahkan referensi jika ada
        if references:
            answer_text += "\n\nReferensi:\n" + "\n".join(references[:3])
        
        # 10. Maintain session history
        self.session_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": answer_text
        })
        
        return answer_text


# ============= MAIN APPLICATION =============
if __name__ == "__main__":
    print("Inisialisasi Asisten Kesehatan...")
    assistant = HealthAssistant()
    
    # Aktifkan debug mode untuk melihat dokumen yang diambil
    assistant.debug_mode = True
    
    print("\n=== Sistem Siap ===")
    print("Ketik 'menu' untuk opsi tambahan")
    print("Ketik 'keluar' untuk mengakhiri\n")
    
    while True:
        question = input("Pertanyaan Anda: ").strip()
        
        if question.lower() in ("exit", "quit", "keluar"):
            break
            
        if question.lower() == "menu":
            print("\nMenu Tambahan:")
            print("1. Setel ulang konteks")
            print("2. Lihat riwayat")
            print("3. Mode konsultasi dokter")
            print("4. Toggle debug mode")
            choice = input("Pilihan: ").strip()
            
            if choice == "1":
                assistant.session_history = []
                print("=> Konteks percakapan direset")
            elif choice == "2":
                print("\nRiwayat Percakapan:")
                for i, entry in enumerate(assistant.session_history, 1):
                    time_str = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M")
                    print(f"{i}. [{time_str}] Q: {entry['question']}")
                    print(f"   A: {entry['response'][:200]}")
                print()
            elif choice == "3":
                print("\n=> Mode Konsultasi Dokter: Silakan kunjungi fasilitas kesehatan terdekat")
                print("Contoh rumah sakit di Jakarta:")
                print("- RSUD Pasar Minggu, Jl. TB Simatupang No.1, (021) 1234567")
                print("- RS Premier Bintaro, Jl. MH Thamrin No.1, (021) 7654321")
                print("Untuk konsultasi online: 1500-123 (Telemedicine Nasional)")
            elif choice == "4":
                assistant.debug_mode = not assistant.debug_mode
                status = "AKTIF" if assistant.debug_mode else "NONAKTIF"
                print(f"=> Debug mode {status}")
            continue
        
        print("\nMemproses...")
        response = assistant.answer(question)
        print(f"\nAsisten: {response}\n")