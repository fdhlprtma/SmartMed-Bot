# SmartMed Bot - Sistem Q&A Kesehatan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Sistem chatbot kesehatan berbasis Retrieval-Augmented Generation (RAG) yang memberikan informasi medis berdasarkan basis pengetahuan terstruktur.

## Fitur Utama

- 🩺 Memberikan informasi kesehatan berdasarkan sumber terpercaya
- 🔍 Pencarian dokumen dengan teknologi FAISS
- 🤖 Generasi jawaban menggunakan model LLM
- ⚠️ Filter keamanan untuk pertanyaan medis sensitif
- 📊 Riwayat percakapan dan manajemen konteks

## Persyaratan Sistem

- Python 3.9+
- RAM 8GB+ (16GB direkomendasikan)
- GPU opsional (untuk percepatan inferensi)

## Instalasi

1. Clone repositori:
    ```bash
    git clone https://github.com/fdhlprtma/SmartMed-Bot.git
    cd rag_health_assistant
2. Buat environment virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate    # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

**Penggunaan**
***Persiapan Data***
1. Siapkan dokumen teks (.txt) dalam folder data/
2. Format dokumen:
   ```bash
   [isi konten kesehatan] [sumber: nama_sumber]
    Contoh:
    AntriPintar adalah sistem antrian digital rumah sakit. [sumber: Manual AntriPintar v2.1]
3. Bangun index:
   ```bash
   python build_index.py

**Menjalankan Aplikasi**
```bash
python app.py
```

**Konfigurasi**
Edit app.py untuk penyesuaian:
```bash
# Konfigurasi utama
INDEX_PATH = "index/index.faiss"
MODEL_PATH = "models/llm_model.gguf"
TOP_K = 5                   # Jumlah dokumen yang diambil
SCORE_THRESHOLD = 0.45      # Threshold relevansi dokumen
```

**Model yang Didukung**
1. DeepSeek-Coder (default)
2. Meditron-7B
3. BioMistral-7B

## ⚠️ Catatan Penting: Status Proyek Saat Ini

Proyek ini masih dalam pengembangan aktif dan **belum mencapai tingkat performa maksimal**. Beberapa keterbatasan yang perlu diketahui:

### Keterbatasan Saat Ini
1. **Akurasi Informasi**:
   - Jawaban yang diberikan mungkin masih mengandung ketidakakuratan
   - Sistem bisa memberikan respons yang tidak relevan untuk pertanyaan kompleks

2. **Cakupan Pengetahuan**:
   - Hanya mencakup informasi yang ada dalam dokumen training
   - Belum bisa menjawab pertanyaan di luar basis pengetahuan yang dimasukkan

3. **Model Dasar**:
   - Menggunakan DeepSeek-Coder yang bukan model khusus kesehatan
   - Kemampuan pemahaman konteks medis terbatas

4. **Masalah Teknis**:
   - Terkadang memberikan jawaban terpenggal (cut-off)
   - Masih ada isu dengan pemrosesan pertanyaan yang mirip tapi memberikan jawaban berbeda

### Area Pengembangan Selanjutnya
1. **Peningkatan Model**:
   - Migrasi ke model khusus kesehatan seperti [Meditron-7B](https://huggingface.co/epfl-llm/meditron-7b)
   - Fine-tuning model untuk domain kesehatan Indonesia

2. **Penyempurnaan Sistem**:
   - Implementasi post-processing jawaban yang lebih baik
   - Penambahan mekanisme feedback untuk koreksi jawaban
   - Sistem verifikasi fakta multi-sumber

3. **Ekspansi Fitur**:
   - Integrasi dengan API kesehatan terpercaya
   - Penambahan multilingual support
   - Sistem follow-up question yang lebih cerdas

4. **Peningkatan Data**:
   - Penambahan sumber pengetahuan yang lebih komprehensif
   - Kurasi dokumen yang lebih ketat

📌 **Disclaimer**: Informasi yang diberikan oleh sistem ini tidak boleh dianggap sebagai saran medis profesional. Selalu konsultasikan dengan dokter untuk masalah kesehatan yang sebenarnya.
