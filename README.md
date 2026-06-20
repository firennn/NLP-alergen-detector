# NLP Alergen Detector — Perbandingan 4 Model

Aplikasi deteksi alergen makanan dari teks bahan makanan Indonesia menggunakan 4 pendekatan NLP yang dibandingkan secara langsung.

## Model yang Dibandingkan

| Model | Tipe | Gold F1 |
|-------|------|---------|
| M1: Dictionary NER | Rule-based | 0.783 |
| M2: TF-IDF + Logistic Regression | Classical ML | 0.347 |
| M3: mBERT NER | Multilingual DL | 0.674 |
| M4: IndoBERT NER | Indonesian DL | 0.741 |

Evaluasi dilakukan pada 137 produk Indonesia yang dianotasi manual (gold label).

## Kategori Alergen

8 kategori berdasarkan regulasi BPOM Indonesia dan EU FIR No. 1169/2011:
`GLUTEN` · `SUSU` · `TELUR` · `KACANG` · `KEDELAI` · `SEAFOOD` · `WIJEN` · `SULFITE`

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Untuk fitur OCR (opsional), install PaddlePaddle terlebih dahulu dari mirror resmi:

```bash
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

### 2. Siapkan model

Buat folder `model/` di root project dan letakkan model di dalamnya:

```
NLP 4 Model Comparison/
├── app/
│   └── app_compare.py
├── model/
│   ├── model_alergen_final_v3/   ← IndoBERT (M4)
│   └── model_mbert_alergen/      ← mBERT (M3)
└── requirements.txt
```

Model M4 (IndoBERT) dan M3 (mBERT) dapat diunduh dari Google Drive tim.
Model M1 (Dictionary) dan M2 (TF-IDF) tidak memerlukan file model — berjalan otomatis.

### 3. Jalankan aplikasi

```bash
python -m streamlit run app/app_compare.py
```

Buka browser di `http://localhost:8501`.

## Cara Penggunaan

Aplikasi memiliki dua mode input:

- **Upload Foto (OCR):** Upload foto kemasan makanan → teks diekstrak otomatis dengan PP-OCRv5
- **Input Teks:** Paste atau ketik langsung teks bahan makanan

Hasil deteksi dari keempat model ditampilkan secara paralel beserta ringkasan alergen yang terdeteksi.
