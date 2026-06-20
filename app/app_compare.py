"""
AlergenScan — 4 Model Comparison
Upload foto kemasan (OCR) atau paste teks → M1 Dictionary | M2 TF-IDF | M3 mBERT | M4 IndoBERT
"""
import os
import re
import zipfile
import pickle
import random
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
from transformers import BertForTokenClassification, BertTokenizerFast

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.normpath(os.path.join(_HERE, '..', 'model'))
M4_PATH    = os.path.join(_MODEL_DIR, 'model_alergen_final_v3')
M3_ZIP     = os.path.join(_MODEL_DIR, 'model_mbert_alergen.zip')
M3_PATH    = os.path.join(_MODEL_DIR, 'model_mbert_alergen')
M2_PKL     = os.path.join(_HERE, '_m2_cache.pkl')

# ── Allergen Dictionary ────────────────────────────────────────────────────────
ALLERGEN_CATEGORIES = {
    'GLUTEN': [
        'tepung terigu', 'tepung gandum', 'wheat starch', 'wheat bran', 'wheat germ',
        'malt extract', 'barley malt', 'hydrolyzed wheat protein',
        'tepung', 'terigu', 'gandum', 'gluten', 'barley', 'rye', 'oats',
        'sereal', 'roti', 'mie', 'pasta', 'biskuit', 'cracker', 'couscous',
        'bulgur', 'semolina', 'spelt', 'durum', 'kamut', 'malt', 'farina',
    ],
    'SUSU': [
        'susu sapi', 'milk powder', 'susu bubuk', 'nonfat milk', 'milk protein',
        'susu kental manis', 'whey protein', 'calcium caseinate', 'sodium caseinate',
        'rennet casein', 'milk solids', 'susu evaporasi', 'milk fat',
        'susu', 'keju', 'mentega', 'krim', 'yoghurt', 'butter', 'cream',
        'casein', 'kasein', 'whey', 'laktosa', 'lactose', 'skimmilk',
        'ghee', 'buttermilk', 'lactalbumin', 'lactoglobulin', 'dairy', 'caseinate',
    ],
    'TELUR': [
        'putih telur', 'kuning telur', 'egg white', 'egg yolk', 'egg powder',
        'telur bubuk', 'egg solids', 'egg protein', 'egg lecithin',
        'telur', 'egg', 'ovalbumin', 'ovomucoid', 'ovomucin',
        'lysozyme', 'mayonnaise', 'mayones', 'albumin', 'meringue',
    ],
    'KACANG': [
        'kacang tanah', 'peanut butter', 'kacang pohon', 'kacang mede', 'kacang mete',
        'kacang almond', 'kacang kenari', 'pine nut', 'brazil nut', 'tree nuts',
        'peanut', 'almond', 'cashew', 'walnut', 'pistachio', 'hazelnut',
        'pecan', 'macadamia', 'chestnut', 'groundnut', 'arachis',
    ],
    'KEDELAI': [
        'soy milk', 'susu kedelai', 'soy sauce', 'soy lecithin', 'soy protein',
        'soy isolate', 'soy flour', 'textured soy', 'lesitin kedelai', 'lesitin nabati',
        'soya lecithin',
        'kedelai', 'soy', 'soybean', 'tahu', 'tempe', 'tofu', 'tempeh',
        'edamame', 'miso', 'natto', 'yuba', 'okara',
        'lesitin', 'lecithin',
    ],
    'SEAFOOD': [
        'ikan teri', 'ikan tongkol', 'ikan kembung', 'ikan lele', 'ikan nila',
        'ikan bandeng', 'fish sauce', 'saus ikan', 'shrimp paste', 'fish oil',
        'krill oil', 'anchovy paste', 'saos tiram', 'saus tiram', 'oyster sauce',
        'ikan', 'tuna', 'salmon', 'sarden', 'makarel', 'cod', 'herring',
        'anchovy', 'udang', 'kerang', 'prawn', 'shrimp', 'cumi', 'sotong',
        'kepiting', 'crab', 'lobster', 'mussel', 'oyster', 'scallop',
        'abalone', 'terasi',
    ],
    'WIJEN': ['sesame oil', 'minyak wijen', 'biji wijen', 'wijen', 'sesame', 'tahini'],
    'SULFITE': [
        'sodium sulfite', 'potassium metabisulfite', 'sulfur dioxide',
        'natrium sulfit', 'kalium metabisulfit', 'sulfite', 'sulfit',
    ],
}
ALL_CATEGORIES = list(ALLERGEN_CATEGORIES.keys())

ALLERGEN_META = {
    'GLUTEN':  {'color': '#C0392B', 'bg': '#FDECEA', 'icon': '🌾', 'label': 'Gluten'},
    'SUSU':    {'color': '#2874A6', 'bg': '#EAF4FB', 'icon': '🥛', 'label': 'Susu'},
    'TELUR':   {'color': '#B7770D', 'bg': '#FEF9E7', 'icon': '🥚', 'label': 'Telur'},
    'KACANG':  {'color': '#1E8449', 'bg': '#EAFAF1', 'icon': '🥜', 'label': 'Kacang'},
    'KEDELAI': {'color': '#76448A', 'bg': '#F5EEF8', 'icon': '🫘', 'label': 'Kedelai'},
    'SEAFOOD': {'color': '#117A65', 'bg': '#E8F8F5', 'icon': '🦐', 'label': 'Seafood'},
    'WIJEN':   {'color': '#A04000', 'bg': '#FDF2E9', 'icon': '🌰', 'label': 'Wijen'},
    'SULFITE': {'color': '#566573', 'bg': '#F2F3F4', 'icon': '⚗️', 'label': 'Sulfit'},
}

MODEL_INFO = {
    'M1': {'name': 'Dictionary NER',  'sub': 'Rule-based · exact match',      'badge': '#495057'},
    'M2': {'name': 'TF-IDF + LR',     'sub': 'Classical ML · doc-level',      'badge': '#0D6EFD'},
    'M3': {'name': 'mBERT NER',        'sub': 'Multilingual DL · F1=0.674',    'badge': '#6F42C1'},
    'M4': {'name': 'IndoBERT NER',     'sub': 'Indonesian DL · F1=0.741 ★',   'badge': '#198754'},
}

# ── OCR helpers ────────────────────────────────────────────────────────────────
_ANCHOR = re.compile(
    r'(komposisi|bahan[-\s]?bahan|bahan|ingredients?|kandungan\s+bahan)', re.IGNORECASE)
_STOP   = re.compile(
    r'(informasi\s+nilai\s+gizi|nutrition\s+facts?|nilai\s+gizi|'
    r'diproduksi\s+oleh|manufactured\s+by|distributor|kadaluarsa|'
    r'best\s+before|tanggal|halal|bpom|no\.\s*reg)', re.IGNORECASE)

@st.cache_resource(show_spinner='Memuat OCR engine (PP-OCRv5, pertama kali ~3 menit)…')
def _load_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(
        text_detection_model_name   ='PP-OCRv5_server_det',
        text_recognition_model_name ='PP-OCRv5_server_rec',
        use_doc_orientation_classify=True,
        use_doc_unwarping           =True,
        device                      ='cpu',
    )

def _run_ocr(img_path: str):
    ocr   = _load_ocr()
    texts = []
    dets  = []  # (bbox, text, score)
    for res in ocr.predict(input=img_path):
        for bbox, text, score in sorted(
            zip(res['rec_polys'], res['rec_texts'], res['rec_scores']),
            key=lambda x: x[0][0][1],
        ):
            dets.append((bbox, text, float(score)))
            if score >= 0.5:
                texts.append(text)
    joined = ' '.join(texts)
    m = _ANCHOR.search(joined)
    if not m:
        return joined, False, dets
    after = joined[m.start():]
    stop  = _STOP.search(after)
    return (after[:stop.start()].strip() if stop else after.strip()), True, dets

def _draw_bboxes(img_path: str, detections: list, conf_thr: float = 0.5) -> Image.Image:
    img  = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for bbox, text, score in detections:
        if score < conf_thr:
            continue
        pts   = [(int(p[0]), int(p[1])) for p in bbox]
        color = '#27AE60' if score >= 0.9 else '#F39C12' if score >= 0.7 else '#E74C3C'
        for i in range(len(pts)):
            draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=color, width=2)
        draw.text((pts[0][0], max(pts[0][1] - 13, 0)), f'{score:.2f}', fill=color)
    return img

def _preprocess(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([%\)\]])([A-Za-z])', r'\1 \2', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── M1: Dictionary NER ─────────────────────────────────────────────────────────
def _build_phrases():
    p = [(w.lower(), cat) for cat, ws in ALLERGEN_CATEGORIES.items() for w in ws]
    p.sort(key=lambda x: len(x[0].split()), reverse=True)
    return p

_PHRASE_LIST = _build_phrases()

def _tok(text: str):
    return re.findall(r'\w+|[^\w\s]', text.lower())

def m1_predict(text: str) -> dict:
    tokens = _tok(text)
    labels = ['O'] * len(tokens)
    i = 0
    while i < len(tokens):
        matched = False
        for phrase, cat in _PHRASE_LIST:
            pt   = _tok(phrase)
            plen = len(pt)
            if tokens[i:i+plen] == pt and all(labels[j] == 'O' for j in range(i, i+plen)):
                labels[i] = f'B-{cat}'
                for j in range(i+1, i+plen): labels[j] = f'I-{cat}'
                i += plen; matched = True; break
        if not matched: i += 1
    found = {}
    for tok, lbl in zip(tokens, labels):
        if lbl != 'O':
            cat = lbl[2:]
            found.setdefault(cat, []).append(tok)
    return found

# ── M2: TF-IDF + Logistic Regression ──────────────────────────────────────────
def _train_m2_from_synthetic():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    random.seed(42)
    NON_ALR   = ['garam','air','gula','minyak sawit','pati','vitamin c','asam sitrat',
                 'pewarna makanan','antioksidan','natrium klorida','glukosa','karagenan','perisa']
    TEMPLATES = ['Komposisi: {b}.', 'Bahan: {b}.', 'Ingredients: {b}.']
    all_words = [(w, cat) for cat, ws in ALLERGEN_CATEGORIES.items() for w in ws]

    texts, labels = [], []
    for _ in range(1200):
        sel_alr = random.sample(all_words, random.randint(1, 3))
        sel_non = random.sample(NON_ALR,   random.randint(2, 5))
        ingr    = [w for w, _ in sel_alr] + sel_non
        random.shuffle(ingr)
        texts.append(random.choice(TEMPLATES).format(b=', '.join(ingr)))
        labels.append(sorted({cat for _, cat in sel_alr}))

    mlb   = MultiLabelBinarizer(classes=ALL_CATEGORIES)
    Y     = mlb.fit_transform(labels)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
                            lowercase=True, min_df=1)
    X     = tfidf.fit_transform(texts)
    clf   = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'), n_jobs=-1)
    clf.fit(X, Y)

    with open(M2_PKL, 'wb') as f:
        pickle.dump({'tfidf': tfidf, 'clf': clf, 'mlb': mlb}, f)
    return tfidf, clf, mlb

@st.cache_resource(show_spinner='Memuat Model 2 (TF-IDF + LR)…')
def load_m2():
    try:
        if os.path.exists(M2_PKL):
            with open(M2_PKL, 'rb') as f:
                d = pickle.load(f)
            return d['tfidf'], d['clf'], d['mlb']
        return _train_m2_from_synthetic()
    except Exception:
        return None, None, None

def m2_predict(text: str, tfidf, clf, mlb) -> dict | None:
    if tfidf is None: return None
    vec  = tfidf.transform([text])
    pred = clf.predict(vec)[0]
    cats = [ALL_CATEGORIES[i] for i, v in enumerate(pred) if v == 1]
    return {cat: [] for cat in sorted(cats)}  # no token detail for doc-level

# ── M3 / M4: BERT NER ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Memuat Model 3 (mBERT NER)…')
def load_m3():
    try:
        path = M3_PATH
        if not os.path.isfile(os.path.join(path, 'config.json')):
            if not os.path.exists(M3_ZIP):
                return None, None, None
            # Always extract into the named subfolder, not _MODEL_DIR root
            os.makedirs(path, exist_ok=True)
            with zipfile.ZipFile(M3_ZIP, 'r') as z:
                z.extractall(path)
            # If zip was nested (one subfolder inside), flatten it
            entries = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
            if len(entries) == 1 and not os.path.isfile(os.path.join(path, 'config.json')):
                sub = os.path.join(path, entries[0])
                for f in os.listdir(sub):
                    os.rename(os.path.join(sub, f), os.path.join(path, f))
                os.rmdir(sub)
        if not os.path.isfile(os.path.join(path, 'config.json')):
            return None, None, None
        tok = BertTokenizerFast.from_pretrained(path)
        mdl = BertForTokenClassification.from_pretrained(path, ignore_mismatched_sizes=True)
        mdl.eval()
        return mdl, tok, mdl.config.id2label
    except Exception:
        return None, None, None

@st.cache_resource(show_spinner='Memuat Model 4 (IndoBERT NER)…')
def load_m4():
    try:
        if not os.path.isdir(M4_PATH):
            return None, None, None
        tok = BertTokenizerFast.from_pretrained(M4_PATH)
        mdl = BertForTokenClassification.from_pretrained(M4_PATH, ignore_mismatched_sizes=True)
        mdl.eval()
        return mdl, tok, mdl.config.id2label
    except Exception:
        return None, None, None

def bert_predict(text: str, model, tokenizer, id2label, conf: float = 0.80) -> dict | None:
    if model is None:
        return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    inputs = tokenizer(text.lower(), return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs    = torch.softmax(logits, dim=2)[0]
    pred_ids = torch.argmax(probs, dim=1).tolist()
    max_prob = probs.max(dim=1).values.tolist()
    tokens   = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    found     = {}
    cur_word  = ''
    cur_label = 'O'
    for tok, pid, prob in zip(tokens, pred_ids, max_prob):
        if tok in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        label = id2label[pid]
        if label != 'O' and prob < conf:
            label = 'O'
        if tok.startswith('##'):
            cur_word += tok[2:]
        else:
            if cur_word and cur_label != 'O':
                found.setdefault(cur_label[2:], []).append(cur_word)
            cur_word, cur_label = tok, label
    if cur_word and cur_label != 'O':
        found.setdefault(cur_label[2:], []).append(cur_word)
    return found

# ── UI ─────────────────────────────────────────────────────────────────────────
def _card_html(allergens, model_id: str) -> str:
    info = MODEL_INFO[model_id]

    if allergens is None:
        body = '<p style="color:#DC3545;font-size:0.8rem;margin:0">⚠️ Model tidak tersedia</p>'
    elif not allergens:
        body = ('<div style="background:#D1FAE5;border-radius:8px;padding:0.55rem 0.75rem;'
                'text-align:center;color:#065F46;font-size:0.85rem;font-weight:600">'
                '✓ Aman</div>')
    else:
        rows = ''
        for cat in ALL_CATEGORIES:
            if cat not in allergens: continue
            meta  = ALLERGEN_META[cat]
            toks  = allergens[cat]
            label = f'{meta["icon"]} {meta["label"]}'
            tok_s = ', '.join(dict.fromkeys(toks)) if toks else ''
            rows += (
                f'<div style="margin-bottom:0.4rem">'
                f'<span style="background:{meta["bg"]};color:{meta["color"]};'
                f'border:1px solid {meta["color"]}55;font-size:0.73rem;font-weight:700;'
                f'padding:0.18rem 0.5rem;border-radius:6px">{label}</span>'
                + (f'<div style="font-size:0.68rem;color:#718096;margin-top:0.1rem;'
                   f'padding-left:0.15rem">{tok_s}</div>' if tok_s else '')
                + '</div>'
            )
        body = rows

    return (
        f'<div style="border:1px solid #E2E8F0;border-radius:10px;padding:0.9rem 1rem;'
        f'background:#fff;min-height:180px">'
        f'<div style="display:flex;align-items:flex-start;gap:0.4rem;margin-bottom:0.7rem;'
        f'padding-bottom:0.6rem;border-bottom:1px solid #F1F5F9">'
        f'<span style="background:{info["badge"]};color:#fff;font-size:0.67rem;font-weight:700;'
        f'padding:0.18rem 0.48rem;border-radius:10px;white-space:nowrap;margin-top:0.1rem">'
        f'{model_id}</span>'
        f'<div><div style="font-weight:700;font-size:0.86rem;color:#1A202C;line-height:1.2">'
        f'{info["name"]}</div>'
        f'<div style="font-size:0.67rem;color:#718096;margin-top:0.08rem">{info["sub"]}</div>'
        f'</div></div>{body}</div>'
    )

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AlergenScan — Model Comparison',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
#MainMenu,footer,header{visibility:hidden}
.main .block-container{padding-top:1.5rem;max-width:1300px}
.stTextArea textarea{font-size:0.85rem}
.stTabs [data-baseweb="tab"]{font-size:0.85rem}
</style>""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin-bottom:0.1rem">AlergenScan — Model Comparison</h2>'
    '<p style="color:#6C757D;font-size:0.88rem;margin-top:0">Upload foto kemasan atau input teks → '
    'M1 Dictionary | M2 TF-IDF | M3 mBERT | M4 IndoBERT</p>',
    unsafe_allow_html=True,
)

# ── Input section ──────────────────────────────────────────────────────────────
ingredient_text = ''

ocr_available = True
try:
    import paddleocr  # noqa: F401
except ImportError:
    ocr_available = False

tab_foto, tab_teks = st.tabs(['📷 Upload Foto (OCR)', '📝 Input Teks'])

with tab_foto:
    if not ocr_available:
        st.warning(
            'PaddleOCR belum terinstall — tab ini tidak tersedia. '
            'Gunakan tab **Input Teks** untuk tetap menggunakan app.\n\n'
            'Untuk mengaktifkan OCR, install dengan:\n'
            '```\npip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n'
            'pip install paddleocr\n```'
        )
    else:
        uploaded = st.file_uploader(
            'Upload foto kemasan makanan:',
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility='collapsed',
        )

        if uploaded:
            file_key = uploaded.name + str(uploaded.size)

            # Only run OCR if this file hasn't been processed yet
            if st.session_state.get('ocr_file_key') != file_key:
                suffix = os.path.splitext(uploaded.name)[1]
                tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read()); tmp.close()

                with st.spinner('Menjalankan OCR…'):
                    try:
                        ocr_text, anchor_found, detections = _run_ocr(tmp.name)
                        ingredient_text = _preprocess(ocr_text)
                        st.session_state['ocr_file_key']     = file_key
                        st.session_state['ocr_text']         = ingredient_text
                        st.session_state['ocr_anchor']       = anchor_found
                        st.session_state['ocr_detections']   = detections
                        st.session_state['ocr_img_bytes']    = open(tmp.name, 'rb').read()
                        st.session_state['ocr_img_suffix']   = suffix
                    except Exception as e:
                        st.error(
                            f'OCR gagal dijalankan: `{e}`\n\n'
                            'Kemungkinan PaddlePaddle belum terinstall dengan benar. Coba:\n'
                            '```\npip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n```\n\n'
                            'Sementara itu, gunakan tab **Input Teks** untuk tetap menggunakan app.'
                        )
                        os.unlink(tmp.name)
                        st.stop()
                os.unlink(tmp.name)

            # Display cached OCR result
            ingredient_text = st.session_state.get('ocr_text', '')
            anchor_found    = st.session_state.get('ocr_anchor', False)
            detections      = st.session_state.get('ocr_detections', [])
            img_bytes       = st.session_state.get('ocr_img_bytes', b'')

            n_det = sum(1 for _, _, s in detections if s >= 0.5)
            col_orig, col_bbox, col_txt = st.columns([1, 1, 1])

            with col_orig:
                st.caption('Foto asli')
                st.image(img_bytes, use_container_width=True)

            with col_bbox:
                st.caption(
                    f'Bounding box · {n_det} deteksi  '
                    '<span style="color:#27AE60">■</span> ≥0.9  '
                    '<span style="color:#F39C12">■</span> ≥0.7  '
                    '<span style="color:#E74C3C">■</span> ≥0.5',
                    unsafe_allow_html=True,
                )
                tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=st.session_state.get('ocr_img_suffix', '.jpg'))
                tmp2.write(img_bytes); tmp2.close()
                st.image(_draw_bboxes(tmp2.name, detections), use_container_width=True)
                os.unlink(tmp2.name)

            with col_txt:
                if not anchor_found:
                    st.warning("Anchor 'Komposisi/Bahan' tidak ditemukan.")
                st.caption('Teks komposisi (hasil OCR)')
                st.text_area('ocr', value=ingredient_text, height=200, disabled=True,
                             label_visibility='collapsed')

with tab_teks:
    manual_text = st.text_area(
        'Paste atau ketik teks bahan makanan:',
        height=150,
        placeholder='Contoh: Komposisi: tepung terigu, gula, susu bubuk, kacang tanah, perisa alami.',
        label_visibility='collapsed',
    )
    if manual_text.strip():
        ingredient_text = manual_text.strip()

# ── Guard ──────────────────────────────────────────────────────────────────────
if not ingredient_text.strip():
    st.info('Upload foto kemasan atau ketik teks bahan untuk mulai.')
    st.stop()

# ── Load models ────────────────────────────────────────────────────────────────
m2_tfidf, m2_clf, m2_mlb     = load_m2()
m3_model, m3_tok, m3_id2label = load_m3()
m4_model, m4_tok, m4_id2label = load_m4()

# ── Run 4 models ───────────────────────────────────────────────────────────────
with st.spinner('Menjalankan 4 model…'):
    r1 = m1_predict(ingredient_text)
    r2 = m2_predict(ingredient_text, m2_tfidf, m2_clf, m2_mlb)
    r3 = bert_predict(ingredient_text, m3_model, m3_tok, m3_id2label)
    r4 = bert_predict(ingredient_text, m4_model, m4_tok, m4_id2label)

# ── 4-column results ───────────────────────────────────────────────────────────
st.divider()
st.markdown('**Hasil Deteksi — 4 Model**')

c1, c2, c3, c4 = st.columns(4)
c1.markdown(_card_html(r1, 'M1'), unsafe_allow_html=True)
c2.markdown(_card_html(r2, 'M2'), unsafe_allow_html=True)
c3.markdown(_card_html(r3, 'M3'), unsafe_allow_html=True)
c4.markdown(_card_html(r4, 'M4'), unsafe_allow_html=True)

# ── Aggregate summary ──────────────────────────────────────────────────────────
st.divider()
all_detected: dict[str, list[str]] = {}
for mid, r in [('M1', r1), ('M2', r2), ('M3', r3), ('M4', r4)]:
    if r:
        for cat in r:
            all_detected.setdefault(cat, []).append(mid)

if not all_detected:
    st.success('✅ Tidak ada alergen terdeteksi oleh semua model.')
else:
    tags = ''
    for cat in ALL_CATEGORIES:
        if cat not in all_detected: continue
        meta     = ALLERGEN_META[cat]
        by_whom  = ', '.join(all_detected[cat])
        n        = len(all_detected[cat])
        opacity  = '1.0' if n >= 3 else ('0.85' if n == 2 else '0.65')
        tags += (
            f'<span style="opacity:{opacity};background:{meta["bg"]};color:{meta["color"]};'
            f'border:1px solid {meta["color"]}55;padding:0.25rem 0.7rem;border-radius:20px;'
            f'font-size:0.8rem;font-weight:600;margin-right:0.4rem;margin-bottom:0.4rem;'
            f'display:inline-block">'
            f'{meta["icon"]} {meta["label"]}'
            f'<span style="font-weight:400;opacity:0.75;font-size:0.72rem"> ({by_whom})</span>'
            f'</span>'
        )
    st.markdown(
        '<p style="color:#6C757D;font-size:0.8rem;margin-bottom:0.4rem">'
        'Alergen terdeteksi · opasitas tinggi = lebih banyak model setuju</p>'
        f'<div style="display:flex;flex-wrap:wrap">{tags}</div>',
        unsafe_allow_html=True,
    )

# ── Debug expanders ────────────────────────────────────────────────────────────
with st.expander('Teks yang dianalisis'):
    st.code(ingredient_text, language=None)

with st.expander('📋 Info Model yang Dimuat'):
    import json as _json
    for label, path in [('M1 Dictionary', None), ('M2 TF-IDF + LR', None),
                        ('M3 mBERT', M3_PATH), ('M4 IndoBERT', M4_PATH)]:
        st.markdown(f'**{label}**')
        if path is None:
            # Rule-based / sklearn — show dict stats
            if label.startswith('M1'):
                total_kw = sum(len(v) for v in ALLERGEN_CATEGORIES.values())
                st.json({'type': 'rule-based', 'keywords': total_kw, 'categories': ALL_CATEGORIES})
            else:
                st.json({'type': 'TF-IDF + OneVsRest LogisticRegression',
                         'trained_from': 'synthetic data (1200 samples)',
                         'pkl_cached': os.path.isfile(M2_PKL)})
        else:
            cfg_file = os.path.join(path, 'config.json')
            if os.path.isfile(cfg_file):
                with open(cfg_file, encoding='utf-8') as f:
                    cfg = _json.load(f)
                st.json({
                    'base_model':    cfg.get('_name_or_path', '—'),
                    'model_type':    cfg.get('model_type', '—'),
                    'architectures': cfg.get('architectures', '—'),
                    'num_labels':    cfg.get('num_labels', '—'),
                    'id2label':      cfg.get('id2label', '—'),
                })
            else:
                st.warning(f'config.json tidak ditemukan di `{path}`')
