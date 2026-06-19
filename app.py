```python
import os
import re
import json
import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "best_rnn_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"
METRICS_PATH = "metrics.json"
LOGO_PATH = "logo_shopee.png"

st.set_page_config(
    page_title="Shopee Sentiment",
    page_icon=LOGO_PATH,
    layout="wide"
)

def check_file(path, name):
    if not os.path.exists(path):
        st.error(f"File {name} tidak ditemukan: {path}")
        st.stop()

check_file(MODEL_PATH, "model")
check_file(TOKENIZER_PATH, "tokenizer")
check_file(METRICS_PATH, "metrics")
check_file(LOGO_PATH, "logo")

@st.cache_resource
def load_rnn_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

model = load_rnn_model()
tokenizer = load_tokenizer()
metrics = load_metrics()

MAX_LEN = metrics.get("max_len", 80)

SLANG_MAP = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "udh": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "dr": "dari",
    "dgn": "dengan",
    "krn": "karena",
    "tp": "tapi",
    "jg": "juga",
    "aja": "saja",
    "bkn": "bukan"
}

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [SLANG_MAP.get(word, word) for word in words]

    return " ".join(words)

def predict_sentiment(text):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prob = float(model.predict(padded, verbose=0)[0][0])

    if prob >= 0.5:
        label = "Positif"
        confidence = prob
    else:
        label = "Negatif"
        confidence = 1 - prob

    return cleaned, label, confidence


# =========================
# SIDEBAR
# =========================

st.sidebar.image(LOGO_PATH, width=80)
st.sidebar.title("Shopee")
st.sidebar.subheader("Informasi Model")

st.sidebar.write("Metode:", "Bidirectional SimpleRNN")
st.sidebar.write("Accuracy:", f"{metrics['accuracy']*100:.2f}%")
st.sidebar.write("Precision:", f"{metrics['precision']*100:.2f}%")
st.sidebar.write("Recall:", f"{metrics['recall']*100:.2f}%")
st.sidebar.write("F1 Score:", f"{metrics['f1_score']*100:.2f}%")
st.sidebar.write("MAX_LEN:", metrics["max_len"])
st.sidebar.write("Dataset:", metrics["data_path"])

st.sidebar.caption("Project Skripsi Suci © 2025")

# =========================
# HEADER
# =========================

col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image(LOGO_PATH, width=90)

with col_title:
    st.title("Shopee Sentiment")
    st.write(
        "Analisis ulasan aplikasi Shopee menggunakan metode "
        "Recurrent Neural Network (RNN)."
    )

st.divider()

# =========================
# PREDIKSI SATU ULASAN
# =========================

col1, col2 = st.columns([1.5, 1])

with col1:

    st.subheader("Input Ulasan")

    text = st.text_area(
        "Masukkan teks ulasan",
        height=200,
        placeholder="contoh: aplikasi bagus dan pengiriman cepat"
    )

    analyze = st.button("MULAI ANALISIS")

with col2:

    st.subheader("Hasil Prediksi")

    if analyze:

        if text.strip() == "":
            st.warning("Masukkan ulasan terlebih dahulu")

        else:

            cleaned, label, confidence = predict_sentiment(text)

            if label == "Positif":
                st.success(f"Sentimen: {label}")
            else:
                st.error(f"Sentimen: {label}")

            st.write(
                f"Tingkat keyakinan: {confidence*100:.2f}%"
            )

            st.progress(float(confidence))

            st.write("Teks setelah preprocessing:")
            st.info(cleaned)

# =========================
# ANALISIS CSV
# =========================

st.divider()

st.subheader("Analisis Banyak Ulasan (CSV)")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.write("Preview Data")

        st.dataframe(df.head())

        if "ulasan" not in df.columns:

            st.error(
                "CSV harus memiliki kolom bernama 'ulasan'"
            )

        else:

            if st.button("ANALISIS CSV"):

                hasil = []

                progress = st.progress(0)

                total = len(df)

                for i, review in enumerate(df["ulasan"]):

                    _, label, confidence = predict_sentiment(
                        str(review)
                    )

                    hasil.append({
                        "ulasan": review,
                        "sentimen": label,
                        "confidence (%)": round(
                            confidence * 100, 2
                        )
                    })

                    progress.progress(
                        (i + 1) / total
                    )

                result_df = pd.DataFrame(hasil)

                st.success(
                    f"Analisis selesai ({total} ulasan)"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Hasil CSV",
                    data=csv,
                    file_name="hasil_sentimen.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"Gagal membaca file: {e}")

# =========================
# EVALUASI MODEL
# =========================

st.divider()

st.subheader("Detail Evaluasi Model")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{metrics['accuracy']*100:.2f}%"
)

m2.metric(
    "Precision",
    f"{metrics['precision']*100:.2f}%"
)

m3.metric(
    "Recall",
    f"{metrics['recall']*100:.2f}%"
)

m4.metric(
    "F1 Score",
    f"{metrics['f1_score']*100:.2f}%"
)

st.subheader("Confusion Matrix")

c1, c2, c3, c4 = st.columns(4)

c1.metric("TP", metrics["TP"])
c2.metric("TN", metrics["TN"])
c3.metric("FP", metrics["FP"])
c4.metric("FN", metrics["FN"])
```
