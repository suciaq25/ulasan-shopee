import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# =========================
# 1. CONFIG
# =========================
DATA_PATH = "review_shopee.csv"

TEXT_COL = "comment"
RATING_COL = "rating"

VOCAB_SIZE = 15000
MAX_LEN = 80
EMBED_DIM = 128

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
SEED = 42

MODEL_PATH = "best_rnn_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"
METRICS_PATH = "metrics.json"


# =========================
# 2. SET SEED
# =========================
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# 3. CEK FILE DATASET
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}\n"
        f"Pastikan file CSV ada di folder project."
    )


# =========================
# 4. PREPROCESSING
# =========================
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


# =========================
# 5. LOAD DATASET
# =========================
df = pd.read_csv(DATA_PATH)

print("===== INFO DATASET =====")
print("Kolom dataset:", df.columns.tolist())
print(df.head())


# =========================
# 6. VALIDASI KOLOM
# =========================
if TEXT_COL not in df.columns:
    raise ValueError(f"Kolom teks '{TEXT_COL}' tidak ditemukan di dataset.")

if RATING_COL not in df.columns:
    raise ValueError(f"Kolom rating '{RATING_COL}' tidak ditemukan di dataset.")


# =========================
# 7. PILIH KOLOM YANG DIPAKAI
# =========================
df = df[[TEXT_COL, RATING_COL]].copy()
df = df.dropna(subset=[TEXT_COL, RATING_COL]).reset_index(drop=True)

df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
df = df.dropna(subset=[RATING_COL]).reset_index(drop=True)

df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""].reset_index(drop=True)


# =========================
# 8. BUAT LABEL DARI RATING
# =========================
def rating_to_label(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return np.nan

df["label"] = df[RATING_COL].apply(rating_to_label)
df = df.dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
df = df[df[TEXT_COL].str.strip() != ""].reset_index(drop=True)

print("\n===== DISTRIBUSI LABEL =====")
print(df["label"].value_counts())
print("Jumlah data setelah cleaning:", len(df))


# =========================
# 9. SPLIT DATA
# =========================
X = df[TEXT_COL]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train
)


# =========================
# 10. TOKENIZER + PADDING
# =========================
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")


# =========================
# 11. CLASS WEIGHT
# =========================
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))
print("\nClass weight:", class_weight_dict)


# =========================
# 12. BUILD MODEL RNN
# =========================
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
    Bidirectional(SimpleRNN(
        64,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )),
    Dropout(0.3),
    Bidirectional(SimpleRNN(
        32,
        dropout=0.2,
        recurrent_dropout=0.2
    )),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n===== MODEL SUMMARY =====")
model.summary()


# =========================
# 13. CALLBACKS
# =========================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]


# =========================
# 14. TRAINING
# =========================
print("\n===== MULAI TRAINING =====")
history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# =========================
# 15. LOAD MODEL TERBAIK
# =========================
best_model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# 16. PREDIKSI TEST
# =========================
y_pred_prob = best_model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()


# =========================
# 17. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()


# =========================
# 18. HITUNG METRICS
# =========================
total = TP + TN + FP + FN

accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# =========================
# 19. TAMPILKAN HASIL
# =========================
print("\n===== CONFUSION MATRIX =====")
print(f"TP : {TP}")
print(f"TN : {TN}")
print(f"FP : {FP}")
print(f"FN : {FN}")

print("\n===== EVALUATION METRICS =====")
print(f"Total Data : {total}")
print(f"Accuracy   : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision  : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall     : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score   : {f1_score:.4f} ({f1_score*100:.2f}%)")


# =========================
# 20. SIMPAN TOKENIZER
# =========================
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)


# =========================
# 21. SIMPAN METRICS
# =========================
metrics = {
    "TP": int(TP),
    "TN": int(TN),
    "FP": int(FP),
    "FN": int(FN),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1_score),
    "total_data": int(total),
    "data_path": DATA_PATH,
    "text_col": TEXT_COL,
    "rating_col": RATING_COL,
    "label_source": "rating >= 4 => positif, rating <= 2 => negatif, rating == 3 dibuang",
    "vocab_size": VOCAB_SIZE,
    "max_len": MAX_LEN,
    "embed_dim": EMBED_DIM,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nModel disimpan ke     : {MODEL_PATH}")
print(f"Tokenizer disimpan ke : {TOKENIZER_PATH}")
print(f"Metrics disimpan ke   : {METRICS_PATH}")