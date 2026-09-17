import string
from transformers import pipeline
from sklearn.metrics import accuracy_score

print("=" * 65)
print("LAB 3 : PIPELINES HUGGING FACE POUR LA CLASSIFICATION DE TEXTES")
print("=" * 65)

# ============================================================
# ÉTAPE 2 : Classification d'opinion (Sentiment Analysis)
# ============================================================
print("\n--- ÉTAPE 2 : ANALYSE DE SENTIMENT (DistilBERT SST-2) ---")

sentiment_pipeline = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Test unitaire
res_single = sentiment_pipeline("I really liked the movie!!")
print(f"Test unitaire : {res_single}")

# Traitement en lot
texts = [
    "I really liked the movie!!",
    "Great job ruining my day.",
    "This product exceeded my expectations.",
    "Wow, just what I needed... another problem.",
    "Absolutely fantastic experience!"
]
true_sentiment = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]

batch_results = sentiment_pipeline(texts)
pred_sentiment = [res["label"] for res in batch_results]

for text, pred, conf in [(t, r["label"], r["score"]) for t, r in zip(texts, batch_results)]:
    print(f"  [{pred} - {conf:.3f}] {text}")

print(f"Accuracy Étape 2 : {accuracy_score(true_sentiment, pred_sentiment) * 100:.1f}%\n")

# ============================================================
# ÉTAPE 3 : Classification Zero-Shot
# ============================================================
print("--- ÉTAPE 3 : CLASSIFICATION ZERO-SHOT (DeBERTa-v3) ---")

zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

text_zs = "The national football team won the cup yesterday."
candidate_labels = ["sports", "technology", "health"]
res_zs = zero_shot_classifier(text_zs, candidate_labels)

print(f"Texte : '{text_zs}'")
for label, score in zip(res_zs["labels"], res_zs["scores"]):
    print(f"  Label: {label:<12} | Score: {score:.4f}")

# Test de variation lexicale
res_zs_var = zero_shot_classifier(text_zs, ["sport news", "tech news", "medical update"])
print("Variation des libellés (top 1) :", res_zs_var["labels"][0], f"({res_zs_var['scores'][0]:.4f})\n")

# ============================================================
# ÉTAPE 4 : Inférence Question–Réponse (QNLI)
# ============================================================
print("--- ÉTAPE 4 : QUESTION-ANSWERING NLI (QNLI Electra) ---")

qnli_pipeline = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base"
)

qnli_dataset = [
    ("Where do penguins live?", "Penguins are found primarily in the Southern Hemisphere.", "LABEL_0"),
    ("What is the capital of France?", "Penguins are found primarily in the Southern Hemisphere.", "LABEL_1"),
    ("Who invented the telephone?", "Alexander Graham Bell was awarded the first US patent for the telephone.", "LABEL_0"),
    ("What causes rain?", "Water evaporates from the surface, condenses into clouds, and falls as precipitation.", "LABEL_0"),
    ("How tall is Mount Everest?", "The Nile River is considered the longest river in Africa.", "LABEL_1")
]

qnli_preds, qnli_trues = [], []
for q, p, label_exp in qnli_dataset:
    out = qnli_pipeline({"text": q, "text_pair": p})
    qnli_preds.append(out["label"])
    qnli_trues.append(label_exp)
    desc = "Réponse présente" if out["label"] == "LABEL_0" else "Pas de réponse"
    print(f"  Q: '{q}' | Passage: '{p[:30]}...' -> {out['label']} ({desc})")

print(f"Accuracy QNLI : {accuracy_score(qnli_trues, qnli_preds) * 100:.1f}%\n")

# ============================================================
# ÉTAPE 5 : Détection de Paraphrases (QQP)
# ============================================================
print("--- ÉTAPE 5 : DÉTECTION DE PARAPHRASES (BERT-QQP) ---")

qqp_pipeline = pipeline(
    task="text-classification",
    model="textattack/bert-base-uncased-QQP"
)

qqp_pairs = [
    ("How can I learn Python?", "What is the best way to study Python?", "LABEL_1"),
    ("How can I learn Python?", "What is the capital of France?", "LABEL_0"),
    ("What is machine learning?", "Can you define machine learning?", "LABEL_1"),
    ("How do I cook pasta?", "Where can I buy an electric car?", "LABEL_0"),
    ("Why is the sky blue?", "What makes the atmosphere appear blue?", "LABEL_1"),
    ("Is it going to rain today?", "What time does the bank close?", "LABEL_0")
]

qqp_preds, qqp_trues = [], []
for q1, q2, target in qqp_pairs:
    res = qqp_pipeline({"text": q1, "text_pair": q2})
    qqp_preds.append(res["label"])
    qqp_trues.append(target)
    status = "Paraphrase" if res["label"] == "LABEL_1" else "Non-paraphrase"
    print(f"  [{status}] '{q1}' <=> '{q2}'")

print(f"Accuracy QQP : {accuracy_score(qqp_trues, qqp_preds) * 100:.1f}%\n")

# ============================================================
# ÉTAPE 6 : Vérification Grammaticale (CoLA)
# ============================================================
print("--- ÉTAPE 6 : ACCEPTABILITÉ GRAMMATICALE (DistilBERT CoLA) ---")

cola_classifier = pipeline(
    task="text-classification",
    model="textattack/distilbert-base-uncased-CoLA"
)

sentences_cola = [
    ("The cat sat on the mat.", "LABEL_1"),
    ("The cat on sat mat the.", "LABEL_0"),
    ("She reads books every day.", "LABEL_1"),
    ("She readed books yesterday.", "LABEL_0"),
    ("Colorless green ideas sleep furiously.", "LABEL_1")  # Absurde mais syntaxiquement valide
]

cola_preds, cola_trues = [], []
for s, expected in sentences_cola:
    res = cola_classifier(s)[0]  # Récupération du premier élément de la liste
    cola_preds.append(res["label"])
    cola_trues.append(expected)
    verdict = "Grammatical" if res["label"] == "LABEL_1" else "Agrammatical"
    print(f"  [{verdict:<12}] '{s}'")

print(f"Accuracy CoLA : {accuracy_score(cola_trues, cola_preds) * 100:.1f}%")