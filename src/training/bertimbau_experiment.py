# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.paths import (
    GOLD_DASHBOARD_MODEL_COMPARISON_PATH,
    GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH,
    MODELS_DIR,
    SILVER_TEXTUAL_DATASET_PATH,
)

# %%
df = pd.read_csv(SILVER_TEXTUAL_DATASET_PATH, encoding="utf-8")

print(df.shape)
print(df.columns.tolist())
print(df.head())

# %%
df_train = df[df["sentimento_real"].notna()].copy()
df_pred = df[df["sentimento_real"].isna()].copy()

print("Treino:", df_train.shape)
print("Predição:", df_pred.shape)
print(df_train["sentimento_real"].value_counts())
print(df_pred["fonte"].value_counts())

# %%
label2id = {
    "Negativo": 0,
    "Neutro": 1,
    "Positivo": 2,
}

id2label = {v: k for k, v in label2id.items()}

df_train["label"] = df_train["sentimento_real"].map(label2id)

print(df_train[["sentimento_real", "label"]].head(10))
print(df_train["label"].value_counts())

# %%
train_df, test_df = train_test_split(
    df_train[["texto_original", "label", "sentimento_real"]].copy(),
    test_size=0.2,
    random_state=42,
    stratify=df_train["label"],
)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

print("\nDistribuição no train:")
print(train_df["sentimento_real"].value_counts())

print("\nDistribuição no test:")
print(test_df["sentimento_real"].value_counts())

# %%
model_name = "neuralmind/bert-base-portuguese-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

print("Tokenizer carregado")
print("Modelo carregado")

# %%
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
pred_dataset = Dataset.from_pandas(df_pred[["texto_original"]].reset_index(drop=True))

print(train_dataset)
print(test_dataset)
print(pred_dataset)

# %%
def tokenize_function(examples):
    return tokenizer(
        examples["texto_original"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


# %%
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
pred_dataset = pred_dataset.map(tokenize_function, batched=True)

# %%
print(train_dataset)
print(train_dataset[0])

# %%
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# %%
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
)

pred_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids"],
)

# %%
print(train_dataset)
print(train_dataset[0])

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


# %%
training_args = TrainingArguments(
    output_dir=str(MODELS_DIR / "bertimbau"),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none",
)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# %%
train_result = trainer.train()
train_result

# %%
pred_output = trainer.predict(test_dataset)

# %%
print(type(pred_output))
print(pred_output.predictions.shape)
print(pred_output.label_ids.shape)
print(pred_output.metrics)

# %%
y_pred = np.argmax(pred_output.predictions, axis=-1)
y_true = pred_output.label_ids

print(y_pred[:20])
print(y_true[:20])

# %%
print(classification_report(y_true, y_pred, target_names=["Negativo", "Neutro", "Positivo"]))
print(confusion_matrix(y_true, y_pred))

# %%
pred_youtube_output = trainer.predict(pred_dataset)
youtube_pred_labels = np.argmax(pred_youtube_output.predictions, axis=-1)

df_pred_bert = df_pred.copy()
df_pred_bert["sentimento_previsto_bert"] = [id2label[label] for label in youtube_pred_labels]

print(df_pred_bert["sentimento_previsto_bert"].value_counts())
print(df_pred_bert[["texto_original", "sentimento_previsto_bert"]].head(20))

# %%
df_pred_bert.to_csv(GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH, index=False, encoding="utf-8-sig")
print(GOLD_DASHBOARD_YOUTUBE_BERT_DATASET_PATH)

# %%
classes = [0, 1, 2]
class_names = ["Negativo", "Neutro", "Positivo"]

y_true_bin = label_binarize(y_true, classes=classes)
y_score_bert = pred_output.predictions

print("ROC AUC macro:", roc_auc_score(y_true_bin, y_score_bert, multi_class="ovr", average="macro"))
print("ROC AUC weighted:", roc_auc_score(y_true_bin, y_score_bert, multi_class="ovr", average="weighted"))

# %%
fpr = {}
tpr = {}
roc_auc = {}

for i, class_name in enumerate(class_names):
    fpr[class_name], tpr[class_name], _ = roc_curve(y_true_bin[:, i], y_score_bert[:, i])
    roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

plt.figure(figsize=(8, 6))

for class_name in class_names:
    plt.plot(
        fpr[class_name],
        tpr[class_name],
        label=f"{class_name} (AUC = {roc_auc[class_name]:.3f})",
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC One-vs-Rest - BERTimbau")
plt.legend()
plt.grid(True)
plt.show()

# %%
comparison_rows = [
    {
        "modelo": "LogisticRegression",
        "accuracy": 0.85,
        "f1_macro": 0.63,
        "roc_auc_macro": 0.8791,
        "roc_auc_weighted": 0.9392,
        "f1_negativo": 0.75,
        "f1_neutro": 0.20,
        "f1_positivo": 0.93,
    },
    {
        "modelo": "MultinomialNB",
        "accuracy": 0.88,
        "f1_macro": 0.57,
        "roc_auc_macro": 0.8888,
        "roc_auc_weighted": 0.9393,
        "f1_negativo": 0.77,
        "f1_neutro": 0.00,
        "f1_positivo": 0.94,
    },
    {
        "modelo": "LinearSVC",
        "accuracy": 0.87,
        "f1_macro": 0.62,
        "roc_auc_macro": 0.8607,
        "roc_auc_weighted": 0.9379,
        "f1_negativo": 0.76,
        "f1_neutro": 0.17,
        "f1_positivo": 0.94,
    },
    {
        "modelo": "BERTimbau",
        "accuracy": 0.90,
        "f1_macro": 0.61,
        "roc_auc_macro": 0.9146,
        "roc_auc_weighted": 0.9548,
        "f1_negativo": 0.81,
        "f1_neutro": 0.08,
        "f1_positivo": 0.95,
    },
]

df_compare_final = pd.DataFrame(comparison_rows)

df_compare_final = df_compare_final.sort_values(
    by=["roc_auc_macro", "f1_macro", "accuracy"],
    ascending=False,
).reset_index(drop=True)

df_compare_final

# %%
df_compare_final.round(4)

# %%
df_compare_final.to_csv(GOLD_DASHBOARD_MODEL_COMPARISON_PATH, index=False, encoding="utf-8-sig")
print(GOLD_DASHBOARD_MODEL_COMPARISON_PATH)
