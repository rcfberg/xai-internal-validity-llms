#The following code includes the code for training the BERT (and DistilBERT) model and running the LIME and SHAP analyses. Due to GDPR, the data is not included in the code.

#installing libraries for training the BERT model.
pip install transformers torch pandas numpy scikit-learn

#importing libraries for training the BERT model
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up paths
data_path = '/content/twitterpredictor_data_final.csv'
output_dir = '/content/classifier_output'

id2label = {0: "EI POL", 1: "POL"}
label2id = {"EI POL": 0, "POL": 1}

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased") #for DistilBERT, use "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained("bert-large-cased", num_labels=2, id2label=id2label, label2id=label2id) #for DistilBERT, use "distilbert-base-uncased"
model.to(device)

# Load and preprocess data
data = pd.read_csv(data_path)
print("Data sample:")
print(data.head())
print("\nClass distribution:")
print(data['Koodi'].value_counts(normalize=True))

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet = self.data.iloc[idx]['Twiitti_final']
        label = self.data.iloc[idx]['Koodi']

        encoding = self.tokenizer(tweet, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split the dataset
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)
test_dataset = CustomDataset(test_data, tokenizer)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data['Koodi']), y=data['Koodi'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "kappa": kappa
    }

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="kappa",
    greater_is_better=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()


# Evaluate the model on the test dataset
print("Evaluating on test dataset:")
test_results = trainer.evaluate(test_dataset)

# Print all metrics
for metric, value in test_results.items():
    if metric.startswith("eval_"):
        print(f"{metric[5:]}: {value:.4f}")

# Save the model
model_save_path = os.path.join(output_dir, 'final_model')
trainer.save_model(model_save_path)
print(f"Model saved to {model_save_path}")

#importing data and libraries for LIME and SHAP

#specifying data path
data_path = 'twitterpredictor_data_final.csv'
data = pd.read_csv(data_path)
data = data.dropna(subset=['Twiitti_final'])  # ensure no NaNs
print("Data sample for interpretation:")
print(data.head())

# select a single tweet for LIME and SHAP demonstration
selected_text = data.iloc[1]['Twiitti_final']
print("Selected tweet 

pip install lime

# Specify GPU
import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths and labels (change path to your own setup if needed)
model_path = '/content/classifier_bert/final_model'  # Your saved fine-tuned model path
id2label = {0: "EI POL", 1: "POL"}
label2id = {"EI POL": 0, "POL": 1}
class_names = ["EI POL", "POL"]

# Load tokenizer and model from your fine-tuned checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2, id2label=id2label, label2id=label2id
)
model.to(device)
model.eval()

# Predictor function for LIME
def predictor(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probas = F.softmax(logits, dim=1).detach().cpu().numpy()

    print(f"Class order: {list(id2label.values())}")
    print(f"Predicted probabilities: {probas}")

    return probas

# Example text to explain
text = "What is #mobile testing? It's that #korona'tests are taken to the testing site. And this is how it works in practice outside of @XXXXX's construction site. Fast and preventative. #koronafi <URL>using @XXXXX"

# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=class_names)

# Generate explanation
exp = explainer.explain_instance(
    text_instance=text,
    classifier_fn=predictor,
    num_features=15,   # Top tokens to display
    num_samples=2000   # Number of perturbed samples LIME will generate
)

# Show in notebook
exp.show_in_notebook(text=text)

# Optional: Get the token weights programmatically
token_weights = exp.as_list()  # Example: [('mobility', 0.16), ('Toyota', -0.11), ...]
print(token_weights)

#importing libraries for SHAP and conducting the analysis

pip install mplcursors
pip install ipympl
matplotlib inline
pip install lime
pip install shap

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy as sp
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import MaxAbsScaler
import mplcursors
from IPython.display import display

sns.set()

def predictor(x):
    # Returns the log-odds for class 1 ("POL") from probabilities
    probs = predict_proba(x)
    # Instead of sp.special.logit, directly return probabilities for class 1
    return probs[:, 1]

'''
def predictor(x):
    # Returns the log-odds for class 1 ("POL") from probabilities
    probs = predict_proba(x)
    p_class1 = probs[:, 1]
    val = sp.special.logit(p_class1)
    return val

'''

def predict_proba(texts):
    # Ensure texts is a list of strings. If SHAP passes tokens as a list of strings, join them:
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    # If texts are lists of tokens, join them into a single string:
    texts = [" ".join(t) if isinstance(t, list) else t for t in texts]

    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**encoding.to(device))
        probs = nn.Softmax(dim=1)(outputs.logits)
    return probs.cpu().numpy()

def f_batch(x):
    return predictor(x)


tokenizer = AutoTokenizer.from_pretrained("bert-large-cased", use_fast=True)

explainer = shap.Explainer(
    f_batch,
    masker=shap.maskers.Text(tokenizer=tokenizer)
)

def calc_shap_values(text):
    shap_values = explainer([text])
    return shap_values

def show_shap_values(text, plot_type='bar'):
    shap_values = calc_shap_values(text)
    if plot_type == 'bar':
        shap.plots.bar(shap_values[0])
    else:
        shap.plots.text(shap_values[0])

clean_text = selected_text.replace("…", "...")
clean_text = clean_text.strip()  # remove leading/trailing whitespace

# Show SHAP values
show_shap_values(selected_text, plot_type='text')
show_shap_values(selected_text, plot_type='bar')