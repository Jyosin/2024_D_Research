import torch
from transformers import BertTokenizer,BertForSequenceClassification
from datasets import load_dataset
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModel
import numpy as np

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
emotions = load_dataset("emotion")
train_ds = emotions["train"]

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

def extract_hidden_states(batch):
    inputs = {k:v.to(device) for k,v in batch.items()
                if k in tokenizer.model_input_names}
    
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def tokenize(batch):
    return tokenizer(batch["text"],padding=True, truncation = True)

emotions_encoded = emotions.map(tokenize, batched = True, batch_size = None)
emotions_encoded.set_format("torch",
                            columns = ["input_ids", "attention_mask","label"])

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched = True)

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

X_scaled = MinMaxScaler().fit_transform(X_train)
mapper = UMAP(n_components = 2, metric = "cosine").fit(X_scaled)

df_emb = pd.DataFrame(mapper.embedding_, columns = ["X","Y"])
df_emb["label"] = y_train

fig, axes = plt.subplots(2,3, figsize = (7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (label, cmap) in enumerate(zip(labels,cmaps)):
    df_emb_sub = df_emb.query(f"label =={i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap = cmap,
                    gridsize = 20, linewidths = (0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]),axes[i].set_yticks([])

plt.tight_layout()
plt.show()

# print(emotions_encoded["train"].column_names)
# print(train_ds[:5])

