from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1"
)

model = AutoModel.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1"
)


def embed_smiles(smiles):

    tokens = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():

        outputs = model(**tokens)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze().numpy()
