from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1"
)

model = AutoModelForCausalLM.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1"
)


def generate_smiles(prompt="C", n=10):

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=40,
        num_return_sequences=n,
        do_sample=True
    )

    return tokenizer.batch_decode(outputs)
