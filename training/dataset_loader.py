import pandas as pd


def load_chembl_dataset(path):

    df = pd.read_csv(path)

    df = df.dropna(subset=["smiles"])

    smiles = df["smiles"].tolist()

    labels = df.iloc[:, 1:].values

    return smiles, labels
