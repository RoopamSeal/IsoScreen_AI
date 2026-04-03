from rdkit import Chem


def standardize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    Chem.SanitizeMol(mol)

    return Chem.MolToSmiles(mol)