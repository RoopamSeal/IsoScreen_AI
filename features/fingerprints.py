from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def compute_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    arr = np.zeros((2048,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    return arr
