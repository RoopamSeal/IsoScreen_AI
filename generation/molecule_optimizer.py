from generation.smiles_generator import generate_smiles


def optimize_molecule(seed):

    candidates = generate_smiles(seed, 50)

    valid = []

    for s in candidates:

        if len(s) > 3:
            valid.append(s)

    return valid
