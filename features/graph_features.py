import deepchem as dc
from rdkit import Chem


featurizer = dc.feat.MolGraphConvFeaturizer()


def featurize_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    graph = featurizer.featurize([mol])

    return graph[0]
