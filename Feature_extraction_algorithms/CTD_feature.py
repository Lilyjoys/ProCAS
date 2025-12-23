import numpy as np
from propy import PyPro

def extract_ctd_features(sequences, lamda=1, properties=None):

    if properties is None:
        properties = [
            'Hydrophobicity', 'Hydrophilicity', 'Charge',
            'Polarity', 'Polarizability', 'SecondaryStructure'
        ]

    all_features = []
    for seq in sequences:
        obj = PyPro.GetProDes(seq)
        features = []

        ctd = obj.GetCTD()
        features += list(ctd.values())  # 21

        # Moran
        #features += list(obj.GetMoranAuto().values())
        # Geary
        #features += list(obj.GetGearyAuto().values())
        all_features.append(features)

    return np.array(all_features)


if __name__ == "__main__":
    sequences = ['ACDEFGHIKLMNPQRSTVWY', 'ACDYYGHIKLMPPPRSTVWY']
    feature_matrix = extract_ctd_features(sequences, lamda=1)

    print(feature_matrix)