import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# -----------------------------
# Atom feature
# -----------------------------
def atom_features(atom: Chem.rdchem.Atom):
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs()
    explicit_valence = atom.GetExplicitValence()
    total_valence = atom.GetTotalValence()
    degree = atom.GetDegree()
    hybridization = int(atom.GetHybridization())
    radical_e = atom.GetNumRadicalElectrons()
    aromatic = int(atom.GetIsAromatic())

    ring_member = int(atom.IsInRing())
    ring_count = sum(atom.IsInRingSize(i) for i in range(3, 8))
    ring_3 = int(atom.IsInRingSize(3))
    ring_4 = int(atom.IsInRingSize(4))
    ring_5 = int(atom.IsInRingSize(5))
    ring_6 = int(atom.IsInRingSize(6))
    ring_7 = int(atom.IsInRingSize(7))
    chiral = int(atom.GetChiralTag())
    neighbors = len(atom.GetNeighbors())
    heavy_neighbors = sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() > 1)

    try:
        g_charge = int(atom.GetDoubleProp("_GasteigerCharge") * 100)
    except:
        g_charge = 0

    pt = Chem.GetPeriodicTable()
    try:
        electronegativity = pt.GetElectronegativity(atomic_num) or 0.0
    except:
        electronegativity = 0.0

    try:
        rc = pt.GetRcovalent(atomic_num)
        volume = rc ** 3
    except:
        volume = 0.0

    h_donor = int(atomic_num in [7, 8] and total_h > 0)
    h_acceptor = int(atomic_num in [7, 8])

    return [
        atomic_num, formal_charge, total_h, explicit_valence,
        total_valence, degree, hybridization, radical_e, aromatic,
        ring_member, ring_count, ring_3, ring_4, ring_5, ring_6, ring_7,
        chiral, neighbors, heavy_neighbors,
        g_charge, electronegativity, h_donor, h_acceptor, volume
    ]


# -----------------------------
# SMILES -> PyG Data
# -----------------------------
def smiles_to_data(smiles: str, y: float):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles
    )


# -----------------------------
# Dataset
# -----------------------------
class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, targets, atom_features_list=None):

        self.data_list = []
        for i, (s, y) in enumerate(zip(smiles_list, targets)):
            if y is None or np.isnan(y):
                continue

            data = smiles_to_data(s, y)
            if data is None:
                continue


            if atom_features_list is not None:
                atom_feat = atom_features_list[i]
                data.x_extra = torch.tensor(atom_feat, dtype=torch.float32)

            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

