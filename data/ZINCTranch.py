from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch_geometric.data
import math
import pathlib
from pysmiles import read_smiles
from torch_geometric import utils
from torch_geometric import data
from models.LEGS_module import Scatter

import networkx as nx
from pysmiles import read_smiles

class ZINCDataset(Dataset):
    """Zinc Tranch data"""

    def __init__(self, file_name, transform=None, prop_stat_dict=None, include_ki=True):
        

        self.prop_list = ['qed', 'HeavyAtomMolWt', 'MolWt', 'BalabanJ', 'BertzCT', 'Ipc', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'RingCount']
        if include_ki:
            self.prop_list.append('Ki')
        self.tranch = np.load(file_name, allow_pickle=True).item()
        if prop_stat_dict != None:
            self.stats = np.load(prop_stat_dict, allow_pickle=True).item()
        else:
            self.stats = None
        self.transform = transform
        self.num_node_features = 10
        self.num_classes = len(self.prop_list)
        self.smi = list(self.tranch.keys())
        # Store all SMILES strings for negative sampling
        self.all_smi = self.smi 

    def __len__(self):
        return len(self.smi)

    def _get_mol_data(self, smi_string):
        """Helper function to process a SMILES string into graph data and properties."""
        props = np.zeros(self.num_classes)
        no_zscore = np.zeros(self.num_classes) # This might not be needed for negative samples

        if self.stats is not None:
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi_string].get(entry) # Use .get for safety, though Ki might be missing
                if prop_value is not None:
                    z_scored = (prop_value - self.stats[entry]['mean']) / self.stats[entry]['std']
                    props[i] = z_scored
                # else: props[i] remains 0, or handle as error/default
        else:
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi_string].get(entry)
                if prop_value is not None:
                    props[i] = prop_value
                    no_zscore[i] = prop_value # Again, may not be relevant for negative

        mol = read_smiles(smi_string)
        graph_data = from_networkx_custom(mol)
        graph_data.y = torch.Tensor([props]) # Properties for the anchor
        graph_data.no_zscore_props = no_zscore # Store non-zscored if needed

        node_feats = []
        for i, entry in enumerate(graph_data.element):
            node_feat = np.zeros(self.num_node_features)
            if entry == 'C': node_feat[0] = 1.
            elif entry == 'O': node_feat[1] = 1.
            elif entry == 'N': node_feat[2] = 1.
            elif entry == 'S': node_feat[3] = 1.
            if entry == 'C' or entry == 'O': node_feat[4] = 1.
            if entry == 'C' or entry == 'N': node_feat[5] = 1.
            if entry == 'C' or entry == 'S': node_feat[6] = 1.
            if entry == 'O' or entry == 'N': node_feat[7] = 1.
            if entry == 'O' or entry == 'S': node_feat[8] = 1.
            if entry == 'N' or entry == 'S': node_feat[9] = 1.
            node_feats.append(node_feat)
        graph_data.x = torch.Tensor(node_feats)
        return graph_data, props # Return graph_data and its specific properties

    def __getitem__(self, idx):
        anchor_smi = self.smi[idx]
        anchor_graph_data, anchor_properties = self._get_mol_data(anchor_smi)

        # Positive sample is the same as the anchor
        positive_graph_data = anchor_graph_data 
        # Note: If transforms modify data in-place, a deepcopy might be needed for positive_graph_data
        # For Scattering, it seems to compute features based on input, so original graph_data is fine.

        # Negative sampling
        num_total_samples = len(self.all_smi)
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = np.random.randint(0, num_total_samples)
        
        negative_smi = self.all_smi[neg_idx]
        negative_graph_data, _ = self._get_mol_data(negative_smi) # We don't need props for negative

        if self.transform:
            # Apply transform to anchor, positive, and negative samples
            # The Scattering transform returns (features, properties)
            # We only need features for positive and negative in the contrastive loss
            anchor_transformed_features, anchor_transformed_properties = self.transform(anchor_graph_data)
            # Since positive is the same as anchor, its transformed features are the same.
            # Re-applying transform might be redundant if transform is deterministic and no in-place changes.
            # However, if transform has internal state or randomness, apply separately.
            # Let's assume transform is deterministic for now.
            positive_transformed_features, _ = self.transform(positive_graph_data) # Properties for positive are not used later
            negative_transformed_features, _ = self.transform(negative_graph_data) # Properties for negative are not used

            return (anchor_transformed_features, 
                    anchor_transformed_properties, # These are the Y values for the main regression task
                    positive_transformed_features, 
                    negative_transformed_features)
        else:
            # Return raw graph data if no transform
            # The model will expect (anchor_features, anchor_properties, positive_features, negative_features)
            # anchor_properties here is anchor_graph_data.y
            # For features, the model will likely use graph_data.x, graph_data.edge_index etc.
            # This part needs to align with how the model consumes raw graph data if transform is None.
            # Assuming the model's embed function can take this raw graph_data object.
            return (anchor_graph_data, 
                    anchor_graph_data.y.squeeze(0), # Squeeze to remove extra dimension if y is [1, num_props]
                    positive_graph_data, 
                    negative_graph_data)

class Scattering(object):

    def __init__(self, scatter_model_name=None):
        model = Scatter(10, trainable_laziness=None)
        if scatter_model_name == None:
            raise ValueError("Please specify a pretrained scatter module. If you'd like to use an untrained model, specify\
            scatter_model_name='untrained'. Otherwise, use the .npy file of the model")
        elif scatter_model_name != 'untrained':
            model.load_state_dict(torch.load(scatter_model_name))
        model.eval()
        self.model = model
    
    def __call__(self, sample_graph_data): # Modified to accept graph_data directly
        # sample.y was torch.Tensor([props]), so sample.y[0] was props
        # Now sample_graph_data.y is already torch.Tensor([props])
        # The Scattering transform expects a PyG Data object.
        props_tensor = sample_graph_data.y[0] # Get the actual properties tensor
        
        # The model call expects a PyG Data object. 
        # sample_graph_data already is this object.
        transformed_features_tuple = self.model(sample_graph_data) # model returns a tuple
        
        # Assuming transformed_features_tuple[0][0] is the desired feature tensor
        return transformed_features_tuple[0][0].detach(), props_tensor


def from_networkx_custom(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            if(str(key) != "stereo"):
                data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


