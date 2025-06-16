from __future__ import print_function, division

import os, math, torch, pathlib
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch_geometric.data
from torch_geometric import utils
from torch_geometric import data
from pysmiles import read_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors
from models.LEGS_module import Scatter


class ToxicityDataset(Dataset):
    """
    Dataset for toxicity-labeled molecules from CSV format.
    Expected columns: CasNo, SMILES, Molecular Weight, LogP score, Drug Name, IUPAC name (if present), Toxicity
    """

    def __init__(self, csv_file, transform=None, prop_stat_dict=None, include_toxicity=True):
        """
        Args:
            csv_file (string): Path to the csv file with molecular data and toxicity labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            prop_stat_dict (string, optional): Path to statistics for property normalization.
            include_toxicity (bool): Whether to include toxicity as a prediction target.
        """
        
        self.df = pd.read_csv(csv_file)
        
        # Define property list - we'll compute some properties from SMILES
        self.prop_list = ['qed', 'HeavyAtomMolWt', 'MolWt', 'BalabanJ', 'BertzCT', 'Ipc', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'RingCount']
        
        if include_toxicity:
            self.prop_list.append('Toxicity')
        
        # Load statistics if provided
        if prop_stat_dict is not None:
            self.stats = np.load(prop_stat_dict, allow_pickle=True).item()
        else:
            self.stats = None

        self.transform = transform
        self.num_node_features = 10
        self.num_classes = len(self.prop_list)
        self.include_toxicity = include_toxicity
        
        # Pre-compute molecular properties for all valid SMILES
        self._precompute_properties()

    def _precompute_properties(self):
        """Pre-compute molecular properties from SMILES strings."""
        valid_indices = []
        self.molecular_data = {}
        
        for idx, row in self.df.iterrows():
            smiles = row['SMILES']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Compute molecular properties
                    props = self._compute_molecular_properties(mol)
                    
                    # Add toxicity if available
                    if self.include_toxicity and 'Toxicity' in row:
                        props['Toxicity'] = float(row['Toxicity'])
                    
                    self.molecular_data[idx] = {
                        'smiles': smiles,
                        'properties': props,
                        'toxicity_label': int(row['Toxicity']) if 'Toxicity' in row else 0,
                        'cas_no': row.get('CasNo', ''),
                        'drug_name': row.get('Drug Name', ''),
                        'iupac_name': row.get('IUPAC name', '')
                    }
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing SMILES {smiles} at index {idx}: {e}")
                continue
        
        self.valid_indices = valid_indices

    def _compute_molecular_properties(self, mol):
        """Compute molecular properties from RDKit mol object."""
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
        
        props = {}
        try:
            # Compute QED (if available)
            try:
                from rdkit.Chem import QED
                props['qed'] = QED.qed(mol)
            except:
                props['qed'] = 0.5  # Default value
            
            props['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
            props['MolWt'] = Descriptors.MolWt(mol)
            props['BalabanJ'] = Descriptors.BalabanJ(mol)
            props['BertzCT'] = Descriptors.BertzCT(mol)
            props['Ipc'] = Descriptors.Ipc(mol)
            props['TPSA'] = Descriptors.TPSA(mol)
            props['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            props['NumHDonors'] = Descriptors.NumHDonors(mol)
            props['RingCount'] = Descriptors.RingCount(mol)
            
        except Exception as e:
            print(f"Error computing properties: {e}")
            # Set default values
            for prop in self.prop_list:
                if prop not in props:
                    props[prop] = 0.0
        
        return props

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a single molecular sample with toxicity information."""
        actual_idx = self.valid_indices[idx]
        mol_data = self.molecular_data[actual_idx]
        
        smiles = mol_data['smiles']
        properties = mol_data['properties']
        toxicity_label = mol_data['toxicity_label']
        
        # Prepare property vector
        props = np.zeros(self.num_classes)
        no_zscore = np.zeros(self.num_classes)
        
        if self.stats is not None:
            # Apply z-score normalization
            for i, prop_name in enumerate(self.prop_list):
                if prop_name in properties:
                    prop_value = properties[prop_name]
                    if prop_name in self.stats:
                        z_scored = (prop_value - self.stats[prop_name]['mean']) / self.stats[prop_name]['std']
                        props[i] = z_scored
                    else:
                        props[i] = prop_value
                    no_zscore[i] = prop_value
        else:
            for i, prop_name in enumerate(self.prop_list):
                if prop_name in properties:
                    props[i] = properties[prop_name]
                    no_zscore[i] = properties[prop_name]

        # Create molecular graph
        try:
            mol = read_smiles(smiles)
            data = self._from_networkx_custom(mol)
        except Exception as e:
            print(f"Error creating graph for {smiles}: {e}")
            # Return a dummy graph
            data = torch_geometric.data.Data()
            data.x = torch.zeros(1, self.num_node_features)
            data.edge_index = torch.zeros(2, 0, dtype=torch.long)
            data.num_nodes = 1

        data.no_zscore_props = no_zscore
        data.y = torch.Tensor([props])
        data.toxicity_label = torch.tensor(toxicity_label, dtype=torch.long)
        data.smiles = smiles
        
        # optional - if we want this data
        data.cas_no = mol_data['cas_no']
        data.drug_name = mol_data['drug_name']
        data.iupac_name = mol_data['iupac_name']

        # Add node features
        self._add_node_features(data)

        if self.transform:
            return self.transform(data)
        else:
            return data

    def _add_node_features(self, data):
        """Add node features to the molecular graph."""
        if not hasattr(data, 'element') or len(data.element) == 0:
            # Default single node if no elements
            data.x = torch.zeros(1, self.num_node_features)
            return

        node_feats = []
        for element in data.element:
            node_feat = np.zeros(self.num_node_features)
            
            # One-hot encoding of atoms
            if element == 'C':
                node_feat[0] = 1.
            elif element == 'O':
                node_feat[1] = 1.
            elif element == 'N':
                node_feat[2] = 1.
            elif element == 'S':
                node_feat[3] = 1.
            
            # Pair encoding of atoms
            if element == 'C' or element == 'O':
                node_feat[4] = 1.
            if element == 'C' or element == 'N':
                node_feat[5] = 1.
            if element == 'C' or element == 'S':
                node_feat[6] = 1.
            if element == 'O' or element == 'N':
                node_feat[7] = 1.
            if element == 'O' or element == 'S':
                node_feat[8] = 1.
            if element == 'N' or element == 'S':
                node_feat[9] = 1.

            node_feats.append(node_feat)

        data.x = torch.Tensor(node_feats)

    def _from_networkx_custom(self, G):
        """Convert networkx graph to PyTorch Geometric data object."""
        import networkx as nx

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        data = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                if str(key) != "stereo":
                    data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

        for key, item in data.items():
            try:
                data[key] = torch.tensor(item)
            except ValueError:
                pass

        data['edge_index'] = edge_index.view(2, -1) if edge_index.numel() > 0 else torch.zeros(2, 0, dtype=torch.long)
        data = torch_geometric.data.Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()

        return data

    def get_toxicity_labels(self):
        """Return all toxicity labels for contrastive learning."""
        return [self.molecular_data[idx]['toxicity_label'] for idx in self.valid_indices]


class ToxicityScattering(object):
    """Scattering transform for toxicity dataset."""

    def __init__(self, scatter_model_name=None):
        model = Scatter(10, trainable_laziness=None)
        if scatter_model_name is None:
            raise ValueError("Please specify a pretrained scatter module. If you'd like to use an untrained model, specify\
            scatter_model_name='untrained'. Otherwise, use the .npy file of the model")
        elif scatter_model_name != 'untrained':
            model.load_state_dict(torch.load(scatter_model_name))
        model.eval()
        self.model = model
    
    def __call__(self, sample):
        props = sample.y
        toxicity_label = sample.toxicity_label
        scattered = self.model(sample)
        
        # Return scattered features, properties, toxicity label, and additional metadata
        return (scattered[0][0].detach(), 
                sample.y[0], 
                toxicity_label,
                sample.smiles,
                sample.cas_no,
                sample.drug_name,
                sample.iupac_name) 