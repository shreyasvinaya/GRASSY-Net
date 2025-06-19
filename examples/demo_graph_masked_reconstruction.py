"""
Demo script for Graph Masked Reconstruction in GRASSY-Net.

This script demonstrates how to use the GraphMaskedReconstructor class
to perform masked reconstruction on molecular graphs with scaffold guidance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

# Import GRASSY components
from models.GRASSY_enhanced import GraphMaskedReconstructor, ScaffoldEncoder
from datasets.toxicity_dataset import ToxicityScattering

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    print("PyTorch Geometric available: Using full graph functionality")
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not available: Using fallback tensor implementation")
    TORCH_GEOMETRIC_AVAILABLE = False


def create_sample_molecular_graph(smiles, node_feature_dim=10):
    """
    Create a sample molecular graph from SMILES string.
    
    Args:
        smiles: SMILES string
        node_feature_dim: Number of node features
    
    Returns:
        PyTorch Geometric Data object or tensor (fallback)
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        # Fallback: return random tensor data
        return torch.randn(1, node_feature_dim * 5)  # Simulate flattened graph
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Create node features (simple atomic properties)
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                atom.GetTotalValence(),
                atom.GetNumRadicalElectrons(),
                int(atom.IsInRing()),
                atom.GetImplicitValence()
            ]
            node_features.append(features)
        
        # Pad or truncate to desired feature dimension
        node_features = np.array(node_features, dtype=np.float32)
        if node_features.shape[1] < node_feature_dim:
            padding = np.zeros((node_features.shape[0], node_feature_dim - node_features.shape[1]))
            node_features = np.concatenate([node_features, padding], axis=1)
        else:
            node_features = node_features[:, :node_feature_dim]
        
        # Create edge indices
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Undirected graph
            
            # Simple edge features
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing())
            ]
            edge_features.extend([bond_features, bond_features])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    except Exception as e:
        print(f"Error creating graph for {smiles}: {e}")
        # Return dummy graph
        num_nodes = np.random.randint(5, 15)
        x = torch.randn(num_nodes, node_feature_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        return Data(x=x, edge_index=edge_index)


def create_sample_batch_data(smiles_list, node_feature_dim=10):
    """Create a batch of molecular graphs."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        # Fallback: return stacked tensors
        graphs = [create_sample_molecular_graph(smiles, node_feature_dim) for smiles in smiles_list]
        return torch.stack(graphs)
    
    graphs = [create_sample_molecular_graph(smiles, node_feature_dim) for smiles in smiles_list]
    return Batch.from_data_list(graphs)


def demo_graph_masked_reconstruction():
    """Demonstrate graph masked reconstruction functionality."""
    print("=" * 60)
    print("GRASSY-Net Graph Masked Reconstruction Demo")
    print("=" * 60)
    
    # Configuration
    node_feature_dim = 10
    edge_feature_dim = 3
    hidden_dim = 128
    scaffold_dim = 64
    batch_size = 4
    
    # Sample molecular data
    sample_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CCN(CC)CC",  # Triethylamine
        "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    ]
    
    sample_scaffolds = [
        "CO",  # Alcohol
        "c1ccccc1",  # Benzene ring
        "CCN",  # Amine
        "c1ccccc1C(=O)O"  # Benzoic acid
    ]
    
    print(f"Sample molecules: {sample_smiles}")
    print(f"Sample scaffolds: {sample_scaffolds}")
    print()
    
    # Create molecular graphs
    print("Creating molecular graphs...")
    batch_data = create_sample_batch_data(sample_smiles, node_feature_dim)
    print(f"Batch data type: {type(batch_data)}")
    
    if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
        print(f"Number of nodes: {batch_data.x.size(0)}")
        print(f"Node feature dim: {batch_data.x.size(1)}")
        print(f"Number of edges: {batch_data.edge_index.size(1)}")
        if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
            print(f"Edge feature dim: {batch_data.edge_attr.size(1)}")
    else:
        print(f"Fallback tensor shape: {batch_data.shape}")
    print()
    
    # Initialize components
    print("Initializing Graph Masked Reconstructor...")
    graph_reconstructor = GraphMaskedReconstructor(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim if TORCH_GEOMETRIC_AVAILABLE else None,
        hidden_dim=hidden_dim,
        scaffold_dim=scaffold_dim,
        num_gnn_layers=3
    )
    
    scaffold_encoder = ScaffoldEncoder(
        embedding_dim=scaffold_dim,
        vocab_size=100
    )
    
    print(f"Graph Reconstructor initialized with:")
    print(f"  - Node feature dim: {node_feature_dim}")
    print(f"  - Edge feature dim: {edge_feature_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Scaffold dim: {scaffold_dim}")
    print(f"  - PyTorch Geometric: {graph_reconstructor.torch_geometric_available}")
    print()
    
    # Encode scaffolds
    print("Encoding scaffold information...")
    scaffold_embeddings = scaffold_encoder(sample_scaffolds)
    print(f"Scaffold embeddings shape: {scaffold_embeddings.shape}")
    print()
    
    # Perform graph masked reconstruction
    print("Performing graph masked reconstruction...")
    with torch.no_grad():
        outputs = graph_reconstructor(batch_data, scaffold_embeddings)
    
    print("Reconstruction outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}")
        else:
            print(f"  - {key}: {type(value)}")
    print()
    
    # Compute reconstruction loss
    print("Computing reconstruction loss...")
    loss_dict = graph_reconstructor.compute_reconstruction_loss(outputs, batch_data)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.item():.4f}")
        else:
            print(f"  - {key}: {value}")
    print()
    
    # Demonstrate masking patterns
    print("Analyzing masking patterns...")
    node_mask = outputs['node_mask']
    feature_mask = outputs['feature_mask']
    
    if node_mask is not None:
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
            total_nodes = batch_data.x.size(0)
            masked_nodes = node_mask.sum().item()
            print(f"Node masking: {masked_nodes}/{total_nodes} nodes masked ({100*masked_nodes/total_nodes:.1f}%)")
        
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
            total_features = batch_data.x.numel()
            masked_features = feature_mask.sum().item()
            print(f"Feature masking: {masked_features}/{total_features} features masked ({100*masked_features/total_features:.1f}%)")
    else:
        total_features = batch_data.numel() if not TORCH_GEOMETRIC_AVAILABLE else batch_data.x.numel()
        masked_features = feature_mask.sum().item()
        print(f"Feature masking: {masked_features}/{total_features} features masked ({100*masked_features/total_features:.1f}%)")
    print()
    
    # Demonstrate different masking ratios
    print("Testing different masking ratios...")
    test_ratios = [0.05, 0.15, 0.25, 0.35]
    
    for ratio in test_ratios:
        # Temporarily change masking ratio
        original_node_ratio = graph_reconstructor.node_mask_ratio
        original_feature_ratio = graph_reconstructor.feature_mask_ratio
        
        graph_reconstructor.node_mask_ratio = ratio
        graph_reconstructor.feature_mask_ratio = ratio * 0.7  # Slightly less for features
        
        with torch.no_grad():
            test_outputs = graph_reconstructor(batch_data, scaffold_embeddings)
            test_loss = graph_reconstructor.compute_reconstruction_loss(test_outputs, batch_data)
        
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
            masked_nodes = test_outputs['node_mask'].sum().item() if test_outputs['node_mask'] is not None else 0
            masked_features = test_outputs['feature_mask'].sum().item()
            total_nodes = batch_data.x.size(0)
            total_features = batch_data.x.numel()
            
            print(f"  Ratio {ratio:.2f}: Loss={test_loss['total_loss'].item():.4f}, "
                  f"Nodes={masked_nodes}/{total_nodes}, Features={masked_features}/{total_features}")
        else:
            masked_features = test_outputs['feature_mask'].sum().item()
            total_features = batch_data.numel()
            print(f"  Ratio {ratio:.2f}: Loss={test_loss['total_loss'].item():.4f}, "
                  f"Features={masked_features}/{total_features}")
        
        # Restore original ratios
        graph_reconstructor.node_mask_ratio = original_node_ratio
        graph_reconstructor.feature_mask_ratio = original_feature_ratio
    print()
    
    # Visualization of reconstruction quality
    print("Analyzing reconstruction quality...")
    reconstructed_nodes = outputs['reconstructed_nodes']
    
    if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
        original_nodes = batch_data.x
        mse_per_feature = F.mse_loss(reconstructed_nodes, original_nodes, reduction='none').mean(dim=0)
        
        print("MSE per node feature:")
        for i, mse in enumerate(mse_per_feature):
            print(f"  Feature {i:2d}: {mse.item():.4f}")
        
        # Overall reconstruction statistics
        total_mse = F.mse_loss(reconstructed_nodes, original_nodes).item()
        masked_mse = F.mse_loss(reconstructed_nodes[feature_mask], original_nodes[feature_mask]).item() if feature_mask.sum() > 0 else 0
        
        print(f"\nOverall reconstruction MSE: {total_mse:.4f}")
        print(f"Masked regions MSE: {masked_mse:.4f}")
        print(f"Reconstruction improvement: {(masked_mse/total_mse if total_mse > 0 else 1.0):.2f}x")
    print()
    
    print("Graph Masked Reconstruction Demo completed successfully!")
    print("=" * 60)


def demo_integration_with_enhanced_grassy():
    """Demo integration with EnhancedGRASSY model."""
    print("\nDemonstrating integration with EnhancedGRASSY...")
    
    # Create mock hyperparameters
    class MockHparams:
        def __init__(self):
            self.alpha = 1.0
            self.beta = 1.0
            self.gamma = 0.1
            self.delta = 0.1
            self.epsilon = 0.1
            self.zeta = 0.1  # Graph reconstruction weight
            self.input_dim = 50
            self.bottle_dim = 20
            self.hidden_dim = 128
            self.scaffold_dim = 64
            self.learning_rate = 0.001
            self.n_gpus = 0
            
            # Graph-specific parameters
            self.use_graph_reconstruction = True
            self.node_feature_dim = 10
            self.edge_feature_dim = 3
            self.num_gnn_layers = 3
    
    hparams = MockHparams()
    
    # This would normally import and use EnhancedGRASSY
    print("Mock EnhancedGRASSY configuration:")
    print(f"  - Graph reconstruction enabled: {hparams.use_graph_reconstruction}")
    print(f"  - Graph reconstruction weight (zeta): {hparams.zeta}")
    print(f"  - Node feature dimension: {hparams.node_feature_dim}")
    print(f"  - Edge feature dimension: {hparams.edge_feature_dim}")
    print(f"  - Number of GNN layers: {hparams.num_gnn_layers}")
    
    print("\nTo use with EnhancedGRASSY:")
    print("1. Set use_graph_reconstruction=True in hyperparameters")
    print("2. Include graph_data in your batch format: (x, y, toxicity_labels, scaffold_smarts, graph_data)")
    print("3. The model will automatically use GraphMaskedReconstructor when graph_data is provided")
    print("4. Graph reconstruction loss will be added to the total loss with weight 'zeta'")


if __name__ == "__main__":
    try:
        demo_graph_masked_reconstruction()
        demo_integration_with_enhanced_grassy()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc() 