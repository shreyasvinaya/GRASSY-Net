# Enhanced GRASSY-Net: Molecular Graph Generation with Contrastive Learning and Masked Reconstruction

This repository extends the original GRASSY-Net (Geometric Scattering) model with two key enhancements:
1. **Contrastive Loss** based on toxicity labels for better molecular representation learning
2. **Masked Reconstruction** with scaffold information for improved generation capabilities

## 🆕 New Features

### 1. Contrastive Learning for Toxicity-Aware Representations
- Learns molecular representations that cluster molecules with similar toxicity profiles
- Implements a contrastive loss that pulls together molecules with the same toxicity label and pushes apart those with different labels
- Supports binary toxicity classification (toxic/non-toxic)

### 2. Masked Reconstruction with Scaffold Information
- Adds masked autoencoder-style training with scaffold conditioning
- Takes scaffold information in SMARTS format as additional input
- Randomly masks portions of molecular features and reconstructs them using scaffold information
- Helps the model learn better scaffold-aware molecular representations

### 2.1. Graph Masked Reconstruction
- Advanced graph-based masked reconstruction using Graph Neural Networks (GNNs)
- **Node-level masking**: Randomly masks 15% of graph nodes with learnable mask tokens
- **Feature-level masking**: Masks 10% of node features within non-masked nodes
- **Graph Convolutional Encoding**: Uses GCN layers with batch normalization for graph encoding
- **Attention-based Scaffold Integration**: Combines graph embeddings with scaffold information via multi-head attention
- **Edge Feature Reconstruction**: Reconstructs both node and edge features when available
- **Fallback Support**: Gracefully handles non-graph data with tensor-based implementation
- Enables more sophisticated understanding of molecular graph structure and chemical bonds

### 3. Scaffold-Constrained Generation
- Forces the model to generate molecules that always contain a specific scaffold structure
- Reads target scaffold configurations from YAML/JSON files
- Supports real-time scaffold switching during inference
- Validates scaffold compliance in generated molecules
- Enables targeted drug design with required structural motifs

### 3. Enhanced Dataset Support
- New `ToxicityDataset` class for loading CSV data with toxicity labels
- Expected CSV columns: `CasNo`, `SMILES`, `Molecular Weight`, `LogP score`, `Drug Name`, `IUPAC name (if present)`, `Toxicity`
- Automatic molecular property computation from SMILES strings
- Flexible data loading with support for missing values

## 📁 Project Structure

```
GRASSY-Net/
├── models/
│   ├── GRASSY_model.py           # Original GRASSY model
│   ├── GRASSY_enhanced.py        # Enhanced model with new features
│   └── LEGS_module.py            # Geometric scattering components
├── datasets/
│   ├── load_ZINC_tranche.py      # Original ZINC dataset loader
│   └── toxicity_dataset.py       # New toxicity dataset loader
├── scripts/
│   ├── train_grassy.py           # Original training script
│   └── train_enhanced_grassy.py  # Enhanced training script
├── examples/
│   ├── demo_enhanced_grassy.py                  # Demonstration script for enhanced features
│   ├── demo_scaffold_constrained_generation.py  # Scaffold generation demo
│   └── demo_graph_masked_reconstruction.py      # Graph reconstruction demo 
└── README.md                                    # This file
```

## 🚀 Quick Start

### Installation

Install the required dependencies:

```bash
pip install torch pytorch-lightning pandas numpy scikit-learn
pip install rdkit-pypi pysmiles torch-geometric torch-scatter
pip install matplotlib seaborn tqdm
```

### Running the Enhanced Model

#### 1. With Sample Data (Quick Demo)

```bash
# Create and run with sample toxicity data
cd examples/
python demo_enhanced_grassy.py --create_sample --n_epochs 20
```

#### 2. With Your Own Toxicity Dataset

```bash
# Train with enhanced features
python scripts/train_enhanced_grassy.py \
    --dataset_type toxicity \
    --csv_file path/to/your/toxicity_data.csv \
    --use_enhanced_model \
    --include_scaffolds \
    --gamma 0.1 \
    --delta 0.1 \
    --n_epochs 100
```

#### 3. Scaffold-Constrained Generation

```bash
# Train with scaffold constraints using configuration file
python scripts/train_enhanced_grassy.py \
    --dataset_type toxicity \
    --csv_file path/to/your/toxicity_data.csv \
    --use_enhanced_model \
    --use_scaffold_generation \
    --scaffold_config_path configs/scaffold_configs/benzene_scaffold.yaml \
    --n_epochs 100

# Or specify scaffold directly
python scripts/train_enhanced_grassy.py \
    --dataset_type toxicity \
    --csv_file path/to/your/toxicity_data.csv \
    --use_enhanced_model \
    --use_scaffold_generation \
    --target_scaffold "c1ccccc1" \
    --constraint_strength 1.0 \
    --n_epochs 100
```

#### 4. Demo Scaffold-Constrained Generation

```bash
# Run the scaffold generation demo
python examples/demo_scaffold_constrained_generation.py \
    --config configs/scaffold_configs/benzene_scaffold.yaml \
    --n_epochs 15

# Create custom scaffold configuration and run demo
python examples/demo_scaffold_constrained_generation.py \
    --scaffold "c1ccncc1" \
    --n_epochs 15
```

#### 5. Graph Masked Reconstruction Demo

```bash
# Run the graph reconstruction demo with sample molecular data
python examples/demo_graph_masked_reconstruction.py
```

#### 6. Original GRASSY Model (Backward Compatibility)

```bash
# Train original model on ZINC data
python scripts/train_enhanced_grassy.py \
    --dataset_type zinc \
    --zinc_tranch P14416_BindingDB_train \
    --n_epochs 100
```

## 📊 CSV Data Format

Your toxicity dataset should be a CSV file with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `CasNo` | Chemical Abstracts Service number | Yes |
| `SMILES` | SMILES string representation | Yes |
| `Molecular Weight` | Molecular weight (will be computed if missing) | No |
| `LogP score` | Partition coefficient (will be computed if missing) | No |
| `Drug Name` | Name of the compound | No |
| `IUPAC name (if present)` | IUPAC nomenclature | No |
| `Toxicity` | Binary toxicity label (0=non-toxic, 1=toxic) | Yes |

### Example CSV:

```csv
CasNo,SMILES,Molecular Weight,LogP score,Drug Name,IUPAC name (if present),Toxicity
50-00-0,C=O,30.03,-0.77,Formaldehyde,methanal,1
64-17-5,CCO,46.07,-0.31,Ethanol,ethanol,0
71-43-2,C1=CC=CC=C1,78.11,2.13,Benzene,benzene,1
```

## 📋 Scaffold Configuration Files

### Scaffold Config Format

Create configuration files to specify target scaffolds for constrained generation:

**YAML Format (`configs/scaffold_configs/benzene_scaffold.yaml`):**
```yaml
# Target scaffold in SMARTS format
target_scaffold: "c1ccccc1"

# Constraint enforcement strength (0.0 = no constraint, 1.0 = strict enforcement)
constraint_strength: 1.0

# Generation parameters
generation_params:
  temperature: 1.0           # Sampling temperature for diversity
  num_samples: 10           # Number of molecules to generate per latent code
  enforce_constraint: true  # Whether to enforce scaffold constraint during generation

# Model parameters specific to scaffold generation
model_params:
  scaffold_dim: 128         # Dimension of scaffold embeddings
  hidden_dim: 256          # Hidden layer dimension for constraint networks
  epsilon: 0.1             # Weight for scaffold constraint loss
```

**JSON Format (`configs/scaffold_configs/pyridine_scaffold.json`):**
```json
{
  "target_scaffold": "c1ccncc1",
  "constraint_strength": 0.9,
  "generation_params": {
    "temperature": 1.2,
    "num_samples": 15,
    "enforce_constraint": true
  },
  "model_params": {
    "scaffold_dim": 128,
    "hidden_dim": 256,
    "epsilon": 0.15
  }
}
```

### Example Scaffolds

Common scaffolds available for testing:
- **Benzene**: `c1ccccc1` (aromatic compounds)
- **Pyridine**: `c1ccncc1` (nitrogen heterocycles)
- **Thiophene**: `c1ccsc1` (sulfur heterocycles)
- **Furan**: `c1ccoc1` (oxygen heterocycles)
- **Imidazole**: `c1c[nH]cn1` (nitrogen heterocycles)
- **Indole**: `c1ccc2[nH]ccc2c1` (fused ring systems)

## 🔧 Model Configuration

### Enhanced GRASSY Parameters

```python
# Original GRASSY parameters
alpha = 0.01      # Regression loss weight
beta = 0.0005     # KL divergence weight

# New enhanced parameters
gamma = 0.1       # Contrastive loss weight
delta = 0.1       # Masked reconstruction loss weight
epsilon = 0.1     # Scaffold constraint loss weight
zeta = 0.1        # Graph reconstruction loss weight 
scaffold_dim = 128    # Scaffold embedding dimension
contrastive_margin = 1.0
contrastive_temperature = 0.07

# Graph reconstruction parameters 
use_graph_reconstruction = True  # Enable graph-based masked reconstruction
node_feature_dim = 10           # Number of node features in molecular graphs
edge_feature_dim = 3            # Number of edge features
num_gnn_layers = 3              # Number of GNN layers
```

### Training Options

```bash
# Key command-line arguments for enhanced training:
--use_enhanced_model      # Use EnhancedGRASSY instead of original
--include_scaffolds       # Enable masked reconstruction with scaffolds
--gamma 0.1              # Weight for contrastive loss
--delta 0.1              # Weight for masked reconstruction loss
--dataset_type toxicity  # Use toxicity dataset instead of ZINC
--csv_file path/data.csv # Path to your toxicity CSV file
```

## 📈 Model Architecture

### Enhanced Components

1. **ContrastiveLoss Module**
   - Computes pairwise distances between molecular embeddings
   - Applies contrastive loss based on toxicity labels
   - Configurable margin and temperature parameters

2. **ScaffoldEncoder Module**
   - Character-level LSTM encoder for SMARTS strings
   - Converts variable-length SMARTS to fixed-size embeddings
   - Supports vocabulary of chemical characters and symbols

3. **MaskedReconstructor Module**
   - Randomly masks 15% of molecular features
   - Fuses original molecular embeddings with scaffold information
   - Reconstructs masked features using both molecular and scaffold context

4. **GraphMaskedReconstructor Module**
   - Advanced GNN-based reconstruction for molecular graph data
   - Node-level masking with learnable mask tokens
   - Multi-layer Graph Convolutional Networks with batch normalization
   - Multi-head attention for scaffold-graph integration
   - Reconstructs both node features and edge attributes
   - Automatic fallback to tensor-based processing when PyTorch Geometric unavailable

### Loss Function

The total loss combines multiple components:

```
Total Loss = Reconstruction Loss + α×Regression Loss + β×KL Loss + γ×Contrastive Loss + δ×Masked Reconstruction Loss + ε×Scaffold Constraint Loss + ζ×Graph Reconstruction Loss
```

Where:
- **Reconstruction Loss**: Standard VAE reconstruction loss
- **Regression Loss**: Property prediction loss (original GRASSY)
- **KL Loss**: KL divergence for variational component
- **Contrastive Loss**: Toxicity-based contrastive learning
- **Masked Reconstruction Loss**: Scaffold-conditioned reconstruction
- **Scaffold Constraint Loss**: Forces generation of molecules with specific scaffolds
- **Graph Reconstruction Loss**: GNN-based reconstruction of masked graph nodes and features

## 📊 Results and Analysis

The enhanced model produces:

1. **Molecular Embeddings**: Latent representations that cluster by toxicity
2. **Property Predictions**: Maintains original GRASSY prediction capabilities
3. **Contrastive Representations**: Embeddings optimized for toxicity discrimination
4. **Scaffold-Aware Features**: Representations that incorporate structural scaffold information

### Visualization

The demonstration script generates:
- PCA visualization of molecular embeddings colored by toxicity
- Training loss curves for all loss components
- Analysis of embedding quality and clustering

## 🔬 Usage Examples

### 1. Basic Enhanced Training

```python
from models.GRASSY_enhanced import EnhancedGRASSY
from datasets.toxicity_dataset import ToxicityDataset, ToxicityScattering

# Load toxicity dataset
transform = ToxicityScattering(scatter_model_name='untrained')
dataset = ToxicityDataset('toxicity_data.csv', transform=transform)

# Initialize enhanced model
model = EnhancedGRASSY(hparams)

# Train with contrastive loss and masked reconstruction
trainer.fit(model, train_loader, val_loader)
```

### 2. Extract Embeddings for Analysis

```python
# Extract molecular embeddings
embeddings, toxicity_labels = [], []
for batch in dataloader:
    x, y, tox_labels = batch
    z, mu, logvar = model.embed(x)
    embeddings.append(z.detach().numpy())
    toxicity_labels.extend(tox_labels)

# Analyze clustering by toxicity
from sklearn.manifold import TSNE
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
```

### 3. Generate Molecules with Scaffold Conditioning

```python
# Generate molecules conditioned on scaffold
scaffold_smarts = ['c1ccccc1']  # Benzene scaffold
scaffold_embedding = model.scaffold_encoder(scaffold_smarts)

# Sample from latent space and decode
z_sample = torch.randn(1, model.bottle_dim)
reconstructed = model.decode(z_sample)
```

## 🧪 Experimental Features

### Scaffold Generation Strategies

The model supports different scaffold extraction methods:

1. **Murcko Scaffolds**: Extract core scaffold using RDKit
2. **Custom SMARTS**: User-defined scaffold patterns
3. **Learned Scaffolds**: Automatically discovered structural motifs

### Contrastive Learning Variants

- **InfoNCE**: Alternative contrastive loss formulation
- **Triplet Loss**: Margin-based triplet learning
- **Multi-class**: Support for multi-class toxicity labels

## 📚 Research Applications

This enhanced model is particularly useful for:

1. **Drug Discovery**: Generating molecules with desired toxicity profiles
2. **Chemical Safety**: Predicting and understanding molecular toxicity
3. **Scaffold Hopping**: Generating molecules with preserved core structures
4. **Representation Learning**: Learning better molecular embeddings for downstream tasks

## 🤝 Contributing

To contribute to the enhanced GRASSY model:

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Write tests and documentation
5. Submit a pull request

## 📄 License

This project maintains the same license as the original GRASSY-Net repository.

## 🙏 Acknowledgments

- Original GRASSY-Net authors for the foundational geometric scattering approach
- RDKit community for molecular informatics tools
- PyTorch Lightning for the training framework

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **RDKit Import Error**: Install rdkit-pypi package
3. **Scaffold Generation Fails**: Check SMILES validity in your dataset
4. **Contrastive Loss NaN**: Reduce learning rate or contrastive temperature

### Performance Tips

- Use GPU acceleration for faster training
- Precompute molecular properties for large datasets
- Use data augmentation for better generalization
- Monitor loss components individually during training

---

For more information and advanced usage, see the example scripts and model documentation. 