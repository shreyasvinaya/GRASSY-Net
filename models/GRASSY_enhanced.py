import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import random

# Import the new scaffold-constrained generator
from .scaffold_constrained_generator import ScaffoldConstrainedGenerator


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for toxicity-based molecular representation learning.
    Pulls together molecules with same toxicity labels and pushes apart those with different labels.
    """
    
    def __init__(self, margin=1.0, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings, toxicity_labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim] - molecular embeddings
            toxicity_labels: [batch_size] - binary toxicity labels (0: non-toxic, 1: toxic)
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create label mask: 1 if same toxicity, 0 if different
        labels_expanded = toxicity_labels.unsqueeze(0)
        label_mask = (labels_expanded == labels_expanded.T).float()
        
        # Positive pairs (same toxicity)
        pos_mask = label_mask * (1 - torch.eye(batch_size, device=embeddings.device))
        pos_distances = distances * pos_mask
        
        # Negative pairs (different toxicity)
        neg_mask = (1 - label_mask)
        neg_distances = distances * neg_mask
        
        # Contrastive loss computation
        pos_loss = torch.sum(pos_distances ** 2 * pos_mask) / (torch.sum(pos_mask) + 1e-8)
        neg_loss = torch.sum(torch.clamp(self.margin - neg_distances, min=0) ** 2 * neg_mask) / (torch.sum(neg_mask) + 1e-8)
        
        return pos_loss + neg_loss


class ScaffoldEncoder(nn.Module):
    """
    Encoder for scaffold SMARTS patterns.
    Converts SMARTS strings to fixed-size embeddings.
    """
    
    def __init__(self, embedding_dim=128, vocab_size=100):
        super(ScaffoldEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Character-level embedding for SMARTS
        self.char_embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, embedding_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        
        # Create character vocabulary for SMARTS
        self.char_to_idx = self._create_vocab()
    
    def _create_vocab(self):
        """Create character vocabulary for SMARTS patterns."""
        chars = ['<PAD>', '<UNK>'] + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]{}=+-#@/%\\')
        return {char: idx for idx, char in enumerate(chars[:self.vocab_size])}
    
    def encode_smarts(self, smarts_list, max_len=100):
        """Convert SMARTS strings to indices."""
        batch_size = len(smarts_list)
        encoded = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, smarts in enumerate(smarts_list):
            for j, char in enumerate(smarts[:max_len]):
                encoded[i, j] = self.char_to_idx.get(char, 1)  # 1 is <UNK>
        
        return encoded
    
    def forward(self, smarts_list):
        """
        Args:
            smarts_list: List of SMARTS strings
        Returns:
            Scaffold embeddings: [batch_size, embedding_dim]
        """
        # Encode SMARTS to indices
        encoded = self.encode_smarts(smarts_list)
        if torch.cuda.is_available():
            encoded = encoded.cuda()
        
        # Embed characters
        embedded = self.char_embedding(encoded)  # [batch_size, max_len, 64]
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embedded)  # [batch_size, max_len, embedding_dim]
        
        # Use last hidden state as scaffold representation
        scaffold_embedding = torch.cat([hidden[0], hidden[1]], dim=1)  # [batch_size, embedding_dim]
        
        return self.fc(scaffold_embedding)
    
# class ScaffoldGraphEncoder(nn.Module):
#     """
#     Encoder for scaffold graphs.
#     Converts scaffold graphs to fixed-size embeddings.
    
#     EDIT THIS
    
#     """
    
#     def __init__(self, embedding_dim=128):
#         super(ScaffoldGraphEncoder, self).__init__()
#         self.embedding_dim = embedding_dim

#         # Graph convolutional layers
#         self.conv1 = GCNConv(10, embedding_dim // 2)
#         self.conv2 = GCNConv(embedding_dim // 2, embedding_dim)
        
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self.conv2(x, edge_index)
#         return x
    
#     def out_shape(self):
#         return self.embedding_dim
    
    


class MaskedReconstructor(nn.Module):
    """
    Masked reconstruction module that takes molecular embeddings and scaffold information
    to perform masked reconstruction tasks.
    """
    
    def __init__(self, input_dim, hidden_dim, scaffold_dim=128):
        super(MaskedReconstructor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaffold_dim = scaffold_dim
        
        # Fusion layers for combining molecular and scaffold embeddings
        self.fusion_fc1 = nn.Linear(input_dim + scaffold_dim, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Reconstruction head
        self.recon_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.recon_fc2 = nn.Linear(hidden_dim, input_dim)
        
        # Masking strategy
        self.mask_ratio = 0.15  # Mask 15% of features
    
    def create_mask(self, x, mask_ratio=None):
        """Create random mask for input features."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        batch_size, feature_dim = x.shape
        num_masked = int(feature_dim * mask_ratio)
        
        mask = torch.zeros_like(x)
        for i in range(batch_size):
            masked_indices = torch.randperm(feature_dim)[:num_masked]
            mask[i, masked_indices] = 1
        
        return mask.bool()
    
    def forward(self, x, scaffold_embedding, mask=None):
        """
        Args:
            x: [batch_size, input_dim] - molecular features
            scaffold_embedding: [batch_size, scaffold_dim] - scaffold embeddings
            mask: [batch_size, input_dim] - optional custom mask
        Returns:
            reconstructed: [batch_size, input_dim] - reconstructed features
            mask: [batch_size, input_dim] - applied mask
        """
        if mask is None:
            mask = self.create_mask(x)
        
        # Apply mask to input
        masked_x = x.clone()
        masked_x[mask] = 0
        
        # Fuse molecular and scaffold information
        fused_input = torch.cat([masked_x, scaffold_embedding], dim=1)
        
        # Fusion layers
        h1 = F.relu(self.fusion_fc1(fused_input))
        h2 = F.relu(self.fusion_fc2(h1))
        
        # Reconstruction
        recon_h = F.relu(self.recon_fc1(h2))
        reconstructed = self.recon_fc2(recon_h)
        
        return reconstructed, mask


class GraphMaskedReconstructor(nn.Module):
    """
    Graph-based masked reconstruction module that works with molecular graph data.
    Masks portions of node features and edge information, then reconstructs using
    graph neural networks and scaffold guidance.
    """
    
    def __init__(self, node_feature_dim, edge_feature_dim=None, hidden_dim=256, scaffold_dim=128, num_gnn_layers=3):
        super(GraphMaskedReconstructor, self).__init__()
        
        try:
            from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool
            self.torch_geometric_available = True
        except ImportError:
            print("Warning: torch_geometric not available. GraphMaskedReconstructor will use fallback implementation.")
            self.torch_geometric_available = False
            
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim or 1
        self.hidden_dim = hidden_dim
        self.scaffold_dim = scaffold_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Masking parameters
        self.node_mask_ratio = 0.15    # Mask 15% of nodes
        self.feature_mask_ratio = 0.10  # Mask 10% of features per node
        
        if self.torch_geometric_available:
            # Graph neural network layers for encoding
            self.node_encoder_layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            # First layer
            self.node_encoder_layers.append(GCNConv(node_feature_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
            
            # Hidden layers
            for _ in range(num_gnn_layers - 2):
                self.node_encoder_layers.append(GCNConv(hidden_dim, hidden_dim))
                self.batch_norms.append(BatchNorm(hidden_dim))
            
            # Final layer
            self.node_encoder_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
            
            # Attention mechanism for scaffold integration
            self.scaffold_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            
            # Graph-level pooling
            self.global_pool = global_mean_pool
            
        else:
            # Fallback to standard neural networks
            self.node_encoder = nn.Sequential(
                nn.Linear(node_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            )
        
        # Scaffold fusion network
        self.scaffold_fusion = nn.Sequential(
            nn.Linear(hidden_dim + scaffold_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node feature reconstruction head
        self.node_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )
        
        # Edge reconstruction head
        if edge_feature_dim > 0:
            self.edge_reconstructor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate source and target node embeddings
                nn.ReLU(),
                nn.Linear(hidden_dim, edge_feature_dim)
            )
        
        # Mask token (learnable parameter)
        self.mask_token = nn.Parameter(torch.randn(node_feature_dim))
        
    def create_graph_masks(self, batch_data):
        """
        Create masks for graph data (nodes and features).
        
        Args:
            batch_data: PyTorch Geometric batch data object
            
        Returns:
            node_mask: Boolean mask for nodes to be masked
            feature_mask: Boolean mask for features to be masked per node
        """
        if not self.torch_geometric_available:
            # Fallback for non-graph data
            batch_size, num_features = batch_data.shape
            num_masked_features = int(num_features * self.feature_mask_ratio)
            feature_mask = torch.zeros_like(batch_data, dtype=torch.bool)
            
            for i in range(batch_size):
                masked_indices = torch.randperm(num_features)[:num_masked_features]
                feature_mask[i, masked_indices] = True
                
            return None, feature_mask
        
        num_nodes = batch_data.x.size(0)
        num_features = batch_data.x.size(1)
        
        # Node-level masking
        num_masked_nodes = int(num_nodes * self.node_mask_ratio)
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch_data.x.device)
        masked_node_indices = torch.randperm(num_nodes, device=batch_data.x.device)[:num_masked_nodes]
        node_mask[masked_node_indices] = True
        
        # Feature-level masking (for non-masked nodes)
        feature_mask = torch.zeros_like(batch_data.x, dtype=torch.bool)
        for i in range(num_nodes):
            if not node_mask[i]:  # Only mask features for non-masked nodes
                num_masked_features = int(num_features * self.feature_mask_ratio)
                if num_masked_features > 0:
                    masked_feature_indices = torch.randperm(num_features)[:num_masked_features]
                    feature_mask[i, masked_feature_indices] = True
        
        return node_mask, feature_mask
    
    def apply_masks(self, batch_data, node_mask, feature_mask):
        """Apply masks to graph data."""
        if not self.torch_geometric_available:
            # Fallback implementation
            masked_data = batch_data.clone()
            masked_data[feature_mask] = 0
            return masked_data
        
        masked_data = batch_data.clone()
        
        # Apply node-level masks (replace entire node features with mask token)
        if node_mask is not None:
            masked_data.x[node_mask] = self.mask_token.unsqueeze(0).expand(node_mask.sum(), -1)
        
        # Apply feature-level masks
        masked_data.x[feature_mask] = 0
        
        return masked_data
    
    def encode_graph(self, batch_data):
        """Encode graph using GNN layers."""
        if not self.torch_geometric_available:
            # Fallback to standard encoding
            return self.node_encoder(batch_data)
        
        x, edge_index = batch_data.x, batch_data.edge_index
        
        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.node_encoder_layers, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.node_encoder_layers) - 1:  # No activation on last layer
                x = F.relu(x)
        
        return x
    
    def integrate_scaffold_information(self, node_embeddings, scaffold_embeddings, batch_data=None):
        """
        Integrate scaffold information with node embeddings using attention.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim] or [batch_size, hidden_dim]
            scaffold_embeddings: [batch_size, scaffold_dim]
            batch_data: PyTorch Geometric batch data (for graph pooling)
        """
        if not self.torch_geometric_available or batch_data is None:
            # Fallback: simple concatenation and fusion
            if len(node_embeddings.shape) == 3:
                # Batch format: [batch_size, num_nodes, hidden_dim]
                batch_size, num_nodes, hidden_dim = node_embeddings.shape
                scaffold_expanded = scaffold_embeddings.unsqueeze(1).expand(-1, num_nodes, -1)
                fused_input = torch.cat([node_embeddings, scaffold_expanded], dim=-1)
                return self.scaffold_fusion(fused_input.view(-1, hidden_dim + self.scaffold_dim)).view(batch_size, num_nodes, -1)
            else:
                # Assume node_embeddings is already pooled: [batch_size, hidden_dim]
                fused_input = torch.cat([node_embeddings, scaffold_embeddings], dim=-1)
                return self.scaffold_fusion(fused_input)
        
        # Pool node embeddings to graph level
        batch_idx = batch_data.batch if hasattr(batch_data, 'batch') else torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
        graph_embeddings = self.global_pool(node_embeddings, batch_idx)  # [batch_size, hidden_dim]
        
        # Apply attention between graph and scaffold embeddings
        graph_embeddings_expanded = graph_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        scaffold_embeddings_expanded = scaffold_embeddings.unsqueeze(1)  # [batch_size, 1, scaffold_dim]
        
        # Project scaffold to same dimension for attention
        scaffold_projected = F.linear(scaffold_embeddings_expanded, 
                                    torch.randn(self.scaffold_dim, self.hidden_dim, device=scaffold_embeddings.device))
        
        # Self-attention between graph and scaffold
        combined_input = torch.cat([graph_embeddings_expanded, scaffold_projected], dim=1)  # [batch_size, 2, hidden_dim]
        attended_output, _ = self.scaffold_attention(combined_input, combined_input, combined_input)
        
        # Take the attended graph representation
        attended_graph = attended_output[:, 0, :]  # [batch_size, hidden_dim]
        
        # Broadcast back to node level if needed
        if len(node_embeddings.shape) > 1 and batch_data is not None:
            # Expand attended graph embedding back to node level
            node_attended = attended_graph[batch_idx]  # [num_nodes, hidden_dim]
            return node_attended
        
        return attended_graph
    
    def forward(self, batch_data, scaffold_embeddings, node_mask=None, feature_mask=None):
        """
        Forward pass for graph masked reconstruction.
        
        Args:
            batch_data: PyTorch Geometric batch data or tensor
            scaffold_embeddings: [batch_size, scaffold_dim] - scaffold embeddings
            node_mask: Optional pre-computed node mask
            feature_mask: Optional pre-computed feature mask
            
        Returns:
            reconstructed_nodes: Reconstructed node features
            reconstructed_edges: Reconstructed edge features (if applicable)
            node_mask: Applied node mask
            feature_mask: Applied feature mask
        """
        # Create masks if not provided
        if node_mask is None or feature_mask is None:
            node_mask, feature_mask = self.create_graph_masks(batch_data)
        
        # Apply masks
        masked_data = self.apply_masks(batch_data, node_mask, feature_mask)
        
        # Encode masked graph
        if self.torch_geometric_available and hasattr(batch_data, 'x'):
            node_embeddings = self.encode_graph(masked_data)
        else:
            # Fallback for tensor input
            node_embeddings = self.encode_graph(masked_data)
        
        # Integrate scaffold information
        scaffold_integrated_embeddings = self.integrate_scaffold_information(
            node_embeddings, scaffold_embeddings, batch_data if self.torch_geometric_available else None
        )
        
        # Reconstruct node features
        reconstructed_nodes = self.node_reconstructor(scaffold_integrated_embeddings)
        
        # Reconstruct edge features if applicable
        reconstructed_edges = None
        if (self.edge_feature_dim > 0 and self.torch_geometric_available and 
            hasattr(batch_data, 'edge_index') and hasattr(batch_data, 'edge_attr')):
            
            edge_index = batch_data.edge_index
            source_embeddings = scaffold_integrated_embeddings[edge_index[0]]
            target_embeddings = scaffold_integrated_embeddings[edge_index[1]]
            edge_input = torch.cat([source_embeddings, target_embeddings], dim=-1)
            reconstructed_edges = self.edge_reconstructor(edge_input)
        
        return {
            'reconstructed_nodes': reconstructed_nodes,
            'reconstructed_edges': reconstructed_edges,
            'node_mask': node_mask,
            'feature_mask': feature_mask,
            'node_embeddings': scaffold_integrated_embeddings
        }
    
    def compute_reconstruction_loss(self, outputs, original_data):
        """
        Compute reconstruction loss for masked portions.
        
        Args:
            outputs: Dictionary with reconstruction outputs
            original_data: Original graph data
            
        Returns:
            loss: Reconstruction loss
        """
        reconstructed_nodes = outputs['reconstructed_nodes']
        node_mask = outputs['node_mask']
        feature_mask = outputs['feature_mask']
        
        if self.torch_geometric_available and hasattr(original_data, 'x'):
            original_nodes = original_data.x
        else:
            original_nodes = original_data
        
        # Node-level reconstruction loss
        node_loss = 0
        if node_mask is not None and node_mask.sum() > 0:
            node_loss = F.mse_loss(reconstructed_nodes[node_mask], original_nodes[node_mask])
        
        # Feature-level reconstruction loss
        feature_loss = 0
        if feature_mask.sum() > 0:
            feature_loss = F.mse_loss(reconstructed_nodes[feature_mask], original_nodes[feature_mask])
        
        # Edge reconstruction loss
        edge_loss = 0
        if (outputs['reconstructed_edges'] is not None and 
            hasattr(original_data, 'edge_attr') and original_data.edge_attr is not None):
            edge_loss = F.mse_loss(outputs['reconstructed_edges'], original_data.edge_attr)
        
        total_loss = node_loss + feature_loss + 0.1 * edge_loss  # Weight edge loss less
        
        return {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'feature_loss': feature_loss,
            'edge_loss': edge_loss
        }


class EnhancedGRASSY(pl.LightningModule):
    """
    Enhanced GRASSY model with contrastive loss, masked reconstruction, and scaffold-constrained generation.
    Maintains all original functionality while adding new capabilities.
    """

    def __init__(self, hparams):
        super(EnhancedGRASSY, self).__init__()
        
        self.hparams = hparams
        self.alpha = hparams.alpha  # Regression loss weight
        self.beta = hparams.beta    # KL divergence weight
        self.gamma = getattr(hparams, 'gamma', 0.1)   # Contrastive loss weight
        self.delta = getattr(hparams, 'delta', 0.1)   # Masked reconstruction loss weight
        self.epsilon = getattr(hparams, 'epsilon', 0.1)  # Scaffold constraint loss weight (NEW)
        
        self.input_dim = hparams.input_dim
        self.bottle_dim = hparams.bottle_dim
        self.hidden_dim = hparams.hidden_dim
        self.scaffold_dim = getattr(hparams, 'scaffold_dim', 128)

        # Original GRASSY components
        self.fc11 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn11 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn12 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc21 = nn.Linear(self.hidden_dim, self.bottle_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.bottle_dim)

        self.fc3 = nn.Linear(self.bottle_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)

        # Property prediction (original)
        self.regfc1 = nn.Linear(self.bottle_dim, 20)
        self.regfc2 = nn.Linear(20, 11)

        # Enhanced components
        self.contrastive_loss = ContrastiveLoss(
            margin=getattr(hparams, 'contrastive_margin', 1.0),
            temperature=getattr(hparams, 'contrastive_temperature', 0.07)
        )
        
        self.scaffold_encoder = ScaffoldEncoder(
            embedding_dim=self.scaffold_dim,
            vocab_size=getattr(hparams, 'vocab_size', 100)
        )
        
        self.masked_reconstructor = MaskedReconstructor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            scaffold_dim=self.scaffold_dim
        )
        
        # NEW: Graph-based masked reconstructor
        self.use_graph_reconstruction = getattr(hparams, 'use_graph_reconstruction', False)
        if self.use_graph_reconstruction:
            self.graph_masked_reconstructor = GraphMaskedReconstructor(
                node_feature_dim=getattr(hparams, 'node_feature_dim', self.input_dim),
                edge_feature_dim=getattr(hparams, 'edge_feature_dim', None),
                hidden_dim=self.hidden_dim,
                scaffold_dim=self.scaffold_dim,
                num_gnn_layers=getattr(hparams, 'num_gnn_layers', 3)
            )
            self.zeta = getattr(hparams, 'zeta', 0.1)  # Graph reconstruction loss weight

        # NEW: Scaffold-constrained generator
        self.scaffold_constrained_generator = ScaffoldConstrainedGenerator(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            scaffold_dim=self.scaffold_dim,
            config_path=getattr(hparams, 'scaffold_config_path', None),
            target_scaffold=getattr(hparams, 'target_scaffold', None),
            constraint_strength=getattr(hparams, 'constraint_strength', 1.0)
        )

        self.loss_list = []
        
        if hparams.n_gpus > 0:
            self.dev_type = 'cuda'
        if hparams.n_gpus == 0:
            self.dev_type = 'cpu'
        
        self.eps = 1e-5

    def kl_div(self, mu, logvar):
        """Compute KL divergence (original)."""
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return KLD
        
    def encode(self, x):
        """Encode input to latent space (original)."""
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick (original)."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """Decode latent representation (original)."""
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def embed(self, x):
        """Get embedding with reparameterization (original)."""
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar 

    def predict(self, z):
        """Property prediction from latent space (original)."""
        h = F.relu(self.regfc1(z))
        y_pred = self.regfc2(h)
        return y_pred
    
    def predict_from_data(self, x):
        """End-to-end property prediction (original)."""
        z = self.embed(x)[0]
        pred = self.predict(z)
        return pred

    def forward(self, x, scaffold_smarts=None, toxicity_labels=None, use_scaffold_generation=False, graph_data=None):
        """
        Enhanced forward pass with optional scaffold and toxicity information.
        
        Args:
            x: Input molecular features
            scaffold_smarts: List of SMARTS strings for scaffolds (optional)
            toxicity_labels: Toxicity labels for contrastive learning (optional)
            use_scaffold_generation: Whether to use scaffold-constrained generation (NEW)
            graph_data: PyTorch Geometric graph data for graph-based reconstruction (optional)
        """
        # Original GRASSY forward pass
        z, mu, logvar = self.embed(x)
        y_pred = self.predict(z)
        x_hat = self.decode(z)
        
        outputs = {
            'x_hat': x_hat,
            'y_pred': y_pred,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
        
        # Enhanced functionality - masked reconstruction
        if scaffold_smarts is not None:
            scaffold_embeddings = self.scaffold_encoder(scaffold_smarts)
            x_recon_masked, mask = self.masked_reconstructor(x, scaffold_embeddings)
            outputs['x_recon_masked'] = x_recon_masked
            outputs['mask'] = mask
            
            # Graph-based masked reconstruction
            if self.use_graph_reconstruction and graph_data is not None:
                graph_recon_outputs = self.graph_masked_reconstructor(
                    graph_data, scaffold_embeddings
                )
                outputs['graph_reconstruction'] = graph_recon_outputs
            outputs['scaffold_embeddings'] = scaffold_embeddings
        
        # NEW: Scaffold-constrained generation
        if use_scaffold_generation:
            scaffold_outputs = self.scaffold_constrained_generator(z, enforce_constraint=True)
            outputs['scaffold_generated'] = scaffold_outputs['generated']
            outputs['scaffold_compliance'] = scaffold_outputs['compliance_scores']
            outputs['scaffold_constraint_mask'] = scaffold_outputs['constraint_mask']
        
        if toxicity_labels is not None:
            outputs['toxicity_labels'] = toxicity_labels
        
        return outputs

    def compute_enhanced_loss(self, outputs, x, y, batch_idx):
        """
        Compute enhanced loss with all components including scaffold-constrained generation.
        
        Args:
            outputs: Forward pass outputs
            x: Input features
            y: Property targets
            batch_idx: Batch index
        """
        # Original GRASSY losses
        recon_loss = nn.MSELoss()(outputs['x_hat'].flatten(), x.flatten()) 
        reg_loss = nn.MSELoss()(outputs['y_pred'], y) 
        KLD = self.kl_div(outputs['mu'], outputs['logvar'])

        # Loss annealing (original)
        num_epochs = self.hparams.n_epochs - 5
        total_batches = self.hparams.len_epoch * num_epochs
        weight = min(1, float(self.trainer.global_step) / float(total_batches))
        kl_loss = weight * KLD
        
        reg_loss = self.alpha * reg_loss.mean()
        kl_loss = self.beta * kl_loss

        # Base loss (original)
        total_loss = recon_loss + reg_loss + kl_loss

        loss_dict = {
            'recon_loss': recon_loss.detach(),
            'pred_loss': reg_loss.detach(),
            'kl_loss': kl_loss.detach()
        }

        # Add contrastive loss if toxicity labels are available
        if 'toxicity_labels' in outputs:
            contrastive_loss = self.contrastive_loss(outputs['z'], outputs['toxicity_labels'])
            contrastive_loss = self.gamma * contrastive_loss
            total_loss += contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.detach()

        # Add masked reconstruction loss if scaffold information is available
        if 'x_recon_masked' in outputs:
            mask = outputs['mask']
            masked_recon_loss = nn.MSELoss()(
                outputs['x_recon_masked'][mask], 
                x[mask]
            )
            masked_recon_loss = self.delta * masked_recon_loss
            total_loss += masked_recon_loss
            loss_dict['masked_recon_loss'] = masked_recon_loss.detach()

        # Add graph masked reconstruction loss if graph data is available
        if 'graph_reconstruction' in outputs and hasattr(self, 'zeta'):
            graph_recon_outputs = outputs['graph_reconstruction']
            # Assume graph_data is passed in outputs for loss computation
            graph_data = outputs.get('original_graph_data', None)
            if graph_data is not None:
                graph_loss_dict = self.graph_masked_reconstructor.compute_reconstruction_loss(
                    graph_recon_outputs, graph_data
                )
                graph_recon_loss = self.zeta * graph_loss_dict['total_loss']
                total_loss += graph_recon_loss
                loss_dict['graph_recon_loss'] = graph_recon_loss.detach()
                loss_dict['graph_node_loss'] = graph_loss_dict['node_loss']
                loss_dict['graph_feature_loss'] = graph_loss_dict['feature_loss']
                loss_dict['graph_edge_loss'] = graph_loss_dict['edge_loss']

        # Add scaffold constraint loss if scaffold generation is used
        if 'scaffold_generated' in outputs:
            scaffold_constraint_loss = self.scaffold_constrained_generator.compute_scaffold_loss(
                outputs['scaffold_generated'], target_compliance=1.0
            )
            scaffold_constraint_loss = self.epsilon * scaffold_constraint_loss
            total_loss += scaffold_constraint_loss
            loss_dict['scaffold_constraint_loss'] = scaffold_constraint_loss.detach()

        loss_dict['train_loss'] = total_loss.detach()
        self.loss_list.append(total_loss.item())

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        """Enhanced training step with new loss components."""
        # Handle different batch formats
        if len(batch) == 2:
            # Original format: (x, y)
            x, y = batch
            x = x.float()
            outputs = self.forward(x)
            loss, log_losses = self.compute_enhanced_loss(outputs, x, y, batch_idx)
        
        elif len(batch) >= 3:
            # Enhanced format: (x, y, toxicity_labels, [smarts], [graph_data], ...)
            x, y = batch[0].float(), batch[1]
            toxicity_labels = batch[2] if len(batch) > 2 else None
            scaffold_smarts = batch[3] if len(batch) > 3 else None
            graph_data = batch[4] if len(batch) > 4 else None
            
            # Enable scaffold generation based on hyperparameters
            use_scaffold_generation = getattr(self.hparams, 'use_scaffold_generation', False)
            
            outputs = self.forward(x, scaffold_smarts, toxicity_labels, use_scaffold_generation, graph_data)
            
            # Add original graph data to outputs for loss computation
            if graph_data is not None:
                outputs['original_graph_data'] = graph_data
            
            loss, log_losses = self.compute_enhanced_loss(outputs, x, y, batch_idx)
        
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
        return {'loss': loss, 'log': log_losses}
   
    def validation_step(self, batch, batch_idx):
        """Enhanced validation step."""
        # Handle different batch formats
        if len(batch) == 2:
            x, y = batch
            x = x.float()
            outputs = self.forward(x)
        elif len(batch) >= 3:
            x, y = batch[0].float(), batch[1]
            toxicity_labels = batch[2] if len(batch) > 2 else None
            scaffold_smarts = batch[3] if len(batch) > 3 else None
            graph_data = batch[4] if len(batch) > 4 else None
            use_scaffold_generation = getattr(self.hparams, 'use_scaffold_generation', False)
            outputs = self.forward(x, scaffold_smarts, toxicity_labels, use_scaffold_generation, graph_data)
            
            # Add original graph data to outputs for loss computation
            if graph_data is not None:
                outputs['original_graph_data'] = graph_data
        
        # Compute losses
        recon_loss = nn.MSELoss()(outputs['x_hat'].flatten(), x.flatten())
        reg_loss = nn.MSELoss()(outputs['y_pred'].reshape(-1), y.reshape(-1)) 
        reg_loss = self.alpha * reg_loss.mean()
        kl_loss = self.kl_div(outputs['mu'], outputs['logvar'])
        kl_loss = self.beta * kl_loss
    
        total_loss = recon_loss + reg_loss + kl_loss

        log_losses = {
            'val_loss': total_loss.detach(), 
            'val_recon_loss': recon_loss.detach(),
            'val_pred_loss': reg_loss.detach(),
            'val_kl_loss': kl_loss.detach()
        }

        # Add enhanced losses if available
        if 'toxicity_labels' in outputs:
            contrastive_loss = self.contrastive_loss(outputs['z'], outputs['toxicity_labels'])
            contrastive_loss = self.gamma * contrastive_loss
            total_loss += contrastive_loss
            log_losses['val_contrastive_loss'] = contrastive_loss.detach()

        if 'x_recon_masked' in outputs:
            mask = outputs['mask']
            masked_recon_loss = nn.MSELoss()(
                outputs['x_recon_masked'][mask], 
                x[mask]
            )
            masked_recon_loss = self.delta * masked_recon_loss
            total_loss += masked_recon_loss
            log_losses['val_masked_recon_loss'] = masked_recon_loss.detach()

        # Add graph masked reconstruction validation loss
        if 'graph_reconstruction' in outputs and hasattr(self, 'zeta'):
            graph_recon_outputs = outputs['graph_reconstruction']
            graph_data = outputs.get('original_graph_data', None)
            if graph_data is not None:
                graph_loss_dict = self.graph_masked_reconstructor.compute_reconstruction_loss(
                    graph_recon_outputs, graph_data
                )
                graph_recon_loss = self.zeta * graph_loss_dict['total_loss']
                total_loss += graph_recon_loss
                log_losses['val_graph_recon_loss'] = graph_recon_loss.detach()

        # Add scaffold constraint validation loss
        if 'scaffold_generated' in outputs:
            scaffold_constraint_loss = self.scaffold_constrained_generator.compute_scaffold_loss(
                outputs['scaffold_generated'], target_compliance=1.0
            )
            scaffold_constraint_loss = self.epsilon * scaffold_constraint_loss
            total_loss += scaffold_constraint_loss
            log_losses['val_scaffold_constraint_loss'] = scaffold_constraint_loss.detach()

        log_losses['val_loss'] = total_loss.detach()
        return log_losses

    def validation_epoch_end(self, outputs):
        """Enhanced validation epoch end."""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_reconloss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        avg_regloss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
        avg_klloss = torch.stack([x['val_kl_loss'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_avg_recon_loss': avg_reconloss,
            'val_avg_pred_loss': avg_regloss,
            'val_avg_kl_loss': avg_klloss
        }

        # Add enhanced loss averages if available
        if 'val_contrastive_loss' in outputs[0]:
            avg_contrastive = torch.stack([x['val_contrastive_loss'] for x in outputs]).mean()
            tensorboard_logs['val_avg_contrastive_loss'] = avg_contrastive

        if 'val_masked_recon_loss' in outputs[0]:
            avg_masked_recon = torch.stack([x['val_masked_recon_loss'] for x in outputs]).mean()
            tensorboard_logs['val_avg_masked_recon_loss'] = avg_masked_recon

        # Add graph reconstruction loss average
        if 'val_graph_recon_loss' in outputs[0]:
            avg_graph_recon = torch.stack([x['val_graph_recon_loss'] for x in outputs]).mean()
            tensorboard_logs['val_avg_graph_recon_loss'] = avg_graph_recon

        # Add scaffold constraint loss average
        if 'val_scaffold_constraint_loss' in outputs[0]:
            avg_scaffold_constraint = torch.stack([x['val_scaffold_constraint_loss'] for x in outputs]).mean()
            tensorboard_logs['val_avg_scaffold_constraint_loss'] = avg_scaffold_constraint

        self.log('val_loss', avg_loss.detach())
        print(f"Validation loss: {avg_loss.detach()}")

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """Configure optimizer (original)."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def get_loss_list(self):
        """Get training loss history (original)."""
        return self.loss_list

    # NEW: Scaffold-constrained generation methods
    def generate_molecules_with_scaffold(self, 
                                       num_molecules: int = 10,
                                       temperature: float = 1.0,
                                       target_scaffold: str = None) -> torch.Tensor:
        """
        Generate molecules that contain a specific scaffold.
        
        Args:
            num_molecules: Number of molecules to generate
            temperature: Sampling temperature for diversity
            target_scaffold: Target scaffold SMARTS (if None, uses current config)
            
        Returns:
            Generated molecular features
        """
        if target_scaffold:
            self.scaffold_constrained_generator.set_target_scaffold(target_scaffold)
        
        # Sample from latent space
        z_samples = torch.randn(num_molecules, self.bottle_dim)
        if self.dev_type == 'cuda':
            z_samples = z_samples.cuda()
        
        # Generate molecules with scaffold constraint
        generated_molecules = self.scaffold_constrained_generator.generate_with_scaffold(
            z_samples, num_samples=1, temperature=temperature
        )
        
        return generated_molecules.squeeze(1)  # Remove num_samples dimension

    def update_target_scaffold(self, scaffold_smarts: str):
        """Update the target scaffold for constrained generation."""
        self.scaffold_constrained_generator.set_target_scaffold(scaffold_smarts)

    def validate_generated_molecules(self, generated_smiles: list, verbose: bool = False):
        """Validate that generated molecules contain the target scaffold."""
        return self.scaffold_constrained_generator.validate_scaffold_constraint(
            generated_smiles, verbose=verbose
        )

    # Additional utility methods for enhanced functionality (existing methods preserved)
    def generate_scaffolds_from_smiles(self, smiles_list):
        """
        Generate scaffold SMARTS from SMILES strings.
        This is a placeholder - in practice, you'd use more sophisticated scaffold extraction.
        """
        scaffolds = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Simple scaffold extraction - you can replace with Murcko scaffolds
                    scaffold = Chem.MolToSmarts(mol)
                    scaffolds.append(scaffold)
                else:
                    scaffolds.append('C')  # Default scaffold
            except:
                scaffolds.append('C')  # Default scaffold
        return scaffolds

    def extract_molecular_embeddings(self, x_batch):
        """Extract molecular embeddings for downstream analysis."""
        with torch.no_grad():
            z, mu, logvar = self.embed(x_batch)
            return z.cpu().numpy(), mu.cpu().numpy(), logvar.cpu().numpy() 