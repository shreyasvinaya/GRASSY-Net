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
    
class ScaffoldGraphEncoder(nn.Module):
    """
    Encoder for scaffold graphs.
    Converts scaffold graphs to fixed-size embeddings.
    
    EDIT THIS
    
    """
    
    def __init__(self, embedding_dim=128):
        super(ScaffoldGraphEncoder, self).__init__()
        self.embedding_dim = embedding_dim

        # Graph convolutional layers
        self.conv1 = GCNConv(10, embedding_dim // 2)
        self.conv2 = GCNConv(embedding_dim // 2, embedding_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x
    
    def out_shape(self):
        return self.embedding_dim
    
    


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

    def forward(self, x, scaffold_smarts=None, toxicity_labels=None, use_scaffold_generation=False):
        """
        Enhanced forward pass with optional scaffold and toxicity information.
        
        Args:
            x: Input molecular features
            scaffold_smarts: List of SMARTS strings for scaffolds (optional)
            toxicity_labels: Toxicity labels for contrastive learning (optional)
            use_scaffold_generation: Whether to use scaffold-constrained generation (NEW)
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

        # NEW: Add scaffold constraint loss if scaffold generation is used
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
            # Enhanced format: (x, y, toxicity_labels, [smarts], ...)
            x, y = batch[0].float(), batch[1]
            toxicity_labels = batch[2] if len(batch) > 2 else None
            scaffold_smarts = batch[3] if len(batch) > 3 else None
            
            # NEW: Enable scaffold generation based on hyperparameters
            use_scaffold_generation = getattr(self.hparams, 'use_scaffold_generation', False)
            
            outputs = self.forward(x, scaffold_smarts, toxicity_labels, use_scaffold_generation)
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
            use_scaffold_generation = getattr(self.hparams, 'use_scaffold_generation', False)
            outputs = self.forward(x, scaffold_smarts, toxicity_labels, use_scaffold_generation)
        
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

        # NEW: Add scaffold constraint validation loss
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

        # NEW: Add scaffold constraint loss average
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