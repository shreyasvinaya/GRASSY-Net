import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import yaml
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Scaffolds
import random
from typing import List, Dict, Optional, Union


class ScaffoldConstrainedGenerator(nn.Module):
    """
    Scaffold-constrained generator that forces molecule generation to contain a specific scaffold.
    
    This module reads a target scaffold from configuration and modifies the generation process
    to ensure all generated molecules contain the specified scaffold structure.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int, 
                 scaffold_dim: int = 128,
                 config_path: Optional[str] = None,
                 target_scaffold: Optional[str] = None,
                 constraint_strength: float = 1.0):
        """
        Args:
            input_dim: Dimension of input molecular features
            hidden_dim: Hidden layer dimension
            scaffold_dim: Dimension of scaffold embeddings
            config_path: Path to configuration file containing target scaffold
            target_scaffold: Direct specification of target scaffold SMARTS
            constraint_strength: Strength of scaffold constraint (0-1, 1=strict enforcement)
        """
        super(ScaffoldConstrainedGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaffold_dim = scaffold_dim
        self.constraint_strength = constraint_strength
        
        # Load target scaffold from config or direct input
        self.target_scaffold = self._load_target_scaffold(config_path, target_scaffold)
        self.target_scaffold_embedding = None
        
        # Scaffold encoder (reuse from enhanced model)
        self.scaffold_encoder = self._create_scaffold_encoder()
        
        # Constraint enforcement networks
        self.constraint_network = nn.Sequential(
            nn.Linear(input_dim + scaffold_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output constraint mask
        )
        
        # Scaffold-guided generation network
        self.generation_network = nn.Sequential(
            nn.Linear(input_dim + scaffold_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Scaffold compliance network (binary classifier)
        self.compliance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize target scaffold embedding
        self._initialize_target_scaffold()
        
    def _load_target_scaffold(self, config_path: Optional[str], direct_scaffold: Optional[str]) -> str:
        """Load target scaffold from config file or direct input."""
        if direct_scaffold:
            return direct_scaffold
            
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                
                return config.get('target_scaffold', 'c1ccccc1')  # Default to benzene
            except Exception as e:
                print(f"Error loading config {config_path}: {e}")
                return 'c1ccccc1'  # Default fallback
        
        # Default scaffold (benzene ring)
        return 'c1ccccc1'
    
    def _create_scaffold_encoder(self):
        """Create scaffold encoder for SMARTS strings."""
        return ScaffoldEncoder(embedding_dim=self.scaffold_dim)
    
    def _initialize_target_scaffold(self):
        """Initialize the target scaffold embedding."""
        if self.target_scaffold:
            with torch.no_grad():
                self.target_scaffold_embedding = self.scaffold_encoder([self.target_scaffold])
                if torch.cuda.is_available():
                    self.target_scaffold_embedding = self.target_scaffold_embedding.cuda()
    
    def set_target_scaffold(self, scaffold_smarts: str):
        """Update the target scaffold during runtime."""
        self.target_scaffold = scaffold_smarts
        self._initialize_target_scaffold()
        print(f"Updated target scaffold to: {scaffold_smarts}")
    
    def forward(self, x: torch.Tensor, enforce_constraint: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate molecules with scaffold constraints.
        
        Args:
            x: Input molecular features [batch_size, input_dim]
            enforce_constraint: Whether to enforce scaffold constraint
            
        Returns:
            Dictionary containing generated molecules and constraint compliance
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Expand target scaffold embedding for batch
        if self.target_scaffold_embedding is None:
            self._initialize_target_scaffold()
        
        target_scaffold_batch = self.target_scaffold_embedding.expand(batch_size, -1).to(device)
        
        # Combine input with target scaffold
        combined_input = torch.cat([x, target_scaffold_batch], dim=1)
        
        # Generate scaffold-constrained molecules
        generated_raw = self.generation_network(combined_input)
        
        if enforce_constraint:
            # Compute constraint mask
            constraint_mask = self.constraint_network(combined_input)
            
            # Apply constraint with specified strength
            generated_constrained = (
                self.constraint_strength * (generated_raw * constraint_mask) + 
                (1 - self.constraint_strength) * generated_raw
            )
        else:
            generated_constrained = generated_raw
            constraint_mask = torch.ones_like(generated_raw)
        
        # Check compliance with target scaffold
        compliance_scores = self.compliance_network(generated_constrained)
        
        return {
            'generated': generated_constrained,
            'generated_raw': generated_raw,
            'constraint_mask': constraint_mask,
            'compliance_scores': compliance_scores,
            'target_scaffold_embedding': target_scaffold_batch
        }
    
    def generate_with_scaffold(self, 
                             latent_codes: torch.Tensor, 
                             num_samples: int = 1,
                             temperature: float = 1.0) -> torch.Tensor:
        """
        Generate molecules from latent codes with scaffold constraint.
        
        Args:
            latent_codes: Latent representations [batch_size, latent_dim]
            num_samples: Number of samples to generate per latent code
            temperature: Sampling temperature for diversity
            
        Returns:
            Generated molecular features with scaffold constraint
        """
        batch_size = latent_codes.shape[0]
        device = latent_codes.device
        
        generated_samples = []
        
        for _ in range(num_samples):
            # Add noise for diversity
            if temperature > 0:
                noise = torch.randn_like(latent_codes) * temperature
                noisy_latent = latent_codes + noise
            else:
                noisy_latent = latent_codes
            
            # Generate with scaffold constraint
            outputs = self.forward(noisy_latent, enforce_constraint=True)
            generated_samples.append(outputs['generated'])
        
        return torch.stack(generated_samples, dim=1)  # [batch_size, num_samples, input_dim]
    
    def compute_scaffold_loss(self, generated: torch.Tensor, target_compliance: float = 1.0) -> torch.Tensor:
        """
        Compute loss to enforce scaffold constraint.
        
        Args:
            generated: Generated molecular features
            target_compliance: Target compliance score (0-1)
            
        Returns:
            Scaffold constraint loss
        """
        compliance_scores = self.compliance_network(generated)
        target_tensor = torch.full_like(compliance_scores, target_compliance)
        
        # Binary cross-entropy loss for compliance
        compliance_loss = F.binary_cross_entropy(compliance_scores, target_tensor)
        
        return compliance_loss
    
    def validate_scaffold_constraint(self, 
                                   generated_smiles: List[str], 
                                   verbose: bool = False) -> Dict[str, float]:
        """
        Validate that generated SMILES contain the target scaffold.
        
        Args:
            generated_smiles: List of generated SMILES strings
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with validation metrics
        """
        if not generated_smiles:
            return {'scaffold_compliance': 0.0, 'valid_molecules': 0.0, 'total_count': 0}
        
        valid_count = 0
        scaffold_compliant_count = 0
        
        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                    
                    # Check if molecule contains target scaffold
                    if self._contains_target_scaffold(mol):
                        scaffold_compliant_count += 1
                        if verbose:
                            print(f"✓ {smiles} - Contains target scaffold")
                    elif verbose:
                        print(f"✗ {smiles} - Missing target scaffold")
                        
            except Exception as e:
                if verbose:
                    print(f"✗ {smiles} - Invalid molecule: {e}")
        
        total_count = len(generated_smiles)
        scaffold_compliance = scaffold_compliant_count / total_count if total_count > 0 else 0.0
        validity = valid_count / total_count if total_count > 0 else 0.0
        
        results = {
            'scaffold_compliance': scaffold_compliance,
            'valid_molecules': validity,
            'scaffold_compliant_count': scaffold_compliant_count,
            'valid_count': valid_count,
            'total_count': total_count
        }
        
        if verbose:
            print(f"\nValidation Results:")
            print(f"Total molecules: {total_count}")
            print(f"Valid molecules: {valid_count} ({validity:.2%})")
            print(f"Scaffold compliant: {scaffold_compliant_count} ({scaffold_compliance:.2%})")
        
        return results
    
    def _contains_target_scaffold(self, mol) -> bool:
        """Check if molecule contains the target scaffold."""
        try:
            # Get Murcko scaffold
            scaffold_mol = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smarts = Chem.MolToSmarts(scaffold_mol)
            
            # Create target scaffold molecule for comparison
            target_mol = Chem.MolFromSmarts(self.target_scaffold)
            
            if target_mol is None:
                return False
            
            # Check if target scaffold is a substructure
            return mol.HasSubstructMatch(target_mol)
            
        except Exception:
            return False
    
    def update_config(self, config_path: str):
        """Update target scaffold from new config file."""
        new_scaffold = self._load_target_scaffold(config_path, None)
        if new_scaffold != self.target_scaffold:
            self.set_target_scaffold(new_scaffold)
    
    def get_current_scaffold(self) -> str:
        """Get the current target scaffold SMARTS."""
        return self.target_scaffold
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        config = {
            'target_scaffold': self.target_scaffold,
            'constraint_strength': self.constraint_strength,
            'scaffold_dim': self.scaffold_dim,
            'hidden_dim': self.hidden_dim
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")


class ScaffoldEncoder(nn.Module):
    """
    Encoder for scaffold SMARTS patterns (reused from enhanced model).
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
        if next(self.parameters()).is_cuda:
            encoded = encoded.cuda()
        
        # Embed characters
        embedded = self.char_embedding(encoded)  # [batch_size, max_len, 64]
        
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embedded)  # [batch_size, max_len, embedding_dim]
        
        # Use last hidden state as scaffold representation
        scaffold_embedding = torch.cat([hidden[0], hidden[1]], dim=1)  # [batch_size, embedding_dim]
        
        return self.fc(scaffold_embedding)


# Utility functions for configuration management
def create_scaffold_config(target_scaffold: str, 
                          config_path: str,
                          constraint_strength: float = 1.0,
                          additional_params: Optional[Dict] = None):
    """
    Create a configuration file for scaffold-constrained generation.
    
    Args:
        target_scaffold: Target scaffold SMARTS string
        config_path: Path to save configuration file
        constraint_strength: Strength of scaffold constraint (0-1)
        additional_params: Additional parameters to include in config
    """
    config = {
        'target_scaffold': target_scaffold,
        'constraint_strength': constraint_strength,
        'generation_params': {
            'temperature': 1.0,
            'num_samples': 10,
            'enforce_constraint': True
        }
    }
    
    if additional_params:
        config.update(additional_params)
    
    # Ensure directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on file extension
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"Scaffold configuration saved to: {config_path}")
    return config_path


def load_scaffold_config(config_path: str) -> Dict:
    """Load scaffold configuration from file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


# Example scaffold configurations
EXAMPLE_SCAFFOLDS = {
    'benzene': 'c1ccccc1',
    'pyridine': 'c1ccncc1', 
    'thiophene': 'c1ccsc1',
    'imidazole': 'c1c[nH]cn1',
    'pyrazole': 'c1c[nH]nn1',
    'furan': 'c1ccoc1',
    'pyrrole': 'c1cc[nH]c1',
    'indole': 'c1ccc2[nH]ccc2c1',
    'quinoline': 'c1ccc2ncccc2c1',
    'biphenyl': 'c1ccc(cc1)c1ccccc1'
} 