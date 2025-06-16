#!/usr/bin/env python3
"""
Demonstration script for Enhanced GRASSY model with contrastive loss and masked reconstruction.

This script shows how to:
1. Load a toxicity dataset from CSV
2. Train the enhanced GRASSY model with contrastive loss
3. Use masked reconstruction with scaffold information
4. Analyze the learned representations

Usage:
    python demo_enhanced_grassy.py --csv_file path/to/toxicity_data.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.toxicity_dataset import ToxicityDataset, ToxicityScattering
from models.GRASSY_enhanced import EnhancedGRASSY
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_sample_toxicity_data(filename="sample_toxicity_data.csv", n_samples=1000):
    """Create a sample toxicity dataset for demonstration."""
    print(f"Creating sample toxicity dataset: {filename}")
    
    # Sample SMILES strings (you can replace with real data)
    sample_smiles = [
        "CCO",  # Ethanol
        "CCN(CC)CC",  # Triethylamine
        "CC(C)CC(C)(C)O",  # tert-Amyl alcohol
        "CCC(=O)O",  # Propionic acid
        "C1=CC=CC=C1",  # Benzene
        "CCO",  # Ethanol (duplicate for variety)
        "CCCO",  # 1-Propanol
        "CC(=O)C",  # Acetone
        "CCC",  # Propane
        "C1=CC=C(C=C1)O",  # Phenol
    ]
    
    data = []
    for i in range(n_samples):
        smiles = sample_smiles[i % len(sample_smiles)]
        
        # Generate synthetic molecular properties
        mol_weight = np.random.uniform(50, 500)
        logp = np.random.uniform(-2, 6)
        
        # Assign toxicity based on some rules (for demo purposes)
        toxicity = 1 if "C1=CC=CC=C1" in smiles or "O" in smiles else 0
        toxicity = np.random.choice([0, 1], p=[0.7, 0.3])  # Add some randomness
        
        data.append({
            'CasNo': f"CAS-{i:06d}",
            'SMILES': smiles,
            'Molecular Weight': mol_weight,
            'LogP score': logp,
            'Drug Name': f"Compound_{i}",
            'IUPAC name (if present)': f"compound_{i}_iupac",
            'Toxicity': toxicity
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {len(df)} samples in {filename}")
    return filename


def demo_enhanced_grassy(csv_file, n_epochs=10, batch_size=32):
    """Demonstrate the enhanced GRASSY model."""
    
    print("=" * 60)
    print("Enhanced GRASSY Model Demonstration")
    print("=" * 60)
    
    # 1. Load the toxicity dataset
    print("\n1. Loading toxicity dataset...")
    try:
        # Create the scattering transform (using untrained for demo)
        transform = ToxicityScattering(scatter_model_name='untrained')
        
        # Load dataset
        dataset = ToxicityDataset(
            csv_file=csv_file,
            transform=transform,
            include_toxicity=True
        )
        
        print(f"   Loaded {len(dataset)} valid samples")
        print(f"   Sample toxicity distribution: {np.bincount([dataset.molecular_data[idx]['toxicity_label'] for idx in dataset.valid_indices])}")
        
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        return
    
    # 2. Split dataset
    print("\n2. Splitting dataset...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   Training samples: {len(train_set)}")
    print(f"   Validation samples: {len(val_set)}")
    
    # 3. Create data loaders
    print("\n3. Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1] if isinstance(sample_batch[0], torch.Tensor) else sample_batch[0][0].shape[0]
    
    print(f"   Input dimension: {input_dim}")
    
    # 4. Setup model parameters
    print("\n4. Setting up Enhanced GRASSY model...")
    
    hparams = Namespace(
        input_dim=input_dim,
        bottle_dim=16,
        hidden_dim=64,
        learning_rate=0.001,
        alpha=0.01,      # Regression loss weight
        beta=0.001,      # KL divergence weight
        gamma=0.1,       # Contrastive loss weight
        delta=0.1,       # Masked reconstruction loss weight
        scaffold_dim=64,
        contrastive_margin=1.0,
        contrastive_temperature=0.07,
        vocab_size=100,
        n_epochs=n_epochs,
        len_epoch=len(train_loader),
        n_gpus=0,  # Use CPU for demo
    )
    
    # Initialize enhanced model
    model = EnhancedGRASSY(hparams=hparams)
    print("   Enhanced GRASSY model initialized")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Training loop (simplified)
    print(f"\n5. Training for {n_epochs} epochs...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
    
    train_losses = []
    contrastive_losses = []
    masked_recon_losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_contrastive = 0
        epoch_masked_recon = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Handle batch format
            if len(batch) >= 3:
                # Enhanced format with toxicity
                x = batch[0].float()
                y = batch[1].float()
                toxicity_labels = batch[2].long()
                
                # Generate simple scaffolds for demo
                batch_size = x.shape[0]
                scaffold_smarts = ['C'] * batch_size  # Simple scaffolds for demo
                
                # Forward pass with enhanced features
                outputs = model.forward(x, scaffold_smarts, toxicity_labels)
                
                # Compute enhanced loss
                loss, log_losses = model.compute_enhanced_loss(outputs, x, y, batch_idx)
                
                # Track individual loss components
                if 'contrastive_loss' in log_losses:
                    epoch_contrastive += log_losses['contrastive_loss'].item()
                if 'masked_recon_loss' in log_losses:
                    epoch_masked_recon += log_losses['masked_recon_loss'].item()
                
            else:
                # Standard format
                x, y = batch
                x = x.float()
                y = y.float()
                outputs = model.forward(x)
                loss, log_losses = model.compute_enhanced_loss(outputs, x, y, batch_idx)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Print progress for first epoch
            if epoch == 0 and batch_idx % 5 == 0:
                print(f"   Batch {batch_idx:3d}: Loss = {loss.item():.4f}")
        
        # Average losses
        avg_loss = epoch_loss / n_batches
        avg_contrastive = epoch_contrastive / n_batches if n_batches > 0 else 0
        avg_masked_recon = epoch_masked_recon / n_batches if n_batches > 0 else 0
        
        train_losses.append(avg_loss)
        contrastive_losses.append(avg_contrastive)
        masked_recon_losses.append(avg_masked_recon)
        
        print(f"   Epoch {epoch+1:2d}/{n_epochs}: Loss = {avg_loss:.4f}, "
              f"Contrastive = {avg_contrastive:.4f}, Masked Recon = {avg_masked_recon:.4f}")
    
    # 6. Extract embeddings for analysis
    print("\n6. Extracting molecular embeddings...")
    
    model.eval()
    embeddings = []
    toxicity_labels = []
    smiles_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) >= 3:
                x = batch[0].float()
                toxicity = batch[2].long()
                
                # Extract embeddings
                z, mu, logvar = model.embed(x)
                embeddings.append(z.cpu().numpy())
                toxicity_labels.extend(toxicity.cpu().numpy())
                
                # Get SMILES if available
                if hasattr(batch, 'smiles'):
                    smiles_list.extend(batch.smiles)
    
    if embeddings:
        embeddings = np.vstack(embeddings)
        toxicity_labels = np.array(toxicity_labels)
        
        print(f"   Extracted embeddings: {embeddings.shape}")
        print(f"   Toxicity distribution: {np.bincount(toxicity_labels)}")
        
        # 7. Visualize embeddings
        print("\n7. Visualizing embeddings...")
        
        if embeddings.shape[0] > 10:  # Only if we have enough samples
            try:
                # PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                
                # Plot
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=toxicity_labels, cmap='RdYlBu', alpha=0.7)
                plt.colorbar(scatter, label='Toxicity')
                plt.title('Molecular Embeddings (PCA)')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                
                # Training loss
                plt.subplot(1, 2, 2)
                plt.plot(train_losses, label='Total Loss', linewidth=2)
                if any(contrastive_losses):
                    plt.plot(contrastive_losses, label='Contrastive Loss', alpha=0.7)
                if any(masked_recon_losses):
                    plt.plot(masked_recon_losses, label='Masked Recon Loss', alpha=0.7)
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                output_file = "enhanced_grassy_demo_results.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"   Saved visualization to: {output_file}")
                
                try:
                    plt.show()
                except:
                    print("   (Could not display plot - running headless?)")
                
            except Exception as e:
                print(f"   Could not create visualization: {e}")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully loaded {len(dataset)} molecular samples")
    print(f"✓ Trained Enhanced GRASSY model for {n_epochs} epochs")
    print(f"✓ Final training loss: {train_losses[-1]:.4f}")
    
    if any(contrastive_losses):
        print(f"✓ Contrastive loss (toxicity-based): {contrastive_losses[-1]:.4f}")
    
    if any(masked_recon_losses):
        print(f"✓ Masked reconstruction loss: {masked_recon_losses[-1]:.4f}")
    
    print(f"✓ Extracted embeddings of shape: {embeddings.shape if 'embeddings' in locals() else 'N/A'}")
    print("\nThe Enhanced GRASSY model successfully integrates:")
    print("  • Contrastive learning based on toxicity labels")
    print("  • Masked reconstruction with scaffold information")
    print("  • All original GRASSY functionality (VAE, property prediction)")
    
    return model, embeddings if 'embeddings' in locals() else None, toxicity_labels if 'toxicity_labels' in locals() else None


def main():
    parser = argparse.ArgumentParser(description="Enhanced GRASSY Demo")
    parser.add_argument('--csv_file', type=str, default=None,
                       help='Path to CSV file with toxicity data')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create a sample dataset for demonstration')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create sample data if requested or if no CSV provided
    if args.create_sample or args.csv_file is None:
        csv_file = create_sample_toxicity_data()
    else:
        csv_file = args.csv_file
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        return
    
    # Run demonstration
    try:
        model, embeddings, toxicity_labels = demo_enhanced_grassy(
            csv_file=csv_file,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size
        )
        print("\nDemo completed successfully! 🎉")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 