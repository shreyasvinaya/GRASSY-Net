import os, datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms

# Import both original and enhanced models
from models.GRASSY_model import GRASSY
from models.GRASSY_enhanced import EnhancedGRASSY

# Import both original and toxicity datasets
from datasets.load_ZINC_tranche import ZINCDataset, Scattering
from datasets.toxicity_dataset import ToxicityDataset, ToxicityScattering

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class ToxicityEnhancedDataLoader:
    """Enhanced data loader that handles both original and toxicity datasets."""
    
    def __init__(self, dataset, batch_size=100, shuffle=True, num_workers=4, include_scaffolds=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.include_scaffolds = include_scaffolds
        
        # Check dataset type
        self.is_toxicity_dataset = isinstance(dataset, ToxicityDataset)
        
        # Create base data loader
        self.loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle, 
            num_workers=num_workers,
            collate_fn=self._enhanced_collate_fn
        )
    
    def _enhanced_collate_fn(self, batch):
        """Enhanced collate function that handles toxicity and scaffold information."""
        if not self.is_toxicity_dataset:
            # Original dataset format
            return torch.utils.data.dataloader.default_collate(batch)
        
        # Enhanced dataset format
        features = []
        properties = []
        toxicity_labels = []
        smiles_list = []
        
        for item in batch:
            if isinstance(item, tuple) and len(item) >= 3:
                # ToxicityScattering output: (features, props, toxicity_label, smiles, ...)
                features.append(item[0])
                properties.append(item[1])
                toxicity_labels.append(item[2])
                smiles_list.append(item[3])
            else:
                # Direct dataset item
                features.append(item[0])
                properties.append(item[1])
                toxicity_labels.append(item.toxicity_label)
                smiles_list.append(item.smiles)
        
        # Stack tensors
        features = torch.stack(features)
        properties = torch.stack(properties)
        toxicity_labels = torch.stack(toxicity_labels)
        
        # Generate scaffolds if requested
        scaffolds = None
        if self.include_scaffolds:
            scaffolds = self._generate_scaffolds(smiles_list)
        
        if scaffolds is not None:
            return features, properties, toxicity_labels, scaffolds
        else:
            return features, properties, toxicity_labels
    
    def _generate_scaffolds(self, smiles_list):
        """Generate Murcko scaffolds from SMILES."""
        scaffolds = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Use Murcko scaffold
                    from rdkit.Chem import Scaffolds
                    scaffold_mol = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smarts = Chem.MolToSmarts(scaffold_mol)
                    scaffolds.append(scaffold_smarts)
                else:
                    scaffolds.append('C')  # Default
            except:
                scaffolds.append('C')  # Default
        return scaffolds
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)


def main():
    parser = ArgumentParser()

    # Original GRASSY parameters
    parser.add_argument('--input_dim', default=None, type=int)
    parser.add_argument('--bottle_dim', default=25, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)

    parser.add_argument('--alpha', default=0.01, type=float, help='Regression loss weight')
    parser.add_argument('--beta', default=0.0005, type=float, help='KL divergence loss weight')
    
    # Enhanced GRASSY parameters
    parser.add_argument('--gamma', default=0.1, type=float, help='Contrastive loss weight')
    parser.add_argument('--delta', default=0.1, type=float, help='Masked reconstruction loss weight')
    parser.add_argument('--epsilon', default=0.1, type=float, help='Scaffold constraint loss weight')
    parser.add_argument('--scaffold_dim', default=128, type=int, help='Scaffold embedding dimension')
    parser.add_argument('--contrastive_margin', default=1.0, type=float)
    parser.add_argument('--contrastive_temperature', default=0.07, type=float)
    parser.add_argument('--vocab_size', default=100, type=int)

    # Scaffold-constrained generation parameters
    parser.add_argument('--use_scaffold_generation', action='store_true',
                       help='Enable scaffold-constrained generation')
    parser.add_argument('--scaffold_config_path', type=str, default=None,
                       help='Path to scaffold configuration file')
    parser.add_argument('--target_scaffold', type=str, default=None,
                       help='Target scaffold SMARTS string')
    parser.add_argument('--constraint_strength', default=1.0, type=float,
                       help='Scaffold constraint strength (0.0-1.0)')

    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--len_epoch', default=None)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--save_dir', default='enhanced_logs/', type=str)

    # Dataset selection
    parser.add_argument('--dataset_type', default='zinc', choices=['zinc', 'toxicity'], 
                       help='Dataset type: zinc (original) or toxicity (enhanced)')
    parser.add_argument('--csv_file', default=None, type=str, 
                       help='Path to toxicity CSV file (required for toxicity dataset)')
    parser.add_argument('--use_enhanced_model', action='store_true', 
                       help='Use enhanced GRASSY model instead of original')
    parser.add_argument('--include_scaffolds', action='store_true', 
                       help='Include scaffold information for masked reconstruction')

    # Original ZINC dataset parameters
    parser.add_argument('--zinc_tranch', default='P14416_BindingDB_train', type=str)
    parser.add_argument('--zinc_tranch_name', default='P14416', type=str)

    parser.add_argument('--GRASSY_version', default='AE+REG', type=str)

    # Add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Configure loss components based on GRASSY version
    if args.GRASSY_version == 'AE+REG':
        kl_div = False
        reg = True
    elif args.GRASSY_version == 'VAE+REG':
        kl_div = True
        reg = True
    elif args.GRASSY_version == 'AE':
        kl_div = False
        reg = False
    elif args.GRASSY_version == 'VAE':
        kl_div = True
        reg = False

    if not kl_div:
        args.beta = 0
    if not reg:
        args.alpha = 0

    # Load dataset based on type
    if args.dataset_type == 'zinc':
        # Original ZINC dataset
        TRANCH = args.zinc_tranch
        TRANCH_NAME = args.zinc_tranch_name
        
        full_dataset = ZINCDataset(
            f'../datasets/{TRANCH}_subset.npy', 
            prop_stat_dict=f'../datasets/{TRANCH}_subset_stats.npy',
            transform=Scattering(scatter_model_name=f'./trained_models/{TRANCH_NAME}.npy')
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Standard data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=15
        )
        valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, num_workers=15
        )
        
        args.input_dim = len(train_set[0][0])
        
    elif args.dataset_type == 'toxicity':
        # Enhanced toxicity dataset
        if args.csv_file is None:
            raise ValueError("csv_file must be specified for toxicity dataset")
        
        # Load toxicity dataset
        scatter_model_name = './trained_models/untrained'  # You can specify a trained scatter model
        transform = ToxicityScattering(scatter_model_name=scatter_model_name)
        
        full_dataset = ToxicityDataset(
            csv_file=args.csv_file,
            transform=transform,
            include_toxicity=True
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Enhanced data loaders
        train_loader = ToxicityEnhancedDataLoader(
            train_set, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4,
            include_scaffolds=args.include_scaffolds
        )
        valid_loader = ToxicityEnhancedDataLoader(
            val_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            include_scaffolds=args.include_scaffolds
        )
        
        # Get input dimension from first sample
        sample_batch = next(iter(train_loader))
        args.input_dim = sample_batch[0].shape[1]
        
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # Set up logging
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    
    model_type = "enhanced" if args.use_enhanced_model else "original"
    dataset_name = args.dataset_type
    
    save_dir = (args.save_dir + 
                f"{dataset_name}_{model_type}/" +
                f"{'regress_' if reg else 'noregress_'}" + 
                f"{'kld_' if kl_div else 'nokld_'}" +
                f"gamma{args.gamma}_delta{args.delta}/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    args.len_epoch = len(train_loader)
    print(f"Input dimension: {args.input_dim}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Model type: {model_type}")
    print(f"Enhanced features - Contrastive: {args.use_enhanced_model}, Scaffolds: {args.include_scaffolds}")
    print(f"Scaffold generation: {args.use_scaffold_generation}")
    if args.use_scaffold_generation:
        print(f"Target scaffold: {args.target_scaffold}")
        print(f"Scaffold config: {args.scaffold_config_path}")
        print(f"Constraint strength: {args.constraint_strength}")

    # Initialize model
    if args.use_enhanced_model:
        model = EnhancedGRASSY(hparams=args)
        features_text = "contrastive loss and masked reconstruction"
        if args.use_scaffold_generation:
            features_text += " and scaffold-constrained generation"
        print(f"Using Enhanced GRASSY model with {features_text}")
    else:
        model = GRASSY(hparams=args)
        print("Using original GRASSY model")

    # Initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.n_epochs,
        gpus=args.n_gpus,
        callbacks=[early_stop_callback],
    )

    # Train model
    print("Starting training...")
    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloaders=valid_loader,
    )

    # Save model and results
    model = model.cpu()
    model.dev_type = 'cpu'

    with torch.no_grad():
        loss_list = model.get_loss_list()

    # Save training loss
    loss_array = np.array(loss_list)
    loss_filename = (f"{dataset_name}_{model_type}_" +
                    f"{'regress' if reg else 'noregress'}_" +
                    f"{'kld' if kl_div else 'nokld'}_" +
                    f"gamma{args.gamma}_delta{args.delta}_loss_list.npy")
    
    np.save(save_dir + loss_filename, loss_array)
    print(f"Saved training loss to {save_dir + loss_filename}")

    # Save model state dict
    model_filename = (f"{dataset_name}_{model_type}_" +
                     f"{'regress' if reg else 'noregress'}_" +
                     f"{'kld' if kl_div else 'nokld'}_" +
                     f"gamma{args.gamma}_delta{args.delta}_model.pth")
    
    torch.save(model.state_dict(), save_dir + model_filename)
    print(f"Saved model to {save_dir + model_filename}")

    # Extract and save embeddings for analysis
    print("Extracting embeddings...")
    embeddings_list = []
    props_list = []
    toxicity_list = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            if args.dataset_type == 'toxicity' and len(batch) >= 3:
                x, y, toxicity_labels = batch[0], batch[1], batch[2]
                x = x.float()
                
                if args.use_enhanced_model:
                    z, mu, logvar = model.embed(x)
                else:
                    z, mu, logvar = model.embed(x)
                
                embeddings_list.append(z.cpu().numpy())
                props_list.append(y.cpu().numpy())
                toxicity_list.append(toxicity_labels.cpu().numpy())
            else:
                x, y = batch[0], batch[1]
                x = x.float()
                
                z, mu, logvar = model.embed(x)
                embeddings_list.append(z.cpu().numpy())
                props_list.append(y.cpu().numpy())

    # Save embeddings and properties
    embeddings = np.vstack(embeddings_list)
    properties = np.vstack(props_list)
    
    embed_filename = (f"{dataset_name}_{model_type}_embeddings_" +
                     f"{'regress' if reg else 'noregress'}_" +
                     f"{'kld' if kl_div else 'nokld'}_" +
                     f"gamma{args.gamma}_delta{args.delta}.npy")
    
    props_filename = (f"{dataset_name}_{model_type}_properties_" +
                     f"{'regress' if reg else 'noregress'}_" +
                     f"{'kld' if kl_div else 'nokld'}_" +
                     f"gamma{args.gamma}_delta{args.delta}.npy")
    
    np.save(save_dir + embed_filename, embeddings)
    np.save(save_dir + props_filename, properties)
    
    if args.dataset_type == 'toxicity' and toxicity_list:
        toxicity_labels = np.hstack(toxicity_list)
        toxicity_filename = (f"{dataset_name}_{model_type}_toxicity_labels_" +
                           f"{'regress' if reg else 'noregress'}_" +
                           f"{'kld' if kl_div else 'nokld'}_" +
                           f"gamma{args.gamma}_delta{args.delta}.npy")
        np.save(save_dir + toxicity_filename, toxicity_labels)
        print(f"Saved toxicity labels to {save_dir + toxicity_filename}")

    print(f"Saved embeddings to {save_dir + embed_filename}")
    print(f"Saved properties to {save_dir + props_filename}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main() 