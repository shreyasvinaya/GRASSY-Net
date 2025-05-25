import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl

class GRASSY(pl.LightningModule):
    def __init__(self, hparams):
        super(GRASSY, self).__init__()
        
        self.hparams = hparams
        self.alpha = hparams.alpha
        self.beta = hparams.beta
        self.gamma = hparams.gamma
        self.contrastive_margin = hparams.contrastive_margin
        
        self.input_dim = hparams.input_dim
        self.bottle_dim = hparams.bottle_dim
        self.hidden_dim = hparams.hidden_dim


        self.fc11 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn11 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn12 = nn.BatchNorm1d(self.hidden_dim)
        
        self.fc21 = nn.Linear(self.hidden_dim, self.bottle_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.bottle_dim)

        self.fc3 = nn.Linear(self.bottle_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        # energy prediction
        self.regfc1 = nn.Linear(self.bottle_dim, 20)
        self.regfc2 = nn.Linear(20, 11)

        self.loss_list = []

        
        if hparams.n_gpus > 0:
            self.dev_type = 'cuda'

        if hparams.n_gpus == 0:
            self.dev_type = 'cpu'
        
        self.eps = 1e-5


    def kl_div(self,mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        
        return KLD

    def contrastive_loss(self, anchor_embed, positive_embed, negative_embed, margin):
        cos_sim = nn.CosineSimilarity(dim=1)
        positive_similarity = cos_sim(anchor_embed, positive_embed)
        negative_similarity = cos_sim(anchor_embed, negative_embed)
        loss = torch.clamp(margin - positive_similarity + negative_similarity, min=0.0)
        return loss.mean()
        
    # main model functions
    def encode(self, x):
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def embed(self, x):
        h = self.bn11(F.relu(self.fc11(x)))
        h = self.bn12(F.relu(self.fc12(h)))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar 

    def predict(self,z):
        h = F.relu(self.regfc1(z))
        y_pred = self.regfc2(h)
        return y_pred
    
    def predict_from_data(self,x):
        z = self.embed(x)[0]
        pred = self.predict(z)
        return pred

    def forward(self, x):
        # encoding
        z, mu, logvar = self.embed(x)
        # predict
        y_pred = self.predict(z)
        # recon
        x_hat = self.decode(z)

        return x_hat, y_pred, mu, logvar, z

    def loss_multi_GRASSY(self, 
                        recon_x, x,  
                        mu, logvar,
                        y_pred, y, 
                        anchor_embed, positive_embed, negative_embed,
                        alpha, beta, batch_idx):

        # reconstruction loss
        recon_loss = nn.MSELoss()(recon_x.flatten(), x.flatten()) 
        
        # regression loss
        reg_loss = nn.MSELoss()(y_pred, y) 

        # kl divergence 
        KLD = self.kl_div(mu, logvar)

        # contrastive loss
        contrast_loss = self.contrastive_loss(anchor_embed, positive_embed, negative_embed, self.contrastive_margin)

        num_epochs = self.hparams.n_epochs - 5
        total_batches = self.hparams.len_epoch * num_epochs

        # loss annealing
        weight = min(1, float(self.trainer.global_step) / float(total_batches))
        #reg_loss = weight * reg_loss
        kl_loss = weight* KLD
        
        reg_loss = alpha * reg_loss.mean()

        kl_loss = beta * kl_loss

        contrast_loss = self.gamma * contrast_loss

        total_loss = recon_loss + reg_loss + kl_loss + contrast_loss

        #no kl_loss
        #total_loss = recon_loss + reg_loss

        #no regression
        #total_loss = recon_loss + kl_loss

        #total_loss = recon_loss

        self.loss_list.append(total_loss.item())

        log_losses = {'train_loss' : total_loss.detach(), 
                    'recon_loss' : recon_loss.detach(),
                    'pred_loss' :reg_loss.detach(),
                    'kl_loss': kl_loss.detach(),
                    'contrastive_loss': contrast_loss.detach()
                    }
        
        return total_loss, log_losses

    def get_loss_list(self):
        return self.loss_list

    def training_step(self, batch, batch_idx):
        anchor_features, anchor_properties, positive_features, negative_features = batch
        
        anchor_features = anchor_features.float()
        anchor_properties = anchor_properties.float() # y
        positive_features = positive_features.float()
        negative_features = negative_features.float()

        # Main forward pass for anchor to get reconstruction, prediction, and anchor embedding
        x_hat, y_hat, mu, logvar, z_anchor = self(anchor_features)

        # Generate embeddings for positive and negative samples
        # self.embed returns (z, mu, logvar). We only need z for contrastive loss.
        z_positive, _, _ = self.embed(positive_features)
        z_negative, _, _ = self.embed(negative_features)

        loss, log_losses = self.loss_multi_GRASSY(recon_x=x_hat, x=anchor_features, 
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=anchor_properties,
                                                anchor_embed=z_anchor, 
                                                positive_embed=z_positive,
                                                negative_embed=z_negative,
                                                alpha=self.hparams.alpha, beta=self.hparams.beta,
                                                batch_idx=batch_idx)
            
        return {'loss': loss, 'log': log_losses}
   
    def validation_step(self, batch, batch_idx):
        # Assuming validation batch also provides positive and negative samples for contrastive loss calculation if needed
        # For now, let's assume the validation step primarily focuses on reconstruction, regression, and KL divergence
        # as contrastive loss is more about relative representation learning.
        # If contrastive loss needs to be validated, this step would need similar modifications to training_step.
        
        anchor_features, anchor_properties, positive_features, negative_features = batch
        anchor_features = anchor_features.float()
        anchor_properties = anchor_properties.float() # y
        positive_features = positive_features.float()
        negative_features = negative_features.float()

        # Main forward pass for anchor
        x_hat, y_hat, mu, logvar, z_anchor = self(anchor_features)

        # reconstruction loss
        recon_loss = nn.MSELoss()(x_hat.flatten(), anchor_features.flatten()) # Use anchor_features for recon against x_hat
        
        # regression loss
        reg_loss = nn.MSELoss()(y_hat.reshape(-1), anchor_properties.reshape(-1)) # Use anchor_properties for y
        reg_loss = self.alpha * reg_loss.mean()

        # kl loss
        kl_loss = self.kl_div(mu, logvar)
        kl_loss = self.beta * kl_loss

        # Contrastive loss for validation
        # Generate embeddings for positive and negative samples
        z_positive, _, _ = self.embed(positive_features)
        z_negative, _, _ = self.embed(negative_features)
        
        contrast_loss_val = self.contrastive_loss(z_anchor, z_positive, z_negative, self.contrastive_margin)
        contrast_loss_val = self.gamma * contrast_loss_val
    
        total_loss = recon_loss  +  reg_loss + kl_loss + contrast_loss_val # Added contrastive loss
        #total_loss = recon_loss  +  reg_loss
        #total_loss = recon_loss + kl_loss
        #total_loss = recon_loss

        log_losses = {'val_loss' : total_loss.detach(), 
                    'val_recon_loss' : recon_loss.detach(),
                    'val_pred_loss' :reg_loss.detach(),
                    'val_kl_loss': kl_loss.detach(),
                    'val_contrastive_loss': contrast_loss_val.detach() # Log contrastive loss for validation
                    }

        return log_losses

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_reconloss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        avg_regloss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
        avg_klloss = torch.stack([x['val_kl_loss'] for x in outputs]).mean()
        avg_contrastiveloss = torch.stack([x['val_contrastive_loss'] for x in outputs]).mean() # Average contrastive loss

        tensorboard_logs = {'val_loss': avg_loss,
                            'val_avg_recon_loss': avg_reconloss,
                            'val_avg_pred_loss':avg_regloss,
                            'val_avg_kl_loss':avg_klloss,
                            'val_avg_contrastive_loss': avg_contrastiveloss # Log average contrastive loss
                            }

        self.log('val_loss', avg_loss.detach())
        self.log('val_contrastive_loss', avg_contrastiveloss.detach()) # Log contrastive loss to PyTorch Lightning logger
        print(f"val_loss: {avg_loss.detach()}, val_contrastive_loss: {avg_contrastiveloss.detach()}")


        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

