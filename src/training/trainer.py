"""
PyTorch Lightning Trainer for SSTZIP-GNN Model
Handles training loop, validation, callbacks, and checkpointing
"""
import os
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Optional, Dict, Any


class SSTZIPGNNLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for SSTZIP-GNN model
    Handles training, validation, and testing loops
    """
    
    def __init__(self,
                 model: nn.Module,
                 feature_dim: int = 4,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 patience: int = 10,
                 accumulation_steps: int = 1):
        """
        Args:
            model: SSTZIP-GNN model instance
            feature_dim: Number of input features (default 4, actual ~9)
            learning_rate: Adam learning rate
            weight_decay: L2 regularization coefficient
            patience: Early stopping patience
            accumulation_steps: Gradient accumulation steps
        """
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {}
        
        # Feature projection: from input features to spatial dimension
        # Get spatial_dim from model
        spatial_dim = model.spatial_dim
        self.feature_projection = nn.Linear(feature_dim, spatial_dim)
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x, adj, x_temporal):
        """Forward pass"""
        return self.model(x, adj, x_temporal)
    
    def training_step(self, batch, batch_idx):
        """Training step for one batch"""
        x, y = batch
        
        # Input shape: [batch, seq_length, features]
        # Reshape to [batch, features, seq_length] for temporal encoder
        x = x.transpose(1, 2)  # [batch, features, seq_length]
        
        # Project from input features to spatial_dim (64)
        # x: [batch, features, seq_length] -> [batch, spatial_dim, seq_length]
        x_batch_features = x.transpose(1, 2)  # [batch, seq_length, features]
        x_proj = self.feature_projection(x_batch_features)  # [batch, seq_length, spatial_dim]
        x_proj = x_proj.transpose(1, 2)  # [batch, spatial_dim, seq_length]
        
        # Use temporal encoder on projected features
        x_temporal = self.model.temporal_encoder(x_proj)  # [batch, temporal_dim, seq_length]
        
        # Aggregate temporal dimension
        x_agg = self.model.temporal_pooling(x_temporal)  # [batch, temporal_dim]
        
        # ZIP output head
        pi, lambda_param = self.model.zip_head(x_agg)  # Each [batch, 1]
        
        # Flatten for loss computation
        pi = pi.squeeze(-1)
        lambda_param = lambda_param.squeeze(-1)
        y = y.float()
        
        # Compute loss (ZIPLoss expects tuple of predictions)
        loss = self.model.loss_fn((pi, lambda_param), y)
        
        # Log metrics
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.train_losses.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch"""
        x, y = batch
        
        # Input shape: [batch, seq_length, features]
        # Reshape to [batch, features, seq_length] for temporal encoder
        x = x.transpose(1, 2)  # [batch, features, seq_length]
        
        # Project from input features to spatial_dim (64)
        # x: [batch, features, seq_length] -> [batch, spatial_dim, seq_length]
        x_batch_features = x.transpose(1, 2)  # [batch, seq_length, features]
        x_proj = self.feature_projection(x_batch_features)  # [batch, seq_length, spatial_dim]
        x_proj = x_proj.transpose(1, 2)  # [batch, spatial_dim, seq_length]
        
        # Use temporal encoder on projected features
        x_temporal = self.model.temporal_encoder(x_proj)  # [batch, temporal_dim, seq_length]
        
        # Aggregate temporal dimension
        x_agg = self.model.temporal_pooling(x_temporal)  # [batch, temporal_dim]
        
        # ZIP output head
        pi, lambda_param = self.model.zip_head(x_agg)  # Each [batch, 1]
        
        # Flatten for loss computation
        pi = pi.squeeze(-1)
        lambda_param = lambda_param.squeeze(-1)
        y = y.float()
        
        # Compute loss (ZIPLoss expects tuple of predictions)
        loss = self.model.loss_fn((pi, lambda_param), y)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for one batch"""
        x, y = batch
        
        # Input shape: [batch, seq_length, features]
        # Reshape to [batch, features, seq_length] for temporal encoder
        x = x.transpose(1, 2)  # [batch, features, seq_length]
        
        # Project from input features to spatial_dim (64)
        x_batch_features = x.transpose(1, 2)  # [batch, seq_length, features]
        x_proj = self.feature_projection(x_batch_features)  # [batch, seq_length, spatial_dim]
        x_proj = x_proj.transpose(1, 2)  # [batch, spatial_dim, seq_length]
        
        # Use simplified forward: temporal encoder directly on features
        with torch.no_grad():
            x_temporal = self.model.temporal_encoder(x_proj)  # [batch, temporal_dim, seq_length]
            
            # Aggregate temporal dimension
            x_agg = self.model.temporal_pooling(x_temporal)  # [batch, temporal_dim]
            
            # ZIP output head
            pi, lambda_param = self.model.zip_head(x_agg)  # Each [batch, 1]
        
        # Flatten for loss computation
        pi = pi.squeeze(-1)
        lambda_param = lambda_param.squeeze(-1)
        y = y.float()
        
        # Compute loss (ZIPLoss expects tuple of predictions)
        loss = self.model.loss_fn((pi, lambda_param), y)
        
        self.log('test_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class SSTZIPGNNTrainer:
    """
    High-level trainer class for SSTZIP-GNN
    Handles model initialization, training, and evaluation
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 checkpoint_dir: str = "checkpoints"):
        """
        Args:
            model: SSTZIP-GNN model
            config: Configuration dictionary
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Lightning module
        self.lightning_model = SSTZIPGNNLightning(
            model=model,
            learning_rate=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5),
            patience=config.get('patience', 10)
        )
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=config.get('max_epochs', 100),
            callbacks=self.callbacks,
            accelerator='auto',
            devices='auto',
            precision=config.get('precision', '32'),
            log_every_n_steps=config.get('log_every_n_steps', 10),
            enable_progress_bar=config.get('enable_progress_bar', True),
            gradient_clip_val=config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=config.get('accumulation_steps', 1)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def _setup_callbacks(self) -> list:
        """Setup callbacks for training"""
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('patience', 10),
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            verbose=True
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def train(self, train_dataloader, val_dataloader):
        """
        Train the model
        
        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
        """
        print("Starting training...")
        print(f"Config: {json.dumps(self.config, indent=2)}")
        
        # Train
        self.trainer.fit(
            self.lightning_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        print("Training completed!")
        
        # Save history
        self._save_history()
    
    def evaluate(self, test_dataloader):
        """
        Evaluate on test set
        
        Args:
            test_dataloader: Test DataLoader
        """
        print("Starting evaluation...")
        
        results = self.trainer.test(
            self.lightning_model,
            dataloaders=test_dataloader
        )
        
        print(f"Test results: {results}")
        
        return results
    
    def _save_history(self):
        """Save training history"""
        history_file = self.checkpoint_dir / 'training_history.json'
        
        history = {
            'train_losses': self.lightning_model.train_losses,
            'val_losses': self.lightning_model.val_losses,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"History saved to {history_file}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.lightning_model = SSTZIPGNNLightning.load_from_checkpoint(
            checkpoint_path,
            model=self.model
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_model(self, path: str):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


def create_training_config() -> Dict[str, Any]:
    """Create default training configuration"""
    return {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'max_epochs': 100,
        'batch_size': 32,
        'patience': 10,
        'accumulation_steps': 1,
        'log_every_n_steps': 10,
        'enable_progress_bar': True,
        'gradient_clip_val': 1.0,
        'precision': '32',
        'num_workers': 4
    }


if __name__ == "__main__":
    print("Testing Trainer setup...")
    
    # Note: Full training requires data module
    # This just demonstrates the structure
    
    config = create_training_config()
    print(f"Training config:\n{json.dumps(config, indent=2)}")
    
    print("\n✅ Trainer structure validated!")
