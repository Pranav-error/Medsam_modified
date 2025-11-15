"""
Flower Federated Learning Client
Represents a hospital node that trains MedSAM locally and shares updates
"""

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict
import logging
from pathlib import Path

from fl_model import MedSAM_FL, dice_loss, train_step
from fl_data_loader import get_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedSAMClient(fl.client.NumPyClient):
    """
    Flower client for federated learning with MedSAM.
    Each client represents a hospital with private local data.
    """
    
    def __init__(
        self,
        client_id: int,
        data_dir: str,
        model_checkpoint: str,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """
        Initialize the hospital client.
        
        Args:
            client_id: Unique identifier for this hospital (0, 1, 2, ...)
            data_dir: Path to the data directory
            model_checkpoint: Path to the initial MedSAM checkpoint
            device: Device to use ('cpu', 'cuda', or 'mps')
            batch_size: Batch size for training
        """
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"[Client {client_id}] Initializing hospital client on {device}")
        
        # Load model
        self.model = MedSAM_FL(checkpoint_path=model_checkpoint, device=device)
        
        # Load data loaders for this hospital
        try:
            self.train_loader, self.val_loader = get_data_loaders(
                client_id=client_id,
                data_dir=data_dir,
                batch_size=batch_size
            )
            logger.info(f"[Client {client_id}] Loaded {len(self.train_loader.dataset)} training samples")
            logger.info(f"[Client {client_id}] Loaded {len(self.val_loader.dataset)} validation samples")
        except Exception as e:
            logger.warning(f"[Client {client_id}] Could not load data: {e}")
            logger.warning(f"[Client {client_id}] Using dummy data loaders for testing")
            self.train_loader = None
            self.val_loader = None
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.
        
        Args:
            config: Configuration from server
        
        Returns:
            List of model parameters
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.
        
        Args:
            parameters: List of model parameters from server
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        logger.info(f"[Client {self.client_id}] Starting training round {config.get('server_round', 'N/A')}")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Get training configuration
        local_epochs = config.get("local_epochs", 2)
        learning_rate = config.get("learning_rate", 1e-4)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        num_samples = 0
        total_loss = 0.0
        
        if self.train_loader is None:
            logger.warning(f"[Client {self.client_id}] No training data, returning dummy metrics")
            return self.get_parameters(config={}), 0, {"loss": 0.0, "accuracy": 0.0}
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (images, masks) in enumerate(self.train_loader):
                # Assume full image bounding box for simplicity
                # In practice, you'd extract boxes from annotations
                boxes = torch.tensor([[0, 0, 1024, 1024]] * images.size(0), dtype=torch.float)
                
                loss = train_step(
                    model=self.model,
                    optimizer=optimizer,
                    images=images,
                    masks=masks,
                    boxes=boxes,
                    device=self.device
                )
                
                epoch_loss += loss
                num_samples += images.size(0)
            
            avg_loss = epoch_loss / len(self.train_loader)
            total_loss += avg_loss
            logger.info(f"[Client {self.client_id}] Epoch {epoch + 1}/{local_epochs}, Loss: {avg_loss:.4f}")
        
        # Calculate metrics
        avg_train_loss = total_loss / local_epochs
        
        # Evaluate on validation set
        val_loss, val_dice = self._evaluate()
        
        metrics = {
            "loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "dice": float(val_dice),
            "accuracy": float(val_dice),  # Using Dice score as accuracy metric
        }
        
        logger.info(f"[Client {self.client_id}] Training complete. Metrics: {metrics}")
        
        return self.get_parameters(config={}), num_samples, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration from server
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        logger.info(f"[Client {self.client_id}] Starting evaluation")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Evaluate
        val_loss, val_dice = self._evaluate()
        
        num_samples = len(self.val_loader.dataset) if self.val_loader else 0
        
        metrics = {
            "loss": float(val_loss),
            "dice": float(val_dice),
            "accuracy": float(val_dice),
        }
        
        logger.info(f"[Client {self.client_id}] Evaluation complete. Metrics: {metrics}")
        
        return float(val_loss), num_samples, metrics
    
    def _evaluate(self) -> Tuple[float, float]:
        """
        Internal evaluation method.
        
        Returns:
            Tuple of (loss, dice_score)
        """
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Full image bounding box
                boxes = torch.tensor([[0, 0, 1024, 1024]] * images.size(0), dtype=torch.float, device=self.device)
                
                # Forward pass
                image_embeddings = self.model(images)
                sparse_emb, dense_emb = self.model.get_prompt_embeddings(boxes)
                logits = self.model.decode(image_embeddings, sparse_emb, dense_emb)
                
                # Resize to original size
                pred = torch.nn.functional.interpolate(
                    logits, 
                    size=(1024, 1024), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Calculate loss
                loss = dice_loss(pred, masks)
                
                # Calculate Dice score
                pred_mask = (torch.sigmoid(pred) > 0.5).float()
                dice = 1 - dice_loss(pred_mask, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_dice


def start_client(
    client_id: int,
    server_address: str = "127.0.0.1:8080",
    data_dir: str = "data/breast_cancer",
    model_checkpoint: str = "work_dir/MedSAM/medsam_vit_b.pth",
    device: str = None,
):
    """
    Start a Flower client.
    
    Args:
        client_id: Unique identifier for this hospital
        server_address: Address of the FL server
        data_dir: Path to data directory
        model_checkpoint: Path to initial model checkpoint
        device: Device to use (auto-detected if None)
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    logger.info(f"[Client {client_id}] Using device: {device}")
    
    # Create client
    client = MedSAMClient(
        client_id=client_id,
        data_dir=data_dir,
        model_checkpoint=model_checkpoint,
        device=device,
    )
    
    # Start client
    logger.info(f"[Client {client_id}] Connecting to server at {server_address}")
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flower FL Client (Hospital Node)")
    parser.add_argument("--client-id", type=int, required=True, help="Hospital client ID (0, 1, 2, ...)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--data-dir", type=str, default="data/breast_cancer", help="Data directory")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth", help="Model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    start_client(
        client_id=args.client_id,
        server_address=args.server,
        data_dir=args.data_dir,
        model_checkpoint=args.checkpoint,
        device=args.device,
    )
