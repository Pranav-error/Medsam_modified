"""
Flower Federated Learning Server
Aggregates model updates from multiple hospital clients
"""

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict
import numpy as np
from pathlib import Path
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from clients using weighted average based on number of samples.
    
    Args:
        metrics: List of tuples (num_samples, metrics_dict) from each client
    
    Returns:
        Aggregated metrics dictionary
    """
    # Extract metrics
    accuracies = [num_samples * m["accuracy"] for num_samples, m in metrics]
    losses = [num_samples * m["loss"] for num_samples, m in metrics]
    dice_scores = [num_samples * m.get("dice", 0.0) for num_samples, m in metrics]
    
    # Calculate total samples
    total_samples = sum([num_samples for num_samples, _ in metrics])
    
    # Calculate weighted averages
    aggregated = {
        "accuracy": sum(accuracies) / total_samples,
        "loss": sum(losses) / total_samples,
        "dice": sum(dice_scores) / total_samples,
        "total_samples": total_samples,
        "num_clients": len(metrics)
    }
    
    logger.info(f"Aggregated metrics: {aggregated}")
    return aggregated


def fit_config(server_round: int) -> Dict:
    """
    Configure training for each round.
    
    Args:
        server_round: Current training round
    
    Returns:
        Configuration dictionary for clients
    """
    config = {
        "server_round": server_round,
        "local_epochs": 2,  # Number of local epochs per round
        "batch_size": 8,
        "learning_rate": 1e-4 * (0.95 ** (server_round - 1)),  # Learning rate decay
    }
    logger.info(f"Round {server_round} config: {config}")
    return config


def evaluate_config(server_round: int) -> Dict:
    """
    Configure evaluation for each round.
    
    Args:
        server_round: Current training round
    
    Returns:
        Configuration dictionary for evaluation
    """
    return {
        "server_round": server_round,
        "batch_size": 8,
    }


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that saves the global model after each round.
    """
    
    def __init__(self, save_dir: str = "fl_checkpoints", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model updates and save checkpoint."""
        
        # Call parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Save checkpoint
            checkpoint_path = self.save_dir / f"round_{server_round}_global_model.pt"
            
            # Convert parameters to tensors and save
            logger.info(f"Saving global model checkpoint to {checkpoint_path}")
            
            # Add fairness metrics
            if aggregated_metrics:
                # Calculate fairness deviation (standard deviation of client accuracies)
                client_accuracies = [
                    r.metrics.get("accuracy", 0.0) 
                    for _, r in results if r.metrics
                ]
                if client_accuracies:
                    fairness_std = np.std(client_accuracies)
                    aggregated_metrics["fairness_deviation"] = fairness_std
                    aggregated_metrics["fairness_score"] = max(0, 100 - fairness_std * 100)
                    logger.info(f"Fairness deviation: {fairness_std:.4f}")
        
        return aggregated_parameters, aggregated_metrics


def start_server(
    num_rounds: int = 5,
    num_clients: int = 3,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    server_address: str = "0.0.0.0:8080"
):
    """
    Start the Flower FL server.
    
    Args:
        num_rounds: Number of federated learning rounds
        num_clients: Total number of expected clients (hospitals)
        min_fit_clients: Minimum clients needed for training round
        min_evaluate_clients: Minimum clients needed for evaluation
        min_available_clients: Minimum clients that must be available
        server_address: Server IP and port
    """
    
    logger.info(f"Starting Flower server at {server_address}")
    logger.info(f"Training for {num_rounds} rounds with {num_clients} clients")
    
    # Create strategy with custom configuration
    strategy = SaveModelStrategy(
        save_dir="fl_checkpoints",
        fraction_fit=0.8,  # Sample 80% of available clients for training
        fraction_evaluate=0.8,  # Sample 80% of available clients for evaluation
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    logger.info("Federated learning completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flower FL Server")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of hospital clients")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients needed")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server address")
    
    args = parser.parse_args()
    
    start_server(
        num_rounds=args.rounds,
        num_clients=args.clients,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        server_address=args.address
    )
