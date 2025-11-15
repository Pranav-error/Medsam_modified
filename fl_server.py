"""Federated learning server for MedSAM hackathon demo.

This uses Flower (flwr) to coordinate training across multiple hospital
clients. It does **not** change any existing inference or MedSAM code.

Usage (on your laptop, as central server):
    python fl_server.py --rounds 5 --min-clients 2 --address 0.0.0.0:8080

Friends (hospitals) then run fl_client.py and connect to your IP:PORT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl

LOG_DIR = Path("fl_logs")
LOG_DIR.mkdir(exist_ok=True)
METRICS_PATH = LOG_DIR / "server_metrics.json"


def _load_existing_metrics() -> Dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_metrics(data: Dict) -> None:
    METRICS_PATH.write_text(json.dumps(data, indent=2))


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that logs per-round metrics for the UI.

    This writes a lightweight JSON file `fl_logs/server_metrics.json` that can be
    consumed by the frontend (Training/FLED pages) to show real FL progress.
    """

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Collect per-client metrics
        per_client = []
        for client_proxy, eval_res in results:
            metrics = {k: float(v) for k, v in (eval_res.metrics or {}).items()}
            per_client.append(
                {
                    "client_id": client_proxy.cid,
                    "loss": float(eval_res.loss),
                    **metrics,
                }
            )

        # Update JSON log
        state = _load_existing_metrics()
        rounds = state.get("rounds", [])
        rounds.append(
            {
                "round": rnd,
                "aggregated_loss": float(aggregated_loss) if aggregated_loss is not None else None,
                "aggregated_metrics": {k: float(v) for k, v in aggregated_metrics.items()},
                "clients": per_client,
            }
        )

        state.update(
            {
                "last_round": rnd,
                "rounds": rounds,
            }
        )
        _save_metrics(state)

        return aggregated_loss, aggregated_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Learning Server (Flower)")
    parser.add_argument(
        "--address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address host:port (default: 0.0.0.0:8080)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of FL rounds (default: 5)",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of available clients to proceed with training (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Use the same value for all minimum client counts so that small demos
    # with 1 client work without Flower complaining.
    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    print(f"[FL-SERVER] Starting server at {args.address} for {args.rounds} rounds...")
    print("[FL-SERVER] Metrics will be logged to fl_logs/server_metrics.json")

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
