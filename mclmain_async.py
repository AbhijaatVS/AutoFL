"""Async FL main entry point for asynchronous federated learning.

This script provides a time-based async FL training loop where clients
train concurrently and update the global model immediately upon completion,
without waiting for synchronization rounds.

Usage:
    python mclmain_async.py --config-path config/experiments --config-name async_cifar10_gpu
"""

from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
import sys
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional
from time import time, sleep
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import OmegaConf, DictConfig
import wandb

from utils.latency_simulator import init_runtime_recorder, flush_runtime_recorder

# Ignore Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_cfg() -> DictConfig:
    """Load configuration from base and experiment configs."""
    base_config_path = "config/config.yaml"
    base_cfg = OmegaConf.load(base_config_path)
    
    exp_cfg = None
    cfg_name = None
    if "--config-path" in sys.argv and "--config-name" in sys.argv:
        p_idx = sys.argv.index("--config-path") + 1
        n_idx = sys.argv.index("--config-name") + 1
        if p_idx < len(sys.argv) and n_idx < len(sys.argv):
            cfg_path = sys.argv[p_idx]
            cfg_name = sys.argv[n_idx]
            candidate = os.path.join(cfg_path, f"{cfg_name}.yaml")
            if os.path.isfile(candidate):
                exp_cfg = OmegaConf.load(candidate)
                if "defaults" in exp_cfg:
                    del exp_cfg["defaults"]
            else:
                print(f"[Config] File {candidate} not found. Using only base config.")
    
    if exp_cfg is not None:
        cfg = OmegaConf.merge(base_cfg, exp_cfg)
        print(f"[Config] Loaded experiment config: {cfg_name}")
    else:
        cfg = base_cfg
        print(f"[Config] Using base config only")
    return cfg


cfg = load_cfg()
print("Configuration Loaded:\n" + OmegaConf.to_yaml(cfg))

# Validate configuration
from utils.model_factory import validate_config
validate_config(cfg)


def _sanitize_segment(text: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    text = text.lower().replace(" ", "-")
    return "".join(ch if ch in allowed else "-" for ch in text).strip("-")


def prepare_run_directory(cfg: DictConfig) -> Path:
    """Prepare output directory for the run."""
    if "logging" not in cfg or cfg.logging is None:
        cfg.logging = OmegaConf.create({})
    if "output_root" not in cfg.logging or cfg.logging.output_root is None:
        cfg.logging.output_root = "outputs"
    output_root = Path(cfg.logging.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = None
    if "wb" in cfg and cfg.wb is not None:
        run_name = cfg.wb.get("name")
    if not run_name:
        dataset_name = cfg.dataset.workload if "dataset" in cfg else "dataset"
        model_name = cfg.model.name if "model" in cfg else "model"
        run_name = f"{dataset_name}_{model_name}"

    run_name = _sanitize_segment(run_name) or "autofl_async"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    folder_segments = [timestamp, run_name, "async"]
    run_folder = "_".join(seg for seg in folder_segments if seg)
    run_dir = output_root / run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logging.run_output_dir = str(run_dir)
    return run_dir


run_dir = prepare_run_directory(cfg)
init_runtime_recorder(cfg)
atexit.register(flush_runtime_recorder)

# Initialize WandB
if cfg.wb.get('mode', 'online') != 'disabled':
    wandb.init(
        project=cfg.wb.get('project', 'autofl-async'),
        name=cfg.wb.get('name', 'async_run') + '_async',
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wb.get('mode', 'online'),
        tags=['async', cfg.dataset.workload, cfg.model.name],
    )
    wandb_enabled = True
    print(f"[WandB] Initialized: {wandb.run.name}")
else:
    wandb_enabled = False
    print("[WandB] Disabled")

# Save to temp config for other modules
with open("temp_config.yaml", "w") as f:
    OmegaConf.save(cfg, f)


def get_async_config(cfg: DictConfig) -> Dict[str, Any]:
    """Extract async configuration from the config."""
    async_cfg = cfg.get("async", {})
    if isinstance(async_cfg, DictConfig):
        async_cfg = OmegaConf.to_container(async_cfg, resolve=True)
    return {
        "total_train_time": async_cfg.get("total_train_time", 300),
        "waiting_interval": async_cfg.get("waiting_interval", 10),
        "max_workers": async_cfg.get("max_workers", 4),
        "aggregation_strategy": async_cfg.get("aggregation_strategy", "fedasync"),
        "staleness_alpha": async_cfg.get("staleness_alpha", 0.5),
        "fedasync_mixing_alpha": async_cfg.get("fedasync_mixing_alpha", 0.9),
        "fedasync_a": async_cfg.get("fedasync_a", 0.5),
        "use_staleness": async_cfg.get("use_staleness", True),
        "use_sample_weighing": async_cfg.get("use_sample_weighing", True),
        "send_gradients": async_cfg.get("send_gradients", False),
        "server_artificial_delay": async_cfg.get("server_artificial_delay", False),
        "is_streaming": async_cfg.get("is_streaming", False),
        "client_local_delay": async_cfg.get("client_local_delay", False),
        "simulate_delay": async_cfg.get("simulate_delay", True),
        "min_delay": async_cfg.get("min_delay", 0.5),
        "max_delay": async_cfg.get("max_delay", 3.0),
    }


def get_model(cfg: DictConfig) -> torch.nn.Module:
    """Get model based on configuration."""
    from utils.model_factory import create_model
    return create_model(cfg)


def get_data_loaders(
    cfg: DictConfig, num_clients: int
) -> tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create train/test data loaders for each client."""
    from torchvision import datasets, transforms
    
    # Get transforms based on dataset
    if cfg.dataset.workload in ["cifar10", "cifar100"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ])
        if cfg.dataset.workload == "cifar10":
            train_dataset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        else:
            train_dataset = datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform
            )
    elif cfg.dataset.workload in ["mnist", "permuted_mnist"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.workload}")
    
    # Split training data among clients (IID split)
    samples_per_client = len(train_dataset) // num_clients
    client_sizes = [samples_per_client] * num_clients
    # Add remaining samples to last client
    client_sizes[-1] += len(train_dataset) - sum(client_sizes)
    
    client_datasets = random_split(train_dataset, client_sizes)
    
    batch_size = cfg.client.batch_size
    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]
    
    # Each client gets full test set for simplicity
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loaders = [test_loader] * num_clients
    
    return train_loaders, test_loaders, test_loader


def run_async_simulation(
    cfg: DictConfig,
    async_cfg: Dict[str, Any],
    model_fn: callable,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    global_test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Run async FL simulation with simulated clients.
    
    This function simulates true async FL where clients train concurrently
    and updates are aggregated immediately upon client completion.
    """
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    
    from algorithms.async_fl import (
        AsynchronousStrategy,
        AsyncHistory,
        create_simulated_clients,
    )
    
    num_clients = len(train_loaders)
    
    # Create simulated clients
    print(f"\nCreating {num_clients} simulated clients...")
    clients = create_simulated_clients(
        num_clients=num_clients,
        model_fn=model_fn,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        device=device,
        local_epochs=cfg.client.local_epochs,
        learning_rate=cfg.client.learning_rate,
        simulate_delay=async_cfg["simulate_delay"],
        min_delay=async_cfg["min_delay"],
        max_delay=async_cfg["max_delay"],
    )
    
    # Initialize global model
    global_model = model_fn().to(device)
    global_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    
    # Create async strategy
    total_samples = sum(len(loader.dataset) for loader in train_loaders)
    async_strategy = AsynchronousStrategy(
        total_samples=total_samples,
        staleness_alpha=async_cfg["staleness_alpha"],
        fedasync_mixing_alpha=async_cfg["fedasync_mixing_alpha"],
        fedasync_a=async_cfg["fedasync_a"],
        num_clients=num_clients,
        async_aggregation_strategy=async_cfg["aggregation_strategy"],
        use_staleness=async_cfg["use_staleness"],
        use_sample_weighing=async_cfg["use_sample_weighing"],
    )
    
    # History tracking
    history = AsyncHistory()
    
    # Synchronization
    param_lock = Lock()
    current_params = ndarrays_to_parameters(global_params)
    
    # Training state
    total_train_time = async_cfg["total_train_time"]
    waiting_interval = async_cfg["waiting_interval"]
    max_workers = async_cfg["max_workers"]
    
    print(f"\n" + "=" * 60)
    print(f"Starting Async FL Simulation")
    print(f"  - Clients: {num_clients}")
    print(f"  - Total train time: {total_train_time}s")
    print(f"  - Max concurrent workers: {max_workers}")
    print(f"  - Aggregation: {async_cfg['aggregation_strategy']}")
    print("=" * 60 + "\n")
    
    # Evaluate initial model
    initial_loss, initial_acc = evaluate_global_model(
        global_model, global_params, global_test_loader, device
    )
    print(f"[t=0.0s] Initial - Loss: {initial_loss:.4f}, Accuracy: {initial_acc:.4f}")
    history.add_loss_centralized_async(timestamp=0.0, loss=initial_loss)
    history.add_metrics_centralized_async(metrics={"accuracy": initial_acc}, timestamp=0.0)
    
    # Log initial metrics to WandB
    if wandb_enabled:
        wandb.log({
            "async/loss": initial_loss,
            "async/accuracy": initial_acc,
            "async/updates": 0,
            "async/elapsed_time": 0.0,
        }, step=0)
    
    start_time = time()
    end_time = start_time + total_train_time
    update_count = 0
    prototypes_recv_count: Dict[str, int] = {}
    binmask_recv_count: Dict[str, int] = {}
    
    def train_client(client_idx: int) -> tuple:
        """Train a single client and return results."""
        from flwr.common import FitIns
        
        client = clients[client_idx]
        with param_lock:
            params = current_params
        
        config = {"start_timestamp": time()}
        fit_ins = FitIns(parameters=params, config=config)
        fit_res = client.fit(fit_ins)
        
        return client_idx, fit_res
    
    def aggregate_result(client_idx: int, fit_res):
        """Aggregate a single client result into global model."""
        nonlocal current_params, update_count
        
        t_diff = time() - fit_res.metrics.get("start_timestamp", time())
        
        with param_lock:
            new_params = async_strategy.average(
                current_params,
                fit_res.parameters,
                t_diff,
                fit_res.num_examples,
            )
            current_params = new_params
            update_count += 1

        cid = str(client_idx)
        proto_count = prototypes_recv_count.get(cid, 0)
        binmask_count = binmask_recv_count.get(cid, 0)

        if "prototypes" in fit_res.metrics:
            try:
                client_prototypes = pickle.loads(fit_res.metrics["prototypes"])
                if isinstance(client_prototypes, dict):
                    proto_count += 1
                    prototypes_recv_count[cid] = proto_count
            except Exception as exc:
                print(f"[WARN] Failed to decode prototypes from client {cid}: {exc}")

        if "binmask" in fit_res.metrics:
            try:
                client_binmask = pickle.loads(fit_res.metrics["binmask"])
                if isinstance(client_binmask, dict):
                    binmask_count += 1
                    binmask_recv_count[cid] = binmask_count
            except Exception as exc:
                print(f"[WARN] Failed to decode binmask from client {cid}: {exc}")
        
        elapsed = time() - start_time
        history.add_metrics_distributed_fit_async(
            client_id=str(client_idx),
            metrics={
                "loss": fit_res.metrics.get("loss", 0),
                "staleness": t_diff,
                "samples": fit_res.num_examples,
                "proto_recv_count": proto_count,
                "binmask_recv_count": binmask_count,
            },
            timestamp=elapsed,
        )

        print(
            f"[recv] client {cid}: prototypes_received={proto_count}, "
            f"binmask_received={binmask_count}"
        )
        
        return t_diff
    
    # Run async training
    executor = ThreadPoolExecutor(max_workers=max_workers)
    active_futures = set()
    client_queue = list(range(num_clients))
    eval_counter = 0
    last_eval_time = start_time
    
    # Submit initial batch of clients
    for _ in range(min(max_workers, num_clients)):
        if client_queue:
            client_idx = client_queue.pop(0)
            future = executor.submit(train_client, client_idx)
            active_futures.add((future, client_idx))
    
    while time() < end_time and (active_futures or client_queue):
        # Check for completed futures
        completed = []
        for future, client_idx in list(active_futures):
            if future.done():
                completed.append((future, client_idx))
        
        for future, client_idx in completed:
            active_futures.discard((future, client_idx))
            try:
                _, fit_res = future.result()
                t_diff = aggregate_result(client_idx, fit_res)
                elapsed = time() - start_time
                print(f"[t={elapsed:.1f}s] Client {client_idx} completed "
                      f"(staleness: {t_diff:.2f}s, loss: {fit_res.metrics.get('loss', 0):.4f})")
            except Exception as e:
                print(f"[WARNING] Client {client_idx} failed: {e}")
            
            # Submit next client if within time limit
            if time() < end_time:
                # Cycle through clients
                next_client = client_idx  # Re-use same client
                future = executor.submit(train_client, next_client)
                active_futures.add((future, next_client))
        
        # Periodic evaluation
        if time() - last_eval_time >= waiting_interval:
            eval_counter += 1
            with param_lock:
                eval_params = parameters_to_ndarrays(current_params)
            
            loss, acc = evaluate_global_model(
                global_model, eval_params, global_test_loader, device
            )
            elapsed = time() - start_time
            print(f"\n[t={elapsed:.1f}s] Evaluation {eval_counter}: "
                  f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, Updates: {update_count}\n")
            
            history.add_loss_centralized_async(timestamp=elapsed, loss=loss)
            history.add_metrics_centralized_async(
                metrics={"accuracy": acc, "updates": update_count}, timestamp=elapsed
            )
            
            # Log to WandB
            if wandb_enabled:
                wandb.log({
                    "async/loss": loss,
                    "async/accuracy": acc,
                    "async/updates": update_count,
                    "async/elapsed_time": elapsed,
                }, step=eval_counter)
            
            last_eval_time = time()
        
        sleep(0.1)  # Small sleep to prevent busy-waiting
    
    executor.shutdown(wait=True)
    
    # Final evaluation
    with param_lock:
        final_params = parameters_to_ndarrays(current_params)
    
    final_loss, final_acc = evaluate_global_model(
        global_model, final_params, global_test_loader, device
    )
    elapsed = time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"Async FL Simulation Complete")
    print(f"  - Total time: {elapsed:.1f}s")
    print(f"  - Total updates: {update_count}")
    print(f"  - Final Loss: {final_loss:.4f}")
    print(f"  - Final Accuracy: {final_acc:.4f}")
    print("  - Receive counts by client:")
    for idx in range(num_clients):
        cid = str(idx)
        print(
            f"      client {cid}: "
            f"prototypes={prototypes_recv_count.get(cid, 0)}, "
            f"binmask={binmask_recv_count.get(cid, 0)}"
        )
    print("=" * 60)
    
    history.add_loss_centralized_async(timestamp=elapsed, loss=final_loss)
    history.add_metrics_centralized_async(
        metrics={"accuracy": final_acc, "updates": update_count}, timestamp=elapsed
    )
    
    # Log final metrics to WandB
    if wandb_enabled:
        wandb.log({
            "async/final_loss": final_loss,
            "async/final_accuracy": final_acc,
            "async/total_updates": update_count,
            "async/total_time": elapsed,
        })
        wandb.summary["final_loss"] = final_loss
        wandb.summary["final_accuracy"] = final_acc
        wandb.summary["total_updates"] = update_count
        wandb.summary["total_time"] = elapsed
    
    # Summary
    summary = history.get_async_metrics_summary()
    
    return {
        "history": history,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "total_updates": update_count,
        "elapsed_time": elapsed,
        "summary": summary,
    }


def evaluate_global_model(
    model: torch.nn.Module,
    params: List[np.ndarray],
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model with given parameters."""
    # Load parameters
    state_dict = model.state_dict()
    for key, param in zip(state_dict.keys(), params):
        state_dict[key] = torch.tensor(param).to(device)
    model.load_state_dict(state_dict)
    
    # Evaluate
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("x")).to(device)
                labels = batch.get("label", batch.get("y")).to(device)
            elif isinstance(batch, (tuple, list)):
                images, labels = batch[0].to(device), batch[1].to(device)
            else:
                continue
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def main() -> None:
    """Run async federated learning simulation."""
    print("=" * 60)
    print("Async Federated Learning Simulation")
    print("=" * 60)
    
    # Get async configuration
    async_cfg = get_async_config(cfg)
    num_clients = cfg.server.num_clients
    
    # Device setup
    if cfg.client.num_gpus > 0.0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model factory
    def model_fn():
        from utils.model_factory import create_model
        return create_model(cfg)
    
    # Load data
    print(f"\nLoading {cfg.dataset.workload} dataset...")
    train_loaders, test_loaders, global_test_loader = get_data_loaders(
        cfg, num_clients
    )
    print(f"Created {len(train_loaders)} client data partitions")
    
    # Run simulation
    results = run_async_simulation(
        cfg=cfg,
        async_cfg=async_cfg,
        model_fn=model_fn,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        global_test_loader=global_test_loader,
        device=device,
    )
    
    # Save results
    results_path = run_dir / "async_results.yaml"
    with open(results_path, "w") as f:
        OmegaConf.save(OmegaConf.create({
            "final_loss": float(results["final_loss"]),
            "final_accuracy": float(results["final_accuracy"]),
            "total_updates": int(results["total_updates"]),
            "elapsed_time": float(results["elapsed_time"]),
        }), f)
    print(f"\nResults saved to: {results_path}")
    
    flush_runtime_recorder()
    
    # Finish WandB
    if wandb_enabled:
        wandb.finish()
    
    # Clean up temp config
    if os.path.exists("temp_config.yaml"):
        os.remove("temp_config.yaml")


if __name__ == "__main__":
    main()
