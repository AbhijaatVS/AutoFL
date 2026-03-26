"""Simulated async client for async federated learning.

This module provides a simulated client that can be used with AsyncServer
for running async FL simulations without real network communication.
"""

from __future__ import annotations

import time
import random
import pickle
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.common import (
    Code,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


@dataclass
class SimulatedClientConfig:
    """Configuration for simulated async client."""
    client_id: str
    model_fn: callable  # Function to create model
    train_loader: DataLoader
    test_loader: DataLoader
    device: torch.device
    local_epochs: int = 2
    learning_rate: float = 0.01
    simulate_delay: bool = True
    min_delay: float = 0.5
    max_delay: float = 3.0


class SimulatedAsyncClient(ClientProxy):
    """Simulated client for async FL that trains locally.
    
    This client simulates local training with configurable delays
    to mimic real-world async behavior.
    """
    
    def __init__(self, config: SimulatedClientConfig):
        super().__init__(config.client_id)
        self.config = config
        self.model = config.model_fn().to(config.device)
        self.train_loader = config.train_loader
        self.test_loader = config.test_loader
        self.device = config.device
        self.local_epochs = config.local_epochs
        self.learning_rate = config.learning_rate
        self._num_examples = len(config.train_loader.dataset)

    def find_last_linear(self) -> torch.nn.Linear:
        """Find the last nn.Linear (classifier head) in the model."""
        last_linear = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        if last_linear is None:
            raise RuntimeError("Model has no nn.Linear layer — cannot extract embeddings")
        return last_linear

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float] = None
    ) -> GetParametersRes:
        """Get current model parameters."""
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params),
        )
    
    def get_properties(self, ins, timeout: Optional[float] = None):
        """Get client properties."""
        from flwr.common import GetPropertiesRes
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={
                "client_id": self.cid,
                "num_examples": self._num_examples,
            },
        )
    
    def fit(
        self, ins: FitIns, timeout: Optional[float] = None
    ) -> FitRes:
        """Train the model on local data."""
        start_time = time.time()
        
        # Simulate network delay (download)
        if self.config.simulate_delay:
            download_delay = random.uniform(
                self.config.min_delay / 2, self.config.max_delay / 2
            )
            time.sleep(download_delay)
        
        # Set parameters from server
        params = parameters_to_ndarrays(ins.parameters)
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), params):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

        # Save the parameters to compare them later for binmasking
        original_params = {key: val.clone() for key, val in state_dict.items()}
        binmask = {key: val.clone() for key, val in state_dict.items()}

        # Train locally
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # Register hook on the classifier head to capture penultimate-layer
        # embeddings during the last epoch (weights are near-final then).
        hook_output: Dict[str, torch.Tensor] = {}
        def hook_fn(module, inp, out):
            hook_output["feat"] = inp[0].detach() #input to the last linear layer is the penultimate layer output
        handle = self.find_last_linear().register_forward_hook(hook_fn)
        
        proto_embeddings: Dict[int, List[torch.Tensor]] = {}  # class -> list of feature vecs
        
        total_loss = 0.0
        num_batches = 0
        for epoch in range(self.local_epochs):
            is_last_epoch = (epoch == self.local_epochs - 1)
            for batch in self.train_loader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("x")).to(self.device)
                    labels = batch.get("label", batch.get("y")).to(self.device)
                elif isinstance(batch, (tuple, list)):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue
                    
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect embeddings only on the last epoch
                if is_last_epoch and "feat" in hook_output:
                    feats = hook_output["feat"]
                    for feat, label in zip(feats, labels):
                        lbl = label.item()
                        proto_embeddings.setdefault(lbl, []).append(feat.cpu())
        
        handle.remove()
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute per-class prototype (mean embedding) and serialize
        prototypes = {
            lbl: torch.stack(vecs).mean(dim=0).numpy() #calculate mean embedding for each class
            for lbl, vecs in proto_embeddings.items()
        }
        prototypes_bytes = pickle.dumps(prototypes)


        #Find binmask for parameters that changed during local training
        for key in original_params.keys():
            if key in state_dict:
                diff = torch.abs(state_dict[key] - original_params[key])
                relative_change = diff / (torch.abs(original_params[key]) + 1e-6)

                # Mark parameters with >30% relative change as 1, else 0
                binmask[key] = (relative_change > 0.3).view(-1).to(torch.uint8).cpu().numpy()
                

        binmask_bytes = pickle.dumps(binmask)

        # Simulate network delay (upload)
        if self.config.simulate_delay:
            upload_delay = random.uniform(
                self.config.min_delay / 2, self.config.max_delay / 2
            )
            time.sleep(upload_delay)
        
        # Get updated parameters
        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        elapsed = time.time() - start_time
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(new_params),
            num_examples=self._num_examples,
            metrics={
                "loss": avg_loss,
                "training_time": elapsed,
                "start_timestamp": ins.config.get("start_timestamp", start_time),
                "client_id": self.cid,
                "prototypes": prototypes_bytes,
                "binmask": binmask_bytes,
            },
        )
    
    def evaluate(
        self, ins: EvaluateIns, timeout: Optional[float] = None
    ) -> EvaluateRes:
        """Evaluate the model on local test data."""
        # Set parameters
        params = parameters_to_ndarrays(ins.parameters)
        state_dict = self.model.state_dict()
        for key, param in zip(state_dict.keys(), params):
            state_dict[key] = torch.tensor(param)
        self.model.load_state_dict(state_dict)
        
        # Evaluate
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(batch, dict):
                    images = batch.get("img", batch.get("x")).to(self.device)
                    labels = batch.get("label", batch.get("y")).to(self.device)
                elif isinstance(batch, (tuple, list)):
                    images, labels = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=avg_loss,
            num_examples=total,
            metrics={"accuracy": accuracy},
        )
    
    def reconnect(self, ins, timeout=None):
        """Handle reconnection request."""
        from flwr.common import DisconnectRes
        return DisconnectRes(reason="")


def create_simulated_clients(
    num_clients: int,
    model_fn: callable,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    device: torch.device,
    local_epochs: int = 2,
    learning_rate: float = 0.01,
    simulate_delay: bool = True,
    min_delay: float = 0.5,
    max_delay: float = 3.0,
) -> List[SimulatedAsyncClient]:
    """Create a list of simulated async clients.
    
    Args:
        num_clients: Number of clients to create
        model_fn: Function that returns a new model instance
        train_loaders: List of training data loaders (one per client)
        test_loaders: List of test data loaders (one per client)
        device: Torch device for training
        local_epochs: Number of local training epochs
        learning_rate: Learning rate for local training
        simulate_delay: Whether to simulate network delays
        min_delay: Minimum simulated delay in seconds
        max_delay: Maximum simulated delay in seconds
    
    Returns:
        List of SimulatedAsyncClient instances
    """
    clients = []
    for i in range(num_clients):
        config = SimulatedClientConfig(
            client_id=str(i),
            model_fn=model_fn,
            train_loader=train_loaders[i % len(train_loaders)],
            test_loader=test_loaders[i % len(test_loaders)],
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            simulate_delay=simulate_delay,
            min_delay=min_delay,
            max_delay=max_delay,
        )
        clients.append(SimulatedAsyncClient(config))
    return clients
