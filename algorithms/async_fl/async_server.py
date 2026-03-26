"""Flower server for asynchronous federated learning."""

from __future__ import annotations

import os
import pickle
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union, Any
from time import sleep, time

import numpy as np

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    GetParametersIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.server import Server

from .async_history import AsyncHistory
from .async_client_manager import AsyncClientManager
from .async_strategy import AsynchronousStrategy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class AsyncServer(Server):
    """Flower server implementing asynchronous FL.
    
    This server runs clients concurrently using a ThreadPoolExecutor and
    updates the global model immediately upon each client completion,
    rather than waiting for all clients in a round.
    """

    def __init__(
        self,
        strategy: Strategy,
        client_manager: ClientManager,
        async_strategy: AsynchronousStrategy,
        total_train_time: int = 300,
        waiting_interval: int = 10,
        max_workers: int = 4,
        server_artificial_delay: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the AsyncServer.
        
        Args:
            strategy: The FL strategy (e.g., FedAvg) for client selection/config
            client_manager: Client manager (should be AsyncClientManager)
            async_strategy: The async aggregation strategy
            total_train_time: Total training time in seconds
            waiting_interval: Interval between evaluations in seconds
            max_workers: Maximum concurrent client workers
            server_artificial_delay: Add artificial delay for testing
            config: Additional configuration dictionary
        """
        self.async_strategy = async_strategy
        self.total_train_time = total_train_time
        self.waiting_interval = waiting_interval
        self.strategy = strategy
        self._client_manager = client_manager
        self.max_workers = max_workers
        self.server_artificial_delay = server_artificial_delay
        
        # Track client data percentages for streaming scenarios
        self.client_data_percs: Dict[str, List[float]] = {}
        
        # Per-client prototype embeddings: {client_id: {class_label: np.ndarray}}
        self.client_prototypes: Dict[str, Dict[int, Any]] = {}

        # Per-client binmask tracking
        self.binmask: Dict[str, Any] = {}
        
        # Configuration
        config = config or {}
        self.is_streaming = config.get("is_streaming", False)
        self.client_local_delay = config.get("client_local_delay", False)
        self.data_loading_strategy = config.get("data_loading_strategy", "full")
        self.n_last_samples_for_data_loading_fit = config.get(
            "n_last_samples_for_data_loading_fit", 1000
        )
        self.dataset_seed = config.get("dataset_seed", 42)
        
        # Timing
        self.start_timestamp = 0.0
        self.end_timestamp = 0.0
        self.model_param_lock = Lock()
        
        # Client iteration tracking
        self.client_iters = np.zeros(100)  # Support up to 100 clients
        
        # Client delay configuration
        if self.client_local_delay:
            np.random.seed(self.dataset_seed)
            n_clients_with_delay = 12
            self.clients_with_delay = np.random.choice(
                n_clients_with_delay, n_clients_with_delay, replace=False
            )
            self.delays_per_iter_per_client = np.random.uniform(
                0.0, 5.0, (1000, n_clients_with_delay)
            )
        else:
            self.clients_with_delay = np.array([])
            self.delays_per_iter_per_client = np.array([])

    def set_new_params(self, new_params: Parameters) -> None:
        """Thread-safe update of global parameters."""
        with self.model_param_lock:
            self.parameters = new_params

    def get_current_params(self) -> Parameters:
        """Thread-safe retrieval of global parameters."""
        with self.model_param_lock:
            return self.parameters

    def busy_wait(self, seconds: float) -> None:
        """Busy wait for a number of seconds."""
        start_time = time()
        while time() - start_time < seconds:
            pass

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """Run asynchronous federated learning.
        
        Note: num_rounds is ignored; training runs for total_train_time seconds.
        
        Args:
            num_rounds: Ignored (kept for API compatibility)
            timeout: Timeout for client operations
            
        Returns:
            Tuple of (History, elapsed_time)
        """
        history = AsyncHistory()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
            history.add_loss_centralized_async(timestamp=time(), loss=res[0])
            history.add_metrics_centralized_async(metrics=res[1], timestamp=time())

        # Run federated learning
        log(INFO, "Async FL starting for %s seconds", self.total_train_time)
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        start_time = time()
        end_timestamp = time() + self.total_train_time
        self.end_timestamp = end_timestamp
        self.start_timestamp = time()
        counter = 1
        
        # Start initial fit round
        self.fit_round(
            server_round=0,
            timeout=timeout,
            executor=executor,
            end_timestamp=end_timestamp,
            history=history,
        )

        best_loss = float("inf")
        patience_init = 50
        patience = patience_init

        while time() - start_time < self.total_train_time:
            sleep(self.waiting_interval)
            if self.server_artificial_delay:
                self.busy_wait(10)
            loss = self.evaluate_centralized(counter, history)
            if loss is not None:
                if loss < best_loss - 1e-4:
                    best_loss = loss
                    patience = patience_init
                else:
                    patience -= 1
                if patience == 0:
                    log(INFO, "Early stopping")
                    break
            counter += 1

        executor.shutdown(wait=True, cancel_futures=True)
        log(INFO, "Async FL finished")
        end_time = time()
        self.save_model()
        elapsed = end_time - start_time
        log(INFO, "Async FL finished in %s seconds", elapsed)
        
        # Log async summary
        summary = history.get_async_metrics_summary()
        log(INFO, "Async metrics summary: %s", summary)
        
        return history, elapsed

    def save_model(self) -> None:
        """Save the model to a file."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_path = f"models/model_async_{timestamp}.pkl"
        if not os.path.exists("models"):
            os.makedirs("models")
        with open(model_path, "wb") as f:
            log(DEBUG, "Saving model to %s", model_path)
            pickle.dump(self.parameters, f)
        log(INFO, "Model saved to %s", model_path)

    def evaluate_centralized(
        self, current_round: int, history: AsyncHistory
    ) -> Optional[float]:
        """Evaluate the current model using centralized evaluation."""
        res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            metrics_cen["end_timestamp"] = self.end_timestamp
            metrics_cen["start_timestamp"] = self.start_timestamp
            history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            history.add_metrics_centralized(
                server_round=current_round, metrics=metrics_cen
            )
            history.add_loss_centralized_async(timestamp=time(), loss=loss_cen)
            history.add_metrics_centralized_async(metrics=metrics_cen, timestamp=time())
            log(
                INFO,
                "Centralized evaluation: loss %s, metrics=%s",
                loss_cen,
                {k: v for k, v in metrics_cen.items() if not k.endswith("timestamp")},
            )
            return loss_cen
        return None

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
        executor: ThreadPoolExecutor,
        end_timestamp: float,
        history: AsyncHistory,
    ) -> None:
        """Perform a single round of federated averaging."""
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        for client_proxy, fitins in client_instructions:
            fitins.config = {
                **fitins.config,
                **self.get_config_for_client_fit(client_proxy.cid),
            }

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Submit fit tasks
        fit_clients(
            client_instructions=client_instructions,
            timeout=timeout,
            server=self,
            executor=executor,
            end_timestamp=end_timestamp,
            history=history,
        )

    def get_config_for_client_fit(self, client_id: str, iter: int = 0) -> Dict[str, Any]:
        """Get configuration for client fit call."""
        config: Dict[str, Any] = {}
        
        # Add staleness tracking
        config["start_timestamp"] = time()
        
        if self.client_local_delay and int(client_id) in self.clients_with_delay:
            idx = np.where(self.clients_with_delay == int(client_id))[0]
            if len(idx) > 0:
                config["client_delay"] = float(
                    self.delays_per_iter_per_client[iter % 1000, idx[0]]
                )
            config["cid"] = client_id
            return config

        if not self.is_streaming:
            return config
            
        curr_timestamp = time()
        if curr_timestamp > self.end_timestamp:
            return config
            
        if client_id not in self.client_data_percs:
            self.client_data_percs[client_id] = [0.0]
        prev_data_perc = self.client_data_percs[client_id][-1]
        start_timestamp = self.end_timestamp - self.total_train_time
        data_perc = (
            (time() - start_timestamp) / self.total_train_time
        ) * 0.9 + 0.1
        config["data_percentage"] = data_perc
        config["prev_data_percentage"] = prev_data_perc
        config["data_loading_strategy"] = self.data_loading_strategy
        if self.data_loading_strategy == "fixed_nr":
            config["n_last_samples_for_data_loading_fit"] = (
                self.n_last_samples_for_data_loading_fit
            )
        self.client_data_percs[client_id].append(data_perc)
        return config

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from strategy or one of the available clients."""
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    timeout: Optional[float],
    server: AsyncServer,
    executor: ThreadPoolExecutor,
    end_timestamp: float,
    history: AsyncHistory,
) -> None:
    """Refine parameters concurrently on all selected clients."""
    submitted_fs = {
        executor.submit(fit_client, client_proxy, ins, timeout)
        for client_proxy, ins in client_instructions
    }
    for f in submitted_fs:
        f.add_done_callback(
            lambda ftr: _handle_finished_future_after_fit(
                ftr,
                server=server,
                executor=executor,
                end_timestamp=end_timestamp,
                history=history,
            ),
        )


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,
    server: AsyncServer,
    executor: ThreadPoolExecutor,
    end_timestamp: float,
    history: AsyncHistory,
) -> None:
    """Update the server parameters, restart the client."""
    failure = future.exception()
    if failure is not None:
        log(WARNING, "Got a failure: %s", failure)
        return

    result: Tuple[ClientProxy, FitRes] = future.result()
    clientProxy, res = result

    if res.status.code == Code.OK:
        # Calculate staleness
        start_ts = res.metrics.get("start_timestamp", time())
        t_diff = time() - start_ts
        
        # Aggregate with async strategy
        parameters_aggregated = server.async_strategy.average(
            server.get_current_params(),
            res.parameters,
            t_diff,
            res.num_examples,
        )
        server.set_new_params(parameters_aggregated)

        # Extract per-class prototype embeddings if present
        if "prototypes" in res.metrics:
            client_prototypes = pickle.loads(res.metrics["prototypes"])
            server.client_prototypes[clientProxy.cid] = client_prototypes
            log(DEBUG, "Received %d class prototypes from client %s",
                len(client_prototypes), clientProxy.cid)

        # Extract binmask if present    
        if "binmask" in res.metrics:
            client_binmask = pickle.loads(res.metrics["binmask"])
            server.binmask[clientProxy.cid] = client_binmask
            log(DEBUG, "Received binmask of size %s from client %s",
                len(client_binmask), clientProxy.cid)

        # Log metrics
        metrics = {"sample_sizes": res.num_examples, "t_diff": t_diff, **res.metrics}
        history.add_metrics_distributed_fit_async(
            clientProxy.cid,
            metrics,
            timestamp=time(),
        )
        log(DEBUG, "Updated global model from client %s (staleness: %.2fs)", clientProxy.cid, t_diff)

    # Restart client if still within training time
    if time() < end_timestamp:
        iter_count = int(server.client_iters[int(clientProxy.cid)]) + 1
        server.client_iters[int(clientProxy.cid)] = iter_count
        new_ins = FitIns(
            server.get_current_params(),
            server.get_config_for_client_fit(clientProxy.cid, iter=iter_count),
        )
        ftr = executor.submit(fit_client, client=clientProxy, ins=new_ins, timeout=None)
        ftr.add_done_callback(
            lambda ftr: _handle_finished_future_after_fit(
                ftr, server, executor, end_timestamp, history
            )
        )


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    if res.status.code == Code.OK:
        results.append(result)
        return

    failures.append(result)
