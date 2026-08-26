#!/usr/bin/env python3
"""
AEGIS Public SDK - PyTorch Distributed Data Parallel (DDP) Integration Example

This example demonstrates how to integrate the AEGIS Trial SDK into a multi-GPU
DistributedDataParallel (DDP) training loop.

To run this script across multiple local GPUs (e.g., using torchrun):
    torchrun --nproc_per_node=2 examples/ddp_integration.py

Install the public SDK and the separately supplied compatible Trial Core wheel into
the same CPython 3.11 environment:
    python3.11 -m pip install /path/to/aegis_trial_core-0.1.1-cp311-cp311-linux_x86_64.whl
    python3.11 -m pip install aegis-trial-sdk

This is an API integration example. It does not establish that a particular
distributed training configuration is validated for the Trial Core.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from aegis import AegisMonitor, AegisEvent
    from aegis.integrations import gather_events
except ImportError as e:
    print(f"[AEGIS Error] Failed to import AEGIS: {e}", file=sys.stderr)
    print("[AEGIS Error] Please ensure you are running this from the correct environment "
          "and the aegis-trial-core is installed locally.", file=sys.stderr)
    sys.exit(1)


# 1. Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        # A representative hidden layer for local observation.
        self.mlp_dense = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu1(h)
        # Representative forward block
        h = self.mlp_dense(h)
        h = self.relu2(h)
        return self.fc2(h)


def setup_ddp():
    """Initialize standard PyTorch DDP distributed process group."""
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy the process group clean up."""
    dist.destroy_process_group()


def train_ddp():
    # Initialize process group and determine device rank
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}/{world_size}] Using device: {device}")

    # Generate synthetic training dataset
    x_train = torch.randn(1000, 128)
    y_train = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x_train, y_train)

    # Use DistributedSampler to partition data across ranks
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Instantiate model and wrap with DistributedDataParallel
    model = SimpleMLP().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # This optimizer belongs entirely to the user's ordinary training loop.
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 2. Instantiate AegisMonitor
    # To avoid writing conflicting files, write logs from Rank 0 only.
    log_file = "aegis_ddp_demo_log.jsonl" if rank == 0 else None

    if rank == 0:
        print(f"\n[Rank 0] Initializing AegisMonitor. Logs will be saved to: {log_file}")

    try:
        aegis = AegisMonitor(
            model=ddp_model, expected_microbatches=1, log_path=log_file
        )
    except Exception as e:
        if rank == 0:
            print(f"\n[AEGIS Initialization Failed on Rank 0] {e}", file=sys.stderr)
            print("\nNote: Install the separately supplied compatible aegis-trial-core wheel.", file=sys.stderr)
        cleanup_ddp()
        sys.exit(1)

    global_step = 0
    epochs = 3

    if rank == 0:
        print("\nMulti-GPU DDP Training loop started...")

    for epoch in range(epochs):
        # Set epoch on DistributedSampler to maintain proper seed partitioning
        sampler.set_epoch(epoch)

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)

            # A. Notify monitor that a training step has begun
            aegis.begin_step(global_step)

            # These operations are user-owned; AEGIS only observes and returns events.
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # B. Finalize the local observation before collecting rank results.
            event: AegisEvent = aegis.end_step(loss=loss)
            rank_events = gather_events(event, dist)

            # C. Rank zero renders every worker's local state.
            if rank == 0:
                print(f"step {global_step}")
                for worker_rank, worker_event in enumerate(rank_events):
                    print(f"rank {worker_rank}: {worker_event['anomaly_state']}")

            global_step += 1

    if rank == 0:
        print("\n[Rank 0] === DDP Demo Complete! ===")
        print(f"[Rank 0] Telemetry log written to: {os.path.abspath(log_file)}")

    cleanup_ddp()


if __name__ == "__main__":
    # If not launched via torchrun, notify the developer how to run DDP
    if "WORLD_SIZE" not in os.environ:
        print("[AEGIS DDP Example] To run as a multi-GPU distributed job, use torchrun:")
        print("  torchrun --standalone --nproc_per_node=2 examples/ddp_integration.py")
        print("\nFalling back to single-process dry-run simulation...")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    train_ddp()
