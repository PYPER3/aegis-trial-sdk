"""Minimal single-process AEGIS Trial SDK integration example.

Install the public SDK and the separately supplied compatible Trial Core wheel in
the same CPython 3.11 environment before running this example.
"""

import torch

from aegis import AegisMonitor


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.mlp(inputs)


def main() -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    monitor = AegisMonitor(model, expected_microbatches=1)

    for step in range(3):
        batch = torch.randn(4, 8)
        monitor.begin_step(step)

        # These are ordinary user-owned training operations. AEGIS does not
        # mutate gradients, optimizer state, learning rate, or model parameters.
        optimizer.zero_grad()
        loss = model(batch).square().mean()
        loss.backward()
        optimizer.step()

        event = monitor.end_step(loss=loss)
        if event.detected:
            print(f"step={event.step} state={event.anomaly_state} confidence={event.detection_confidence}")


if __name__ == "__main__":
    main()
