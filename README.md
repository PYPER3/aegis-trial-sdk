# AEGIS Trial SDK

AEGIS Trial SDK is the public local integration layer for a separately supplied
compatible AEGIS Trial Core wheel. It is detection-only: it observes local model
state, returns local events, and does not change a training run.

## Initial Trial platform

The initial live Trial release requires Linux x86_64 and CPython 3.11. The supplied
Core wheel currently has the form:

```text
aegis_trial_core-0.1.2-cp311-cp311-linux_x86_64.whl
```

Version numbers and filenames do not establish scientific qualification. Use
only a Core wheel whose supplied hash and release status explicitly identify it
as authorized for the trial; engineering-candidate wheels must not be deployed.

The SDK exposes PyTorch, Hugging Face Trainer, and distributed integration helpers.
Those APIs do not by themselves establish that a particular model architecture or
multi-worker training configuration is validated for the Trial Core. Use only the
configuration supplied with your trial materials.

## Install

Install both components into the same CPython 3.11 environment:

```bash
python3.11 -m pip install /path/to/aegis_trial_core-0.1.2-cp311-cp311-linux_x86_64.whl
python3.11 -m pip install aegis-trial-sdk
```

When installing from a source checkout, replace the second command with
`python3.11 -m pip install .`.

Before installation, check the Core wheel against the SHA-256 value supplied with
your trial delivery:

```bash
echo "<supplied-sha256>  aegis_trial_core-0.1.2-cp311-cp311-linux_x86_64.whl" | sha256sum --check -
```

The public SDK does not install or bundle the Core. Constructing `AegisMonitor`
without the separately supplied compatible wheel produces a local error.

## Training-loop lifecycle

Call `begin_step` before the monitored forward pass and `end_step` after the normal
user-owned training step. See [examples/pytorch_training_loop.py](examples/pytorch_training_loop.py).

```python
from aegis import AegisMonitor

# This must equal the number of training forwards accumulated into each
# optimizer step. The monitor aborts on missing or extra captures.
monitor = AegisMonitor(model, expected_microbatches=1)
monitor.begin_step(step)
loss = model(batch).sum()
loss.backward()        # your ordinary training loop
optimizer.step()       # your ordinary training loop
event = monitor.end_step(loss=loss)
if event.detected:
    print(event.anomaly_state, event.detection_confidence)
```

AEGIS does not modify gradients, optimizer state, learning rate, checkpoints,
workers, or training control. It only returns the event.

If the training loop uses gradient accumulation, construct the monitor with the
same explicit count. For example, use `expected_microbatches=4` when four forward
and backward passes precede each optimizer step. A mismatch raises an error
instead of silently evaluating an incomplete optimizer step. The compatible Core
combines every configured observation before producing the local event.

## Event schema

`AegisMonitor.end_step()` returns an immutable `AegisEvent` with:

- `anomaly_state`: a generic local state, such as `clear` or `detected`.
- `detection_confidence`: a numeric confidence when available, otherwise `None`.
- `recommended_next_step`: informational text for the caller. In the Trial, it
  does not select, authorize, or execute an intervention.
- `detected`: a convenience property derived from `anomaly_state`.

The Core's `signal_detected` decision field is not emitted as a separate public
event field; callers should use `event.detected`.

Use `aegis.integrations.AegisCallback(monitor)` with a Hugging Face Trainer, or
`aegis.integrations.gather_events(event, torch.distributed)` to collect one event
per worker on rank zero. The distributed example is an API integration reference,
not a broader validation claim. See `examples/ddp_integration.py`.

## Data and network boundary

AEGIS runs locally. The public SDK does not upload training data, model state, or
event logs, and opens no network connections. See [DATA_FLOW.md](DATA_FLOW.md) and
[PRIVACY.md](PRIVACY.md).

`aegis-demo --preview` writes a clearly simulated evaluation-report preview.

## Public SDK release integrity

Every published SDK release should include a `SHA256SUMS` file alongside its
wheel and source distribution. Verify the downloaded artifact against that
file before installation:

```bash
sha256sum --check SHA256SUMS
```

`SHA256SUMS` covers the public SDK artifacts only. The separately supplied
Trial Core has its own delivery manifest and SHA-256 value; never substitute
one checksum for the other.
