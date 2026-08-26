# Data flow

Inputs are the caller's in-process model and the optional scalar metrics supplied
at step completion. Outputs are in-memory events and, if configured, JSONL written
to a local caller-selected path. The SDK makes zero network connections.
