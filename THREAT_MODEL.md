# Threat model

The SDK is designed to avoid exporting customer data. It is not a security boundary
for an already compromised host, and it does not prevent a caller from exposing its
own logs. Verify the supplied Trial Core wheel against its delivered SHA-256 value
and restrict local log access.
