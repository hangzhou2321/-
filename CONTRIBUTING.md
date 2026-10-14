# Contributing

1. Create a focused branch and keep changes within one contract or feature.
2. Add or update regression tests for any policy-facing behavior change.
3. Run `python -m pytest` locally.
4. Do not commit credentials, private data, datasets, checkpoints, replay
   buffers, or generated experiment outputs.

Bug reports should include the Python version, observation/action mode, the
environment contract hash when available, and a minimal reproduction.
