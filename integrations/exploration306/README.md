# Exploration adapters

This directory contains the deployed SEVA exploration policy, native adapter,
read-only C++ candidate bridge and existing regression tests. See the repository
[README](../../README.md) for architecture and results and the
[integration guide](../../docs/INTEGRATION.md) for external ROS/SDK requirements.

`reward_policy.py` is the policy implementation used for the paper's source
inspection. `reward_runner.py` preserves its candidate, validation and outcome
logic; its bridge executable location is configurable for the public release.
Credential, endpoint and installation-path changes are packaging changes, not
new experiments or a new policy.
