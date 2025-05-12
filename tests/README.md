# Test Scripts

This directory contains test scripts for the Trade Position Calculator application.

## Contents

- `test_compute_direct.py`: A standalone script to test the position calculation logic with actual trade data. This script demonstrates how the system filters out expired contracts based on their expiry dates.

## Running Tests

To run the test scripts:

```bash
# From the project root directory
python tests/test_compute_direct.py
```

## Test Outputs

Test outputs are saved in the `tests/outputs` directory with timestamps in the filename.

## Notes

- These test scripts are separate from the main application code and are used for testing and debugging purposes only.
- The scripts use the actual trade data from the `DATA` directory but save outputs to a separate location to avoid mixing with production data.
- When making changes to the core logic, it's recommended to test with these scripts first before integrating into the main application.
