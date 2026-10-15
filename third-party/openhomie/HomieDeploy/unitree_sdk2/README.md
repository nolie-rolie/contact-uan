# Unitree SDK2 and G1 LCM bridge

This directory contains the Unitree SDK2 dependencies and the HOMIE-derived
`g1_control` bridge used by the Python controllers in this release. CMake builds
only the G1 bridge. `example/wireless_controller/advanced_gamepad.hpp` supplies
its wireless-controller decoding helper.

## Prebuilt library environment

- Ubuntu 20.04 LTS
- `aarch64` and `x86_64`
- GCC 9.4.0

For build dependencies and the complete setup, follow the repository's
[deployment guide](../../../../docs/DEPLOYMENT.md). The Python controllers use
the generated modules in `lcm_types/`.

The SDK's [license](LICENSE), third-party notices under `licenses/`, and
CycloneDDS license files are retained. HOMIE attribution is in the parent
[README](../../README.md).
