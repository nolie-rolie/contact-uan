# G1 LCM deployment bridge

`unitree_sdk2/g1_control.cpp` connects the release's Python controllers to the
Unitree G1. The SDK directory includes the generated LCM message bindings,
Unitree libraries, and CycloneDDS dependencies required to build the bridge.

Follow the repository's [deployment guide](../../../docs/DEPLOYMENT.md) for
dependencies, offline build instructions, and controller setup.

This bridge was adapted from [HOMIE](https://github.com/OpenRobotLab/Homie).
See the parent [attribution and citation](../README.md) and the original
[license](LICENSE). SDK dependency licenses are retained in `unitree_sdk2/`.
