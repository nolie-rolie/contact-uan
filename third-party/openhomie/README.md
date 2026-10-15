# HOMIE deployment bridge

This directory vendors the G1 LCM bridge adapted from
[HOMIE](https://github.com/OpenRobotLab/Homie), by Qingwei Ben, Feiyu Jia,
Jia Zeng, Junting Dong, Dahua Lin, and Jiangmiao Pang.

`HomieDeploy/unitree_sdk2/` contains the bridge, generated LCM message
bindings, and the Unitree/CycloneDDS dependencies used by this release.
Build and usage instructions are in the repository's
[deployment guide](../../docs/DEPLOYMENT.md).

The original HOMIE license is retained in [LICENSE](LICENSE) and
[HomieDeploy/LICENSE](HomieDeploy/LICENSE). The Unitree SDK and its dependencies
retain their own license files under `HomieDeploy/unitree_sdk2/`.

## Citation

```bibtex
@article{ben2025homie,
  title={HOMIE: Humanoid Loco-Manipulation with Isomorphic Exoskeleton Cockpit},
  author={Ben, Qingwei and Jia, Feiyu and Zeng, Jia and Dong, Junting and Lin, Dahua and Pang, Jiangmiao},
  journal={arXiv preprint arXiv:2502.13013},
  year={2025}
}
```
