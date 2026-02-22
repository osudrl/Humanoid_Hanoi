# Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement 📦

**Minku Kim**<sup>†</sup>, **Kuan-Chia Chen**<sup>†</sup>, Aayam Shrestha, Li Fuxin, Stefan Lee, and Alan Fern

[📄 Paper](https://arxiv.org/abs/2602.13850) | [🌐 Website](https://osudrl.github.io/Humanoid_Hanoi/) | [📖 BibTeX](#citing-humanoid-hanoi)


<div align="center">

| H1 Robot | G1 Robot | Digit Robot |
|:--------:|:--------:|:-----------:|
| ![H1 Robot](asset/h1.png) | ![G1 Robot](asset/g1.png) | ![Digit Robot](asset/digit.png) |
| *H1 humanoid robot* | *G1 humanoid robot* | *Digit humanoid robot* |

</div>

## Setup Instructions

Conda is required to run the setup script included with this repository. To avoid licensing issues with Anaconda, it is recommended you install conda on your machine via [Miniconda](https://docs.anaconda.com/miniconda/) rather than Anaconda.

To create a fresh conda environment with all the necessary dependencies, run the following commands at the root directory of this repository:

```bash
chmod +x setup.bash
bash setup.bash
```

This script will:
- Set up a new conda environment
- Install additional pip packages
- Install mujoco210

## Usage

To evaluate the environment, run `eval.py` with the `--robot` flag to specify which robot to use:

```bash
python eval.py --robot <robot_name>
```

### Robot Options

The `--robot` flag accepts one of the following options:

- `"h1"` - H1 humanoid robot
- `"g1"` - G1 humanoid robot  
- `"digit"` - Digit humanoid robot (default)

### Additional Flags

- `--offscreen`: Run the evaluation without a display window

### Interactive Controls

During interactive evaluation, press the `'r'` key to reset the environment.

## Note

⚠️ **This repository contains only the environment implementation. Training code is NOT included.**

## Citing Humanoid Hanoi

If you find this repository useful, please consider giving a star ⭐ and citation 📦:

```bibtex
@article{kim2026humanoid,
  title={Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement},
  author={Kim, Minku and Chen, Kuan-Chia and Shrestha, Aayam and Fuxin, Li and Lee, Stefan and Fern, Alan},
  journal={arXiv preprint arXiv:2602.13850},
  year={2026}
}
```