## Setup Instructions
Conda is required to run the setup script included with this repository.
To avoid licensing issues with Anaconda, it is recommended you install conda on your machine via
[Miniconda](https://docs.anaconda.com/miniconda/) rather than Anaconda.

To create a fresh conda env with all the necessary dependencies, simply run
```
chmod +x setup.bash
bash setup.bash
```
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and install mujoco210.

## Usage

To evaluate the environment, run `eval.py` with the `--robot` flag to specify which robot to use:
```
python eval.py --robot <robot_name>
```

The `--robot` flag accepts one of the following options:
- `"h1"` - H1 humanoid robot
- `"g1"` - G1 humanoid robot  
- `"digit"` - Digit humanoid robot (default)

You can also use the `--offscreen` flag to run the evaluation without a display window.

During interactive evaluation, press the `'r'` key to reset the environment.

## Note

**This repository contains only the environment implementation. Training code is NOT included.**