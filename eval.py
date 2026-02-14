import argparse
import sys
from types import SimpleNamespace

from util.evaluation_factory import interactive_eval
from util.env_factory import env_factory, add_env_parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--offscreen', default=False, action='store_true')
    parser.add_argument('--robot', default="digit", choices=["h1", "g1", "digit"])
    
    # Add environment-specific argument parser
    parser = add_env_parser("BoxTowerOfHanoiEnv", parser, is_eval=True)
    args = parser.parse_args()
    
    # Create default environment arguments for BoxTowerOfHanoiEnv
    # Start with required GenericEnv parameters
    env_args = SimpleNamespace(
        robot_name=args.robot,
        terrain=None,
        state_est=False,
    )
    
    # Add environment-specific defaults using add_env_parser
    env_args = add_env_parser("BoxTowerOfHanoiEnv", env_args, is_eval=True)
    
    # Override with specific defaults for this use case
    env_args.simulator_type = "box_tower_of_hanoi_mesh"  # Use mesh model
    env_args.reward_name = "humanoidhanoi"
    env_args.dynamics_randomization = False
    
    # Overwrite env args with any provided command line arguments
    for arg, val in vars(args).items():
        if hasattr(env_args, arg):
            setattr(env_args, arg, val)
    
    print(f"Environment arguments: {env_args}")
    env = env_factory("BoxTowerOfHanoiEnv", env_args)()

    interactive_eval(env=env, offscreen=args.offscreen)
