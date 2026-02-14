import numpy as np
import sys
import termios
import time
import torch

from util.keyboard import Keyboard
from util.colors import OKGREEN, FAIL, ENDC

def interactive_eval(env, offscreen=False):
    """Simply evaluating policy in visualization window with user input

    Args:
        env: Environment instance
    """

    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print(f"Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds.{ENDC}")
    keyboard = Keyboard()

    with torch.no_grad():
        state = env.reset()
        done = False
        episode_length = 0
        episode_reward = []
        display_state = True

        env.sim.viewer_init(fps = env.default_policy_rate)
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()
        while render_state:
            start_time = time.time()
            cmd = None
            if keyboard.data():
                cmd = keyboard.get_input()
            if display_state and not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                env.step()
                episode_length += 1
            if cmd is not None:
                env.interactive_control(cmd)
            if cmd == "r":
                done = True
            if cmd == " ":
                display_state = not display_state
            # env.hw_step()
            render_state = env.sim.viewer_render()
            delaytime = max(0, 1/env.default_policy_rate - (time.time() - start_time))
            time.sleep(delaytime)
            if done:
                state = env.reset()
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                env.display_control_commands()
                episode_length = 0
                episode_reward = []
                done = False

        keyboard.restore()
        print(f"\033[{len(env.control_commands_dict) + 3 - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)