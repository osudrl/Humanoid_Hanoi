class BoxManipulationCmd():
    def __init__(self, *args, **kwargs):

        self.input_keys_dict = {}
        self.input_keys_dict["w"] = {
            "description": "walk forward",
            "func": lambda self: setattr(self, "x_velocity", self.x_velocity + 0.01)
        }
        self.input_keys_dict["s"] = {
            "description": "walk backward",
            "func": lambda self: setattr(self, "x_velocity", self.x_velocity - 0.01)
        }
        self.input_keys_dict["a"] = {
            "description": "walk left",
            "func": lambda self: setattr(self, "y_velocity", self.y_velocity + 0.01)  
        }
        self.input_keys_dict["d"] = {
            "description": "walk right",
            "func": lambda self: setattr(self, "y_velocity", self.y_velocity - 0.01)
        }
        self.input_keys_dict["q"] = {
            "description": "turn left",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate + 0.01)
        }
        self.input_keys_dict["e"] = {
            "description": "turn right",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate - 0.01)
        }
        self.input_keys_dict["v"] = {
            "description": "reset stand mode",
            "func": lambda self: setattr(self, "stand_mode", not self.stand_mode)
        }
        self.input_keys_dict["r"] = {
            "description": "reset walk mode",
            "func": lambda self: self.reset_walk_mode()
        }
        self.input_keys_dict["f"] = {
            "description": "freeze upper body",
            "func": lambda self: setattr(self, "freeze_upper_body", not self.freeze_upper_body)
        }
        self.input_keys_dict["z"] = {
            "description": "increase height",
            "func": lambda self: setattr(self, "height", self.height + 0.01)
        }
        self.input_keys_dict["x"] = {
            "description": "decrease height",
            "func": lambda self: setattr(self, "height", self.height - 0.01)
        }
        self.input_keys_dict["u"] = {
            "description": "pick up box",
            "func": lambda self: setattr(self, "pick_up_bit", not self.pick_up_bit)
        }
        self.input_keys_dict["p"] = {
            "description": "put down box",
            "func": lambda self: setattr(self, "put_down_bit", not self.put_down_bit)
        }
        self.input_keys_dict["n"] = {
            "description": "current_time",
            "func": lambda self: setattr(self, "current_time", not self.current_time)
        }

    def reset_walk_mode(self):
        """Reset the walk mode"""
        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.turn_rate = 0.0
        self.stand_mode = False

    def get_control_commands_dict(self):
        return {
            "box distance F/B: ": self.box_start_pose[0],
            "box distance L/R: ": self.box_start_pose[1],
            "box distance U/D: ": self.box_start_pose[2],
            "box rotation w: ": self.box_start_pose[3],
            "box rotation x: ": self.box_start_pose[4],
            "box rotation y: ": self.box_start_pose[5],
            "box rotation z: ": self.box_start_pose[6],
            "walk x: ": self.x_velocity,
            "walk y: ": self.y_velocity,
            "turn: ": self.turn_rate,
            "height: ": self.height,
            "move mode: ": self.stand_mode,
            "freeze upper body: ": self.freeze_upper_body,
            "pick up box cmd: ": self.pick_up_bit,
            "put down box cmd: ": self.put_down_bit,
            "current mode: ": self.action_process[self.action_process_index],
            # "stack 1 status: ": self.stack_1_box_count,
            # "stack 2 status: ": self.stack_2_box_count,
            # "remove target: stack ": self.remove_target + 1,
            "total stack count: ": self.finish_count,
            "target position: ": self.box_target,
            "current time: ": self.current_time,
        }

    def interactive_control(self, c):
        if c in self.input_keys_dict:
            self.input_keys_dict[c]["func"](self)
            self.display_control_commands()