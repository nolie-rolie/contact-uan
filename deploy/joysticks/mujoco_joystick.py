import time

import numpy as np


class MujocoJoystick:
  def __init__(self, task_name=None):
    self.task_name = task_name or ""
    self.is_force_control = "ForceControl" in self.task_name
    self.mode = 0
    self.ctrlmode_left = 0
    self.ctrlmode_right = 0
    self.left_stick: list[float] = [0, 0]
    self.right_stick: list[float] = [0, 0]
    self.force_command = [0.0, 0.0, 0.0]
    self.left_upper_switch = 0
    self.left_lower_left_switch = 0
    self.left_lower_right_switch = 0
    self.right_upper_switch = 0
    self.right_lower_left_switch = 0
    self.right_lower_right_switch = 0
    self.left_upper_switch_pressed = 0
    self.left_lower_left_switch_pressed = 0
    self.left_lower_right_switch_pressed = 0
    self.right_upper_switch_pressed = 0
    self.right_lower_left_switch_pressed = 0
    self.right_lower_right_switch_pressed = 0
    self.current_policy = 1
    self.running = False
    self.run_thread = None
    self.root = None
    self.step_counter = 0

    self.r1_button = 0
    self.a_button, self.b_button, self.x_button, self.y_button = 0, 0, 0, 0
    self._r1_last_press_time = 0.0
    self._r1_cooldown_s = 0.25

    # WTW gait parameters (adjustable via keyboard)
    self.gait_frequency_cmd = 0.0  # in [-1, 1]
    self.swing_height_cmd = 0.0  # in [-1, 1]

    # smoothing
    self.left_stick_smooth = [0, 0]
    self.right_stick_smooth = [0, 0]

  def _print_force_command(self):
    fx, fy, fz = self.get_force_command()
    print(f"[MujocoForceCmd] norm=({fx:+.2f}, {fy:+.2f}, {fz:+.2f})")

  def key_callback(self, key):
    """
    Key callback function.
    - window: the window that received the event
    - key: the keyboard key that was pressed or released
    - scancode: the system-specific scancode of the key
    - action: GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
    - mods: bitfield describing which modifier keys were held down
    """
    force_step = 0.1
    force_changed = False
    if key == 32:  # SPACE
      # self.right_lower_right_switch_pressed = 1 - self.right_lower_right_switch_pressed
      # self.right_lower_right_switch = 1 - self.right_lower_right_switch
      self.left_stick[0] = 0
      self.left_stick[1] = 0
      self.right_stick[0] = 0
      self.right_stick[1] = 0
      self.force_command = [0.0, 0.0, 0.0]
      force_changed = self.is_force_control
      self.gait_frequency_cmd = 0.0
      self.swing_height_cmd = 0.0
    elif key == 265:  # UP
      if self.is_force_control:
        self.force_command[0] = min(1.0, self.force_command[0] + force_step)
        force_changed = True
      else:
        self.left_stick[1] += 0.2
    elif key == 264:  # DOWN
      if self.is_force_control:
        self.force_command[0] = max(-1.0, self.force_command[0] - force_step)
        force_changed = True
      else:
        self.left_stick[1] -= 0.2
    elif key == 263:  # LEFT
      if self.is_force_control:
        self.force_command[2] = max(-1.0, self.force_command[2] - force_step)
        force_changed = True
      else:
        self.right_stick[0] -= 0.2
    elif key == 262:  # RIGHT
      if self.is_force_control:
        self.force_command[2] = min(1.0, self.force_command[2] + force_step)
        force_changed = True
      else:
        self.right_stick[0] += 0.2
    elif key == 44:  # ,
      self.current_policy = 1
    elif key == 46:  # .
      self.current_policy = 2
    elif key in (266, 91):  # page up or [
      if self.is_force_control:
        self.force_command[2] = min(1.0, self.force_command[2] + force_step)
        force_changed = True
      else:
        self.right_stick[1] += 0.1
    elif key in (267, 93):  # page down or ]
      if self.is_force_control:
        self.force_command[2] = max(-1.0, self.force_command[2] - force_step)
        force_changed = True
      else:
        self.right_stick[1] -= 0.1
    # 5 / 6 — gait frequency (decrease / increase)
    elif key == 53:  # 5
      self.gait_frequency_cmd = max(-1.0, self.gait_frequency_cmd - 0.2)
    elif key == 54:  # 6
      self.gait_frequency_cmd = min(1.0, self.gait_frequency_cmd + 0.2)
    # 7 / 8 — swing height (decrease / increase)
    elif key == 55:  # 7
      self.swing_height_cmd = max(-1.0, self.swing_height_cmd - 0.2)
    elif key == 56:  # 8
      self.swing_height_cmd = min(1.0, self.swing_height_cmd + 0.2)
    # A button
    elif key == 65:
      self.a_button = 1
    # B button
    elif key == 66:
      self.b_button = 1
    # X button
    elif key == 88:
      self.x_button = 1
    # Y button
    elif key == 89:
      self.y_button = 1
    # R key -> emulate R1 press for sampled force-command toggle
    elif key == 82:
      now = time.time()
      if now - self._r1_last_press_time > self._r1_cooldown_s:
        self.r1_button = 1
        self._r1_last_press_time = now

    if force_changed:
      self._print_force_command()

  def update_stick(self, stick, x, y):
    if stick == "left":
      self.left_stick = [x, y]
    elif stick == "right":
      self.right_stick = [x, y]

  def update_switch(self, switch, state):
    if switch == "left_upper":
      self.left_upper_switch = state
    elif switch == "right_upper":
      self.right_upper_switch = state
    elif switch == "left_lower":
      self.left_lower_left_switch = state
      self.left_lower_left_switch_pressed = state
    elif switch == "right_lower":
      self.right_lower_right_switch = state
      self.right_lower_right_switch_pressed = state

  def get_buttons(self):
    return np.array(
      [
        self.left_lower_left_switch,
        self.left_upper_switch,
        self.right_lower_right_switch,
        self.right_upper_switch,
      ]
    )

  def get_abxy(self):
    abxy = np.array([self.a_button, self.b_button, self.x_button, self.y_button])
    self.a_button, self.b_button, self.x_button, self.y_button = 0, 0, 0, 0
    return abxy

  def get_r1(self):
    r1 = self.r1_button
    self.r1_button = 0
    return r1

  def get_command(self):
    # self.left_stick_smooth = [0.8 * self.left_stick_smooth[0] + 0.2 * self.left_stick[0], 0.8 * self.left_stick_smooth[1] + 0.2 * self.left_stick[1]]
    # self.right_stick_smooth = [0.8 * self.right_stick_smooth[0] + 0.2 * self.right_stick[0], 0.8 * self.right_stick_smooth[1] + 0.2 * self.right_stick[1]]

    # return [self.left_stick_smooth[1], self.left_stick_smooth[0], self.right_stick_smooth[0], self.right_stick_smooth[1]]

    return [
      self.left_stick[1],
      self.left_stick[0],
      self.right_stick[0],
      self.right_stick[1],
    ]

  def get_force_command(self):
    if self.is_force_control:
      return [self.force_command[0], 0.0, self.force_command[2]]
    return [self.right_stick[1], -self.right_stick[0], self.left_stick[1]]

  def get_gait_params(self):
    """Return [gait_frequency_cmd, swing_height_cmd] both in [-1, 1]."""
    return [self.gait_frequency_cmd, self.swing_height_cmd]

  def get_current_policy(self):
    return self.current_policy
