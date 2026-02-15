import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import vgamepad as vg
import cv2
import socket
import subprocess
import time
from PIL import Image
from collections import deque

from steam import (
    LIFTOFF_GAME_SINGLE_PLAYER_BUTTON_POS,
    LIFTOFF_GAME_QUICK_PLAY_BUTTON_POS,
    LIFTOFF_GAME_QUICK_PLAY_RANDOM_BUTTON_POS
)
from steam.liftoff_telemetry import LiftoffTelemetry


class DroneSimulatorHoverEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 240
    }

    def __init__(self, render_mode=None, screen_region=None, action_meanings=None):
        super().__init__()
        self.liftoff_telemetry = LiftoffTelemetry()
        # Continuous action space for:
        # throttle (cannot go below 0),
        # yaw (left=-1,telemetry right=1),
        # pitch (left=-1, right=1),
        # roll (left=-1, right=1)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )
        # Observation Space: Image + Telemetry
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-float("inf"), high=float("inf"), shape=(21,), dtype=np.float32)
        })

        self.render_mode = render_mode
        # Define screen region to capture (x, y, width, height) - adjust via trial/error
        self.screen_region = screen_region or {'top': 100, 'left': 100, 'width': 800, 'height': 600}

        self.gamepad = vg.VX360Gamepad()

        self._elapsed_steps = 0

        self.__ydotoold()
        self.__start_game()

        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.__define_boundaries()
        print("Boundaries")
        print(self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z)
        self.current_obs = None
        self.out_of_bounds_window = deque()

    def __ydotoold(self):
        """
        Start ydotool daemon service for ydotool subprocess commands if it hasn't started.
        """
        subprocess.run(['systemctl', '--user', 'start', 'ydotoold'])

    def __start_game(self):
        time.sleep(5)
        buttons = [
            LIFTOFF_GAME_SINGLE_PLAYER_BUTTON_POS,
            LIFTOFF_GAME_QUICK_PLAY_BUTTON_POS,
            LIFTOFF_GAME_QUICK_PLAY_RANDOM_BUTTON_POS
        ]
        for x, y, w, h in buttons:
            cx, cy = x + w // 2, y + h // 2
            subprocess.run(['hyprctl', 'dispatch', 'movecursor', f'{cx} {cy}'])
            time.sleep(0.5)
            subprocess.run(['ydotool', 'click', '0xC0'])
            time.sleep(1)

    def __quit_game():
        subprocess.run(['ydotool', 'key', 'KEY_ESC'])
        time.sleep(1)
        buttons = [
            LIFTOFF_GAME_QUIT_TO_MAIN_MENU_BUTTON_POS,
            LIFTOFF_GAME_QUIT_TO_MAIN_MENU_CONFIRM_BUTTON_POS
        ]
        for x, y, w, h in buttons:
            cx, cy = x + w // 2, y + h // 2
            subprocess.run(['hyprctl', 'dispatch', 'movecursor', f'{cx} {cy}'])
            time.sleep(0.5)
            # left click
            subprocess.run(['yodotool', 'click', '0xC0'])
            time.sleep(1)

    def __define_boundaries(self):
        """Define boundaries for drone."""
        tel = self.liftoff_telemetry.capture_telemetry()
        min_x, max_x = tel[0] - 5, tel[0] + 5
        min_y, max_y = tel[1], tel[1] + 10
        min_z, max_z = tel[2] - 5, tel[2] + 5

        return min_x, max_x, min_y, max_y, min_z, max_z

    def __is_within_bounds(self, tel: np.array):
        x, y, z = tel[0:3]
        return self.min_x < x < self.max_x and \
            self.min_y < y < self.max_y and \
            self.min_z < z < self.max_z

    def _get_obs(self):
        tel = self.liftoff_telemetry.capture_telemetry()

        # Capture screen
        result = subprocess.run(['grim', '-t', 'ppm', '-'], stdout=subprocess.PIPE)
        # Decode the raw PPM image into a NumPy array for OpenCV
        image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))

        return {"image": image, "telemetry": tel}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.right_joystick(x_value=0, y_value=0)
        self.gamepad.right_trigger(value=0)
        self.gamepad.update()

        subprocess.run(['ydotool', 'key', '19:1', "19:0"])
        time.sleep(2)

        subprocess.run(['ydotool', 'key', '2:1', '2:0'])
        time.sleep(3)

        obs = self._get_obs()
        self.current_obs = obs
        return obs

    def step(self, action):
        throttle = int(action[0] * 32767)
        yaw = int(action[1] * 32767)
        pitch = int(action[2] * 32767)
        roll = int(action[3] * 32767)

        print("Action: ", throttle, yaw, pitch, roll)

        self.gamepad.left_joystick(x_value=roll, y_value=pitch)
        self.gamepad.right_joystick(x_value=yaw, y_value=0)
        self.gamepad.right_trigger(value=throttle // 128)

        self.gamepad.update()

        # time.sleep(1 / 30)  # ~30 FPS step

        obs = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # 3. Logic for Terminated (Crash) and Truncated (Time)
        # Simple crash detection: Velocity becomes 0 while throttle is high, or height < threshold
        velocity_mag = np.linalg.norm(obs["telemetry"][3:6])
        altitude = obs["telemetry"][1] # Adjust based on your Y/Z axis config

        self.out_of_bounds_window.append(not self.__is_within_bounds(obs["telemetry"]))
        if len(self.out_of_bounds_window) > 10:
            self.out_of_bounds_window.popleft()

        terminated = all(self.out_of_bounds_window)    # if drone is out of bounds
        if self._elapsed_steps >= 1000:  # Track self._elapsed_steps in full impl
            truncated = True

        # 4. Reward Logic
        reward = 0.1 * (1 - (altitude - self.current_obs["telemetry"][1]) / altitude) if self.current_obs else 0.1
        if terminated: reward = -10.0
        # print("Reward: ", reward)

        self.current_obs = obs

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Handled in step for simplicity
        self.current_obs = obs

    def close(self):
        self.__quit_game()
        cv2.destroyAllWindows()


register(id="Liftoff-hover-v0", entry_point=DroneSimulatorHoverEnv, max_episode_steps=300)

