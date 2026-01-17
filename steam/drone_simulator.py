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

from steam import (
    LIFTOFF_GAME_SINGLE_PLAYER_BUTTON_POS,
    LIFTOFF_GAME_QUICK_PLAY_BUTTON_POS,
    LIFTOFF_GAME_QUICK_PLAY_RANDOM_BUTTON_POS
)


class DroneSimulatorEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 240
    }

    def __init__(self, render_mode=None, screen_region=None, action_meanings=None):
        super().__init__()
        # Continuous action space for:
        # throttle (cannot go below 0),
        # yaw (left=-1, right=1),
        # pitch (left=-1, right=1),
        # roll (left=-1, right=1)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )
        # Observation Space: Image + Telemetry
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-1, high=1, shape=(21,), dtype=np.float32)
        })

        # UDP Setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 9001))
        self.sock.settimeout(0.5)
        self.fmt = "<f fff fff ffff fff ffff Bffff" # Matches TelemetryConfiguration.json

        self.render_mode = render_mode
        # Define screen region to capture (x, y, width, height) - adjust via trial/error
        self.screen_region = screen_region or {'top': 100, 'left': 100, 'width': 800, 'height': 600}

        self.gamepad = vg.VX360Gamepad()

        self._elapsed_steps = 0

        self.__ydotoold()
        self.__start_game()

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

    def _get_obs(self):
        # Capture Telemetry
        try:
            data, _ = self.sock.recvfrom(1024)
            unpacked = struct.unpack(self.fmt, data[:struct.calcsize(self.fmt)])
            # Flattened vector: Pos(3), Vel(3), Att(4), Gyro(3), Inputs(4), Motors(1 byte ignored + 4)
            # We skip index 18 (motor count)
            tel = np.array(unpacked[1:18] + unpacked[19:23], dtype=np.float32)
        except:
            tel = np.zeros(21, dtype=np.float32) # Fallback if packet missed

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

        subprocess.run(['ydotool', 'key', 'KEY_R'])
        time.sleep(2)

        obs = self._get_obs()
        self.current_obs = obs
        return obs

    def step(self, action):
        throttle = int(action[0] * 32767)
        yaw = int(action[1] * 32767)
        pitch = int(action[2] * 32767)
        roll = int(action[3] * 32767)

        self.gamepad.left_joystick(x_value=roll, y_value=pitch)
        self.gamepad.right_joystick(x_value=yaw, y_value=0)
        self.gamepad.right_trigger(value=throttle // 128)

        self.gamepad.update()

        time.sleep(1 / 30)  # ~30 FPS step

        obs = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # 3. Logic for Terminated (Crash) and Truncated (Time)
        # Simple crash detection: Velocity becomes 0 while throttle is high, or height < threshold
        velocity_mag = np.linalg.norm(obs["telemetry"][3:6])
        altitude = obs["telemetry"][1] # Adjust based on your Y/Z axis config

        terminated = bool(velocity_mag < 0.1 and action[0] > 0.5) # Example crash heuristic
        if self._elapsed_steps >= 1000:  # Track self._elapsed_steps in full impl
            truncated = True

        # 4. Reward Logic
        reward = velocity_mag * 0.1 # Reward for moving fast
        if terminated: reward = -10.0

        self.current_obs = obs
        # if self.render_mode == 'human':
        #     cv2.imshow('Game', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(1)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Handled in step for simplicity

    def close(self):
        self.__quit_game()
        cv2.destroyAllWindows()


register(id="Liftoff-v0", entry_point=DroneSimulatorEnv, max_episode_steps=300)

