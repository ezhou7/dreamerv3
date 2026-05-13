import cv2
import subprocess
import time
import gymnasium as gym
import numpy as np

from collections import deque
from gymnasium.spaces import Box, Dict
from gymnasium.envs.registration import register

from steam import (
    LIFTOFF_GAME_SINGLE_PLAYER_BUTTON_POS,
    LIFTOFF_GAME_QUICK_PLAY_BUTTON_POS,
    LIFTOFF_GAME_QUICK_PLAY_RANDOM_BUTTON_POS,
    LIFTOFF_GAME_QUIT_TO_MAIN_MENU_BUTTON_POS,
    LIFTOFF_GAME_QUIT_TO_MAIN_MENU_CONFIRM_BUTTON_POS
)
from steam.liftoff.telemetry import LiftoffTelemetry
from steam.liftoff.transmitter import EvdevTransmitter


class LiftoffHoverEnvUInput(gym.Env):
    def __init__(self, render_mode=None, screen_region=None, action_meanings=None):
        super().__init__()
        self.action_space = Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "telemetry": Box(low=-float("inf"), high=float("inf"), shape=(21,), dtype=np.float32)
        })

        self.render_mode = render_mode
        self.screen_region = screen_region or {'top': 100, 'left': 100, 'width': 800, 'height': 600}

        self._elapsed_steps = 0
        self._initialized = False
        self.current_obs = None
        self.out_of_bounds_window = deque()
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        self._action_alpha = 0.3

    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True
        self.liftoff_telemetry = LiftoffTelemetry()
        self.transmitter = EvdevTransmitter()
        self.__ydotoold()
        self.__start_game()
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.__define_boundaries()
        self.target_x = (self.min_x + self.max_x) / 2.0
        self.target_y = (self.min_y + self.max_y) / 2.0
        self.target_z = (self.min_z + self.max_z) / 2.0
        print("Boundaries")
        print(self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z)
        print("Target hover position:", self.target_x, self.target_y, self.target_z)

    def __ydotoold(self):
        subprocess.run(['systemctl', '--user', 'start', 'ydotool'])

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

    def __quit_game(self):
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
            subprocess.run(['ydotool', 'click', '0xC0'])
            time.sleep(1)

    def __define_boundaries(self):
        tel = self.liftoff_telemetry.capture_telemetry()
        min_x, max_x = tel[0] - 10, tel[0] + 10
        min_y, max_y = tel[1] - 1, tel[1] + 15
        min_z, max_z = tel[2] - 10, tel[2] + 10
        return min_x, max_x, min_y, max_y, min_z, max_z

    def __is_within_bounds(self, tel: np.array):
        x, y, z = tel[0:3]
        return self.min_x < x < self.max_x and \
            self.min_y < y < self.max_y and \
            self.min_z < z < self.max_z

    def _get_obs(self):
        tel = self.liftoff_telemetry.capture_telemetry()

        result = subprocess.run(['grim', '-t', 'ppm', '-'], stdout=subprocess.PIPE)
        image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))

        return {"image": image, "telemetry": tel}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._lazy_init()
        self._elapsed_steps = 0
        self.out_of_bounds_window.clear()

        self.transmitter.center_all()

        subprocess.run(['ydotool', 'key', '19:1', "19:0"])
        time.sleep(2)

        # Arm drone: hold throttle all the way down for 1 second
        print("Arming drone...")
        self.transmitter.set_sticks(throttle=-32768)
        self.transmitter.update()
        time.sleep(1)
        self.transmitter.center_all()
        time.sleep(3)

        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.__define_boundaries()
        self.target_x = (self.min_x + self.max_x) / 2.0
        self.target_y = (self.min_y + self.max_y) / 2.0
        self.target_z = (self.min_z + self.max_z) / 2.0

        obs = self._get_obs()
        tel = obs["telemetry"]
        x, y, z = tel[0], tel[1], tel[2]
        print(f"Bounding box: X[{self.min_x:.2f}, {self.max_x:.2f}] Y[{self.min_y:.2f}, {self.max_y:.2f}] Z[{self.min_z:.2f}, {self.max_z:.2f}]")
        print(f"Drone position: X={x:.2f} Y={y:.2f} Z={z:.2f}")
        print(f"Relative to box: X={((x - self.min_x) / (self.max_x - self.min_x)):.2%} Y={((y - self.min_y) / (self.max_y - self.min_y)):.2%} Z={((z - self.min_z) / (self.max_z - self.min_z)):.2%}")
        self.current_obs = obs
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._smoothed_action = self._action_alpha * action + (1 - self._action_alpha) * self._smoothed_action

        throttle = int(np.clip(self._smoothed_action[0], -1.0, 1.0) * 32767)
        yaw = int(np.clip(self._smoothed_action[1], -1.0, 1.0) * 32767)
        pitch = int(np.clip(self._smoothed_action[2], -1.0, 1.0) * 32767)
        roll = int(np.clip(self._smoothed_action[3], -1.0, 1.0) * 32767)

        print("Action: ", throttle, yaw, pitch, roll)

        self.transmitter.set_sticks(roll=roll, pitch=pitch, throttle=throttle, yaw=yaw)
        self.transmitter.update()

        self._elapsed_steps += 1

        obs = self._get_obs()
        tel = obs["telemetry"]

        x, altitude, z = tel[0], tel[1], tel[2]
        speed = np.linalg.norm(tel[3:6])
        qw = tel[9]
        gyro_mag = np.linalg.norm(tel[10:13])

        alt_err = altitude - self.target_y
        r_altitude = np.exp(-0.5 * (alt_err / 2.0) ** 2)

        horiz_dist = np.sqrt((x - self.target_x) ** 2 + (z - self.target_z) ** 2)
        r_horizontal = np.exp(-0.5 * (horiz_dist / 2.0) ** 2)

        r_stability = 0.5 * np.exp(-0.5 * speed ** 2) + 0.5 * np.exp(-0.5 * gyro_mag ** 2)

        r_orientation = qw ** 2

        reward = float(
            0.40 * r_altitude
            + 0.20 * r_horizontal
            + 0.25 * r_stability
            + 0.15 * r_orientation
        )

        self.out_of_bounds_window.append(not self.__is_within_bounds(tel))
        if len(self.out_of_bounds_window) > 10:
            self.out_of_bounds_window.popleft()

        terminated = len(self.out_of_bounds_window) == 10 and all(self.out_of_bounds_window)
        truncated = self._elapsed_steps >= 300

        if terminated:
            reward = -1.0

        self.current_obs = obs

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self._initialized:
            self.__quit_game()
            self.transmitter.close()
        cv2.destroyAllWindows()


register(id="Liftoff-hover-uinput-v0", entry_point=LiftoffHoverEnvUInput, max_episode_steps=300)
