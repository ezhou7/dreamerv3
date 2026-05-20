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


UPRIGHT_THRESHOLD = 0.707   # drone's up-axis y-component below this ≈ tilted >45° from level
GYRO_THRESHOLD = 3.0        # rad/s magnitude — above this we treat the drone as spinning out
UNSTABLE_WINDOW = 10        # consecutive unstable steps that triggers a forced reset
OOB_WINDOW = 5              # consecutive out-of-bounds steps that triggers a forced reset
CRASH_PENALTY = -10.0       # one-shot penalty applied on the terminating step
UNSTABLE_STEP_PENALTY = -1.0  # per-step reward when unstable — dominates the normal terms

# Stability streak bonus: a per-step add-on that grows with consecutive stable
# steps so the agent gets directly rewarded for "staying alive longer". Curve
# is `MAX * (1 - exp(-streak / TAU))` — starts at 0, rises quickly at first,
# and asymptotes to MAX. Capped on purpose so it remains a side objective
# (worth at most ~15% of a perfect hover step) rather than out-competing the
# main hover reward.
STABILITY_STREAK_MAX = 0.15
STABILITY_STREAK_TAU = 50.0    # steps to reach ~63% of the plateau (~5s at 10Hz step rate)


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
        self.unstable_window = deque()
        self._stable_streak = 0
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        self._action_alpha = 0.3
        self._prev_action = np.zeros(4, dtype=np.float32)

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
        self.unstable_window.clear()
        self._stable_streak = 0

        self.transmitter.center_all()

        subprocess.run(['ydotool', 'key', '19:1', "19:0"])
        time.sleep(2)

        # Arm drone: hold throttle all the way down for ~2.5 seconds. Liftoff
        # (and most BetaFlight presets) require throttle to sit below the
        # configured zero threshold for a sustained period before arming;
        # 1 s was racy. The longer hold also gives the kernel time to push
        # the min-throttle event through past EV_ABS deduplication.
        print("Arming drone...")
        self.transmitter.set_sticks(throttle=-32768)
        self.transmitter.update()
        subprocess.run(['ydotool', 'key', '2:1', "2:0"])
        time.sleep(2)
        self.transmitter.center_all()
        time.sleep(0.1)

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
        self._prev_action = np.zeros(4, dtype=np.float32)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._smoothed_action = self._action_alpha * action + (1 - self._action_alpha) * self._smoothed_action

        throttle = int(np.clip(self._smoothed_action[0], -1.0, 1.0) * 32767)
        yaw = int(np.clip(self._smoothed_action[1], -1.0, 1.0) * 32767)
        pitch = int(np.clip(self._smoothed_action[2], -1.0, 1.0) * 32767)
        roll = int(np.clip(self._smoothed_action[3], -1.0, 1.0) * 32767)

        # print("Action: ", throttle, yaw, pitch, roll)

        self.transmitter.set_sticks(roll=roll, pitch=pitch, throttle=throttle, yaw=yaw)
        self.transmitter.update()

        self._elapsed_steps += 1

        obs = self._get_obs()
        tel = obs["telemetry"]

        x, altitude, z = tel[0], tel[1], tel[2]
        speed = np.linalg.norm(tel[3:6])
        # Quaternion is (qx, qy, qz, qw) in Unity convention. Drone's up-axis
        # in the world frame has y-component 1 - 2(qx^2 + qz^2): this is +1
        # when perfectly level, 0 on its side, -1 inverted. Better tilt
        # signal than qw^2 because it ignores yaw rotation around vertical.
        qx, _qy, qz, _qw = tel[6], tel[7], tel[8], tel[9]
        up_y = float(np.clip(1.0 - 2.0 * (qx ** 2 + qz ** 2), -1.0, 1.0))
        gyro_mag = float(np.linalg.norm(tel[10:13]))

        # Instability check — drone is tilted past ~45° from level OR spinning
        # faster than `GYRO_THRESHOLD`. Both are bad states for a hover.
        is_unstable = (up_y < UPRIGHT_THRESHOLD) or (gyro_mag > GYRO_THRESHOLD)

        # 3D distance to target with exp(-d) shape — sharp gradient near the target.
        dist = np.sqrt(
            (x - self.target_x) ** 2
            + (altitude - self.target_y) ** 2
            + (z - self.target_z) ** 2
        )
        r_position = np.exp(-dist)                       # 1 at target, ~0.37 @ 1m, ~0.05 @ 3m
        r_velocity = np.exp(-0.5 * speed)                # rewards standing still
        r_gyro = np.exp(-0.5 * gyro_mag)                 # rewards not rotating
        r_upright = up_y                                 # [-1, 1] — direct upright signal

        action_delta = float(np.linalg.norm(action - self._prev_action))
        r_smoothness = -0.02 * action_delta
        self._prev_action = action.copy()

        if is_unstable:
            # Dominate the reward when the drone is in a bad attitude — no
            # amount of being near the target offsets being sideways or
            # spinning out. Gradient toward "get upright" is strong.
            self._stable_streak = 0
            reward = UNSTABLE_STEP_PENALTY
        else:
            self._stable_streak += 1
            r_streak = STABILITY_STREAK_MAX * (
                1.0 - np.exp(-self._stable_streak / STABILITY_STREAK_TAU)
            )
            reward = float(
                0.30 * r_position
                + 0.15 * r_velocity
                + 0.15 * r_gyro
                + 0.30 * r_upright     # heavier weight on attitude
                + r_smoothness
                + 0.10                 # alive bonus
                + r_streak             # capped "stay-alive longer" side bonus
            )

        # Track sustained instability — terminate if drone stays unstable
        # for `UNSTABLE_WINDOW` consecutive steps so the agent doesn't burn
        # the rest of the episode flailing on its back.
        self.unstable_window.append(is_unstable)
        if len(self.unstable_window) > UNSTABLE_WINDOW:
            self.unstable_window.popleft()
        unstable_too_long = (
            len(self.unstable_window) == UNSTABLE_WINDOW
            and all(self.unstable_window)
        )

        self.out_of_bounds_window.append(not self.__is_within_bounds(tel))
        if len(self.out_of_bounds_window) > OOB_WINDOW:
            self.out_of_bounds_window.popleft()
        out_of_bounds_too_long = (
            len(self.out_of_bounds_window) == OOB_WINDOW
            and all(self.out_of_bounds_window)
        )

        terminated = unstable_too_long or out_of_bounds_too_long
        truncated = self._elapsed_steps >= 300

        if terminated:
            reward = CRASH_PENALTY

        self.current_obs = obs

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self._initialized:
            self.__quit_game()
            self.transmitter.close()
        cv2.destroyAllWindows()


register(id="Liftoff-hover-v0", entry_point=LiftoffHoverEnvUInput, max_episode_steps=300)
