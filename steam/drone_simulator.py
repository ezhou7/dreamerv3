import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyautogui
import mss
import cv2
import time
from PIL import Image


class DroneSimulator(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, screen_region=None, action_meanings=None):
        super().__init__()
        # Example action space: 0=do nothing, 1=left, 2=right, 3=jump
        self.action_space = spaces.Discrete(4)
        # Observation: 84x84x3 RGB pixels (Atari-style, grayscale optional)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self.render_mode = render_mode
        # Define screen region to capture (x, y, width, height) - adjust via trial/error
        self.screen_region = screen_region or {'top': 100, 'left': 100, 'width': 800, 'height': 600}
        self.sct = mss.mss()  # Screenshot tool

        # Action mappings (adjust keys for your game)
        self.action_meanings = action_meanings or ['NOOP', 'LEFT', 'RIGHT', 'JUMP']
        self.action_keys = {0: None, 1: 'a', 2: 'd', 3: 'space'}

        self.current_obs = None
        self.prev_hash = None  # For simple progress reward

    def _get_obs(self):
        # Capture screen
        screenshot = self.sct.grab(self.screen_region)
        img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        img = img.resize((84, 84))  # Downscale
        img = np.array(img)  # Shape: (84, 84, 3)
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Restart game: Simulate ESC -> New Game (game-specific; automate menu navigation)
        pyautogui.press('esc')  # Example: Open menu
        time.sleep(0.5)
        pyautogui.press('n')  # Example: 'N' for new game
        time.sleep(2)  # Wait for load
        self.prev_hash = None
        obs = self._get_obs()
        self.current_obs = obs
        return obs, {}

    def step(self, action):
        # Execute action
        key = self.action_keys[action]
        if key:
            pyautogui.keyDown(key)
            time.sleep(0.05)  # Hold briefly
            pyautogui.keyUp(key)

        time.sleep( 1 /30)  # ~30 FPS step

        obs = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Simple reward: Detect horizontal progress (improve with CV: detect player x-pos or score)
        current_hash = hash(obs.tobytes())
        if self.prev_hash is not None:
            reward = 1 if current_hash != self.prev_hash else -0.01  # Sparse progress
        self.prev_hash = current_hash

        # Termination: Detect game over (e.g., via pixel check for "DEATH" screen)
        if np.mean(obs[:, :, 0]) > 200:  # Dummy: Red screen = death
            terminated = True

        # Truncation: Max steps (e.g., 1000)
        if self._elapsed_steps >= 1000:  # Track self._elapsed_steps in full impl
            truncated = True

        self.current_obs = obs
        if self.render_mode == 'human':
            cv2.imshow('Game', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # Handled in step for simplicity

    def close(self):
        cv2.destroyAllWindows()
