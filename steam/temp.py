# Example liftoff gym env integrating liftoff udp telemetry.
# Only use for reference
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import struct
import mss
import cv2
import time
from pynput.keyboard import Key, Controller

class LiftoffEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.keyboard = Controller()
        
        # 1. Observation Space: Image + Telemetry
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "telemetry": spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        })
        
        # 2. Action Space: [Throttle, Yaw, Pitch, Roll] normalized -1 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # UDP Setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 9001))
        self.sock.settimeout(0.5)
        self.fmt = "<f fff fff ffff fff ffff Bffff" # Matches TelemetryConfiguration.json
        
        # Screen Capture Setup
        self.sct = mss.mss()
        self.monitor = {"top": 40, "left": 0, "width": 800, "height": 600}

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

        # Capture Image
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (64, 64))
        
        return {"image": img, "telemetry": tel}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Simulate 'R' key to reset drone in Liftoff
        self.keyboard.press('r')
        time.sleep(0.1)
        self.keyboard.release('r')
        
        # Wait a moment for the game to reposition the drone
        time.sleep(0.5)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # 1. Send Action to Liftoff (Assuming you have a VJoy or API bridge here)
        # self.bridge.send(action) 

        # 2. Get new state
        obs = self._get_obs()

        # 3. Logic for Terminated (Crash) and Truncated (Time)
        # Simple crash detection: Velocity becomes 0 while throttle is high, or height < threshold
        velocity_mag = np.linalg.norm(obs["telemetry"][3:6])
        altitude = obs["telemetry"][1] # Adjust based on your Y/Z axis config

        terminated = bool(velocity_mag < 0.1 and action[0] > 0.5) # Example crash heuristic
        truncated = False # Handle via TimeLimit wrapper

        # 4. Reward Logic
        reward = velocity_mag * 0.1 # Reward for moving fast
        if terminated: reward = -10.0

        return obs, reward, terminated, truncated, {}

