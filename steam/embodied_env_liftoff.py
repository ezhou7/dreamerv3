import numpy as np
import cv2
import subprocess
import time
import embodied

class DroneSimulatorEnv(embodied.Env):
    def __init__(self, render_mode=None):
        self._render_mode = render_mode
        self._elapsed_steps = 0
        self._prev_hash = None
        self._max_steps = 1000
        
        # Placeholder for your gamepad interface
        # self.gamepad = ... 

    @property
    def obs_space(self):
        return {
            'image': embodied.Space(np.uint8, (64, 64, 3)),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, (4,), low=-1.0, high=1.0),
            'reset': embodied.Space(bool),
        }

    def step(self, action):
        # 1. Check for manual reset request from the agent/driver
        if action['reset']:
            return self._reset()

        # 2. Map actions (-1 to 1) to hardware values
        # Index 0: throttle, 1: yaw, 2: pitch, 3: roll
        throttle = int(action['action'][0] * 32767)
        yaw      = int(action['action'][1] * 32767)
        pitch    = int(action['action'][2] * 32767)
        roll     = int(action['action'][3] * 32767)

        self.gamepad.left_joystick(x_value=roll, y_value=pitch)
        self.gamepad.right_joystick(x_value=yaw, y_value=0)
        self.gamepad.right_trigger(value=max(0, throttle // 128))
        self.gamepad.update()

        # Control step frequency (~30 FPS)
        time.sleep(1 / 30)
        self._elapsed_steps += 1

        # 3. Capture Observation
        obs = self._get_obs()

        # 4. Calculate Reward (Sparse progress detection)
        current_hash = hash(obs.tobytes())
        reward = 0.0
        if self._prev_hash is not None:
            # Reward 1 if the screen changed (movement), -0.01 if static
            reward = 1.0 if current_hash != self._prev_hash else -0.01
        self._prev_hash = current_hash

        # 5. Handle Episode Boundaries
        # TERMINATION: Real failure (e.g., Red screen detection)
        is_terminal = np.mean(obs[:, :, 0]) > 200
        
        # TRUNCATION: Time limit reached
        is_truncated = self._elapsed_steps >= self._max_steps
        
        # IS_LAST: End of sequence for any reason
        is_last = is_terminal or is_truncated

        if self._render_mode == 'human':
            self._render_to_screen(obs)

        transition = {
            'image': obs,
            'reward': np.float32(reward),
            'is_first': False,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

        # Automatically reset the internal state if the episode ended
        if is_last:
            # We return the transition marking the end, 
            # the next call to step() will usually be triggered by the driver resetting
            pass 

        return transition

    def _reset(self):
        self._elapsed_steps = 0
        self._prev_hash = None

        # Neutralize Gamepad
        self.gamepad.left_joystick(x_value=0, y_value=0)
        self.gamepad.right_joystick(x_value=0, y_value=0)
        self.gamepad.right_trigger(value=0)
        self.gamepad.update()

        # Trigger In-Game Reset (R key)
        subprocess.run(['ydotool', 'key', 'KEY_R'])
        time.sleep(2) # Wait for game reload

        obs = self._get_obs()
        return {
            'image': obs,
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    def _get_obs(self):
        result = subprocess.run(['grim', '-t', 'ppm', '-'], stdout=subprocess.PIPE)
        # Convert raw PPM to NumPy array
        image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Dreamer prefers RGB
        image = cv2.resize(image, (64, 64))
        return image

    def _render_to_screen(self, obs):
        cv2.imshow('Drone Training View', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def __len__(self):
        return 0

    def close(self):
        cv2.destroyAllWindows()
