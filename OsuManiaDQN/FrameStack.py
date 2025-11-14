# FrameStack.py
import numpy as np
from gymnasium import spaces
from collections import deque

class FrameStack:
    def __init__(self, env, num_stack):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(num_stack, *old_shape), dtype=np.uint8
        )
        print(f"FrameStack initialized with observation shape: {self.observation_space.shape}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        stacked_obs = np.stack(self.frames, axis=0)
        return stacked_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        stacked_obs = np.stack(self.frames, axis=0)
        return stacked_obs, reward, terminated, truncated, info

    def close(self):
        """Pass through close() to underlying environment."""
        if hasattr(self.env, "close"):
            self.env.close()
