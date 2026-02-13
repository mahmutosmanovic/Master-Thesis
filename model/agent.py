import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Agent:
    def __init__(self, config):
        self.optimization_hpt = config.model.optimization
        self.update_hpt = config.model.sampling
        self.path = config.model.path
        self.mode = self.setup(config.model.mode)

        self.rollout_dataset = []

    def clear_buffer(self):
        self.rollout_dataset = []

    def add_to_buffer(self, datapoint):
        self.rollout_dataset.append(datapoint)

    def package_buffer(self):
        """
        package the collected trajectories in batches ready for learning
        """
        ...

    def setup(self, mode):
        return mode

    def mode(mode="train"):
        match mode:
            case "train":
                pass
            case "eval":
                self.load(self.path)
            case _:
                raise ValueError(f"Unexpected value: {mode!r}")
        return mode
    
    def load(self, path):
        """
        load model at path, use it for inference
        """
        ...

    def save(self):
        ...

    def policy(self, observation):
        ...        

    def learn(self, obs, action, reward, next_obs, done):
        ...