import torch
import numpy as np

from tdmpc2.tdmpc2 import TDMPC2
from error_estimator.estimator import ErrorEstimator
from error_estimator.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

class TrainEstimator:

    def __init__(self, cfg, env, logger):
        self.cfg = cfg
        self.env = env
        self.logger = logger
        self.agent = TDMPC2(cfg)
        self.replay_buffer = ReplayBuffer(max_size=cfg.error_model_replay_buffer_size)
        self.error_model = ErrorEstimator(cfg)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.error_model.parameters(), lr=cfg.error_model_lr)
        self.error_model_performance = []

    def gather_data(self):
        obs, reward, done, info = self.env.reset()
        step = 0
        while step <= self.cfg.error_model_data_steps:
            # Evaluate agent periodically
            if step % self.cfg.error_model_eval_freq == 0:
                eval_next = True
            # Sample random actions
            # Not using trained policy for actions as we want diverse data for training error model
            # Using a learned policy will get us similar states
            action = self.env.rand_act()
            new_obs, reward, done, info = self.env.step(action)
            with torch.no_grad():
                encoded_state = self.agent.model.encode(obs, task=None)
                encoded_new_state = self.agent.model.encode(new_obs, task=None)
                imagined_next_state = self.agent.model.next(z=encoded_state, a=action, task=None)
            
            error = self.compute_error(imagined_next_state, encoded_new_state)
            self.replay_buffer.add(encoded_state, action, encoded_new_state, imagined_next_state, error)
            if done:
                obs, reward, done, info = self.env.reset()

            step += 1

        self.logger.finish(self.agent)

    def compute_error(self, imagined_state, real_state):
        error = torch.norm(imagined_state - real_state, p=2, dim=-1)
        return error
    
    def update_error_model(self):
        for i in range(self.cfg.error_model_epochs):
            states, next_states, actions, imagined_next_states, errors = self.replay_buffer.sample(self.cfg.error_model_batch_size)

            predicted_errors = self.error_model(states, actions)
            loss = self.criterion(predicted_errors, errors)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        
        for i in range(self.cfg.error_model_training_iters):
            self.gather_data()
            self.update_error_model()
            if i % self.cfg.error_model_eval_freq == 0:
                self.eval()
                self.plot_performance(i)


    def eval(self):
        """Evaluate error model performance by comparing predicted error to true error while the agent acts in the world"""
        obs, reward, done, info = self.env.reset()
        for i in range(self.cfg.eval_iters):
            action = self.env.rand_act()
            new_obs, reward, done, info = self.env.step(action)
            with torch.no_grad():
                encoded_state = self.agent.model.encode(obs, task=None)
                encoded_new_state = self.agent.model.encode(new_obs, task=None)
                imagined_next_state = self.agent.model.next(z=encoded_state, a=action, task=None)

            error = self.compute_error(imagined_next_state, encoded_new_state)

            predicted_error = self.error_model(encoded_state, action)
            performance = torch.abs(predicted_error - error).item()
            self.error_model_performance.append(performance)

            if i % 10 == 0:
                print(f'Evaluation iteration {i}: predicted error = {predicted_error.item():.4f}, true error = {error.item():.4f}, performance = {performance:.4f}')
            
        
 
    def plot_performance(self, i):
        plt.plot(self.error_model_performance)
        plt.xlabel('Evaluation Iteration')
        plt.ylabel('Absolute Error between Predicted and True Error')
        plt.title('Error Model Performance Over Time')
        plt.savefig(f'./plots/error_model_performance-{i}.png')