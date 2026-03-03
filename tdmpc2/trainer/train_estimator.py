import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

from common import math

class TrainEstimator:

    def __init__(self, cfg, agent, error_model, replay_buffer, env, logger):
        self.cfg = cfg
        self.env = env
        self.logger = self.setup_logging()
        self.device = torch.device('cuda:0')
        self.agent = agent.to(self.device)
        self.agent.load(self.cfg.checkpoint)
        self.replay_buffer = replay_buffer
        self.error_model = error_model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.error_model.parameters(), lr=cfg.error_model_lr)
        self.error_model_performance = []
        self.orig_distances = []
        self.corrected_distances = []
        self.epsilon = 0.3
    def setup_logging(self):

        logger = logging.getLogger(f'error_model_training_{self.cfg.task}')
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(f'{self.cfg.error_model_log_dir}/{self.cfg.task}.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        return logger

    def gather_data(self):
        obs = self.env.reset()
        step = 0
        print('Gathering data for error model training...')

        while step <= self.cfg.error_model_data_steps:

            # Sample random actions
            # Not using trained policy for actions as we want diverse data for training error model
            # Using a learned policy will get us similar states
            num = np.random.rand()
            if num < self.epsilon:
                action = self.env.rand_act()
            else:
                action = self.agent.act(obs, t0=False, task=None, eval_mode=True)[0]
            with torch.no_grad():
                encoded_state = self.agent.model.encode(obs.to(self.device), task=None)
            obs, reward, done, info = self.env.step(action)
            with torch.no_grad():
                encoded_new_state = self.agent.model.encode(obs.to(self.device), task=None)
                imagined_next_state = self.agent.model.next(z=encoded_state, a=action.to(self.device), task=None)
            
            error = self.compute_error(imagined_next_state, encoded_new_state)
            self.replay_buffer.add(encoded_state.cpu(), encoded_new_state.cpu(), action.cpu(), imagined_next_state.cpu(), error.cpu())
            if done:
                obs = self.env.reset()

            step += 1

        # self.logger.finish(self.agent)

    def compute_error(self, imagined_state, real_state):
        # print(imagined_state.shape, real_state.shape)
        # error = torch.norm(imagined_state - real_state, dim=-1).detach()
        # print(error, error.shape)
        # return self.scale_factor * error
        return imagined_state - real_state
    
    def update_error_model(self):
        print('Updating error model...')
        for i in range(self.cfg.error_model_epochs):
            states, next_states, actions, imagined_next_states, errors = self.replay_buffer.sample(self.cfg.error_model_batch_size)

            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            imagined_next_states = torch.FloatTensor(imagined_next_states).to(self.device)
            # print(errors.shape)
            errors = torch.FloatTensor(errors).to(self.device)
            # print("Mean:", errors.mean(dim=0), "Std:", errors.std(dim=0))
            predicted_errors = (self.error_model(states.to(self.device), actions.to(self.device)))
            # print(predicted_errors.shape, rrors.shape)
            loss = self.criterion(predicted_errors, errors.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        
        for i in range(self.cfg.error_model_training_iters):
            print(f'Error model training iteration {i}...')
            self.gather_data()
            self.update_error_model()
            if i % self.cfg.error_model_eval_freq == 0:
                self.logger.info("============================================")
                self.logger.info(f'Evaluating error model at iteration {i}...')
                self.eval()
                self.logger.info("============================================")
                self.logger.info(f'Evaluating error model on planned rollouts at iteration {i}...')
                self.evaluate_rollouts()
                self.logger.info("============================================")

            if i % (2 * self.cfg.error_model_eval_freq) == 0:
                self.plot_performance(i)
            if i % self.cfg.error_model_save_freq == 0:
                self.error_model.save(f'{self.cfg.error_model_save_path}/{self.cfg.task}/error_model_{i}.pth')


    def eval(self):
        """Evaluate error model performance by comparing predicted error to true error while the agent acts in the world"""
        print('Evaluating error model performance...')
        self.error_model.eval()
        obs = self.env.reset()
        prediction_errors = []
        orig_distances = []
        corrected_distances = []

        for i in range(self.cfg.error_model_eval_iters):

            action = self.agent.act(obs, t0=False, task=None, eval_mode=True)[0]
            encoded_state = self.agent.model.encode(obs.to(self.device), task=None)
            obs, reward, done, info = self.env.step(action)
            with torch.no_grad():
                encoded_new_state = self.agent.model.encode(obs.to(self.device), task=None)
                imagined_next_state = self.agent.model.next(z=encoded_state, a=action.to(self.device), task=None)

            # true correction and predicted correction
            error = self.compute_error(imagined_next_state, encoded_new_state)

            predicted_error = self.error_model(encoded_state, action.to(self.device))
            prediction_error = torch.norm(predicted_error - error).item()
            # Before and after correction
            prediction_errors.append(prediction_error)
            corrected_next_state = imagined_next_state - predicted_error
            orig_distance = torch.norm(imagined_next_state - encoded_new_state).item()
            corrected_distance = torch.norm(corrected_next_state - encoded_new_state).item()
            corrected_distances.append(corrected_distance)
            orig_distances.append(orig_distance)
            if done:
                obs = self.env.reset()

            if i % 10 == 0:
                self.logger.info(f'Evaluation iteration {i}: original distance = {orig_distance:.4f}, corrected distance = {corrected_distance:.4f}')
                self.logger.info(f'Evaluation iteration {i}: true error norm = {torch.norm(error).item():.4f}, predicted error norm = {torch.norm(predicted_error).item():.4f}, prediction error = {prediction_error:.4f}')

        self.error_model_performance.append(np.mean(corrected_distances))

    def evaluate_rollouts(self):
        print('Evaluating error model performance on rollouts...')
        self.error_model.eval()
        for i in range(10):
            obs, done, ep_reward, t = self.env.reset(task_idx=None), False, 0, 0
            z = self.agent.model.encode(obs.to(self.agent.device), task=None)
            
            actions = self.agent.act(obs, t0=True, task=None, eval_mode=True)
            imagined_distances = []
            corrected_distances = []
            real_rewards = []
            imagined_rewards = []
            for t, a in enumerate(actions):
                # Imagined
                error = self.error_model(z.to(self.agent.device), a.to(self.agent.device))
                z = self.agent.model.next(z.to(self.agent.device), a.to(self.agent.device), task=None)
                imagined_reward = math.two_hot_inv(self.agent.model.reward(z.to(self.agent.device), a.to(self.agent.device), task=None), self.cfg)
                imagined_rewards.append(imagined_reward.cpu().item())

                new_obs, reward, done, info = self.env.step(a)
                real_rewards.append(reward.item())

                encoded_new_obs = self.agent.model.encode(new_obs.to(self.agent.device), task=None)
                imagined_distances.append(torch.norm(z - encoded_new_obs, dim=-1).cpu().item())
                # Error Corrected
                z = z - error
                corrected_distances.append(torch.norm(z - encoded_new_obs, dim=-1).cpu().item())

                if done:
                    break
            self.logger.info(f"Distance between imagined and real states: {imagined_distances}")
            self.logger.info(f"Distance between imagined and real states (error corrected): {corrected_distances}")
            self.logger.info(f"Reward - imagined: {imagined_rewards}, actual: {real_rewards}")
            
            print("------------------")
    
    def plot_performance(self, i):
        # create a new figure so previous draws don't persist
        plt.figure()
        plt.clf()
        plt.plot(self.orig_distances, label='Original Distance')
        plt.plot(self.error_model_performance, label='Corrected Distance')
        plt.legend()
        plt.xlabel('Evaluation Iteration')
        plt.ylabel('Absolute Error between Imagined and Actual States')
        plt.title('Error Model Performance Over Time')
        plt.savefig(f'{self.cfg.error_model_plot_dir}/{self.cfg.task}/plots/error_model_performance.png')
        plt.close()