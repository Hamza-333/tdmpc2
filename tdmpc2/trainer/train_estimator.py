import torch
import numpy as np
import matplotlib.pyplot as plt

class TrainEstimator:

    def __init__(self, cfg, agent, error_model, replay_buffer, env, logger):
        self.cfg = cfg
        self.env = env
        self.logger = logger
        self.device = torch.device('cuda:0')
        self.agent = agent.to(self.device)
        self.replay_buffer = replay_buffer
        self.error_model = error_model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.error_model.parameters(), lr=cfg.error_model_lr)
        self.error_model_performance = []

    def gather_data(self):
        obs = self.env.reset()
        step = 0
        print('Gathering data for error model training...')
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
                encoded_state = self.agent.model.encode(obs.to(self.device), task=None)
                encoded_new_state = self.agent.model.encode(new_obs.to(self.device), task=None)
                imagined_next_state = self.agent.model.next(z=encoded_state, a=action.to(self.device), task=None)
            
            error = self.compute_error(imagined_next_state, encoded_new_state)
            self.replay_buffer.add(encoded_state.cpu(), encoded_new_state.cpu(), action.cpu(), imagined_next_state.cpu(), error.cpu())
            if done:
                obs = self.env.reset()

            step += 1

        # self.logger.finish(self.agent)

    def compute_error(self, imagined_state, real_state):
        error = torch.norm(imagined_state - real_state, p=2, dim=-1)
        return error
    
    def update_error_model(self):
        print('Updating error model...')
        for i in range(self.cfg.error_model_epochs):
            states, next_states, actions, imagined_next_states, errors = self.replay_buffer.sample(self.cfg.error_model_batch_size)

            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            imagined_next_states = torch.FloatTensor(imagined_next_states).to(self.device)
            errors = torch.FloatTensor(errors).to(self.device)

            predicted_errors = self.error_model(states.to(self.device), actions.to(self.device))
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
                self.eval()
            if i % (2 * self.cfg.error_model_eval_freq) == 0:
                self.plot_performance(i)
            if i % self.cfg.error_model_save_freq == 0:
                self.error_model.save(f'{self.cfg.error_model_save_path}/error_model_{i}.pth')


    def eval(self):
        """Evaluate error model performance by comparing predicted error to true error while the agent acts in the world"""
        obs = self.env.reset()
        prediction_errors = []
        for i in range(self.cfg.error_model_eval_iters):
            action = self.env.rand_act()
            new_obs, reward, done, info = self.env.step(action)
            with torch.no_grad():
                encoded_state = self.agent.model.encode(obs.to(self.device), task=None)
                encoded_new_state = self.agent.model.encode(new_obs.to(self.device), task=None)
                imagined_next_state = self.agent.model.next(z=encoded_state, a=action.to(self.device), task=None)

            error = self.compute_error(imagined_next_state, encoded_new_state)

            predicted_error = self.error_model(encoded_state, action.to(self.device))
            prediction_error = torch.abs(predicted_error - error).item()
            prediction_errors.append(prediction_error)
            

            if i % 10 == 0:
                print(f'Evaluation iteration {i}: predicted error = {predicted_error.item():.4f}, true error = {error.item():.4f}, prediction error = {prediction_error:.4f}')
        self.error_model_performance.append(np.mean(prediction_errors))
        
 
    def plot_performance(self, i):
        plt.plot(self.error_model_performance)
        plt.xlabel('Evaluation Iteration')
        plt.ylabel('Absolute Error between Predicted and True Error')
        plt.title('Error Model Performance Over Time')
        plt.savefig(f'{self.cfg.error_model_plot_dir}/error_model_performance-{i}.png')