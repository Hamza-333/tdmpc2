import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common import math
from envs import make_env
from tdmpc2 import TDMPC2
import time
import matplotlib.pyplot as plt
import os
torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task evaluation)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `checkpoint`: path to model checkpoint to load
        `eval_episodes`: number of episodes to evaluate on per task (default: 10)
        `save_video`: whether to save a video of the evaluation (default: True)
        `seed`: random seed (default: 1)
    
    See config.yaml for a full list of args.
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

    if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
        print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
        print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
    
    # Evaluate
    if cfg.multitask:
        print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
    else:
        print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))

    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)

    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]

    if cfg.eval_actions:
        errors_per_step, rewards_per_step_actual, rewards_per_step_imagined, action_counts = eval_actions(agent, env, cfg, penalty=0)

        plot_rewards(rewards_per_step_actual.sum(axis=0)/(action_counts + 1e-10), rewards_per_step_imagined.sum(axis=0) / (action_counts + 1e-10), cfg, file_name=f'rewards_comparison_{cfg.task}_{cfg.planner}.png')
        print("Imagined Rewards:", rewards_per_step_imagined.sum(axis=0)/(action_counts + 1e-10))
        print("Actual Rewards:", rewards_per_step_actual.sum(axis=0)/(action_counts + 1e-10))
        print("Mean planned errors per step:", errors_per_step.sum(axis=0) / action_counts)
        print("Horizon Counts", action_counts)
        return
    MAX_ACTIONS = 3
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None

        ep_rewards, ep_successes = [], []
        start_time = time.time()
        total_num_actions = []
        for i in range(cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0

            if cfg.save_video:
                frames = [env.render()]
            num_actions = []
            while not done:
                ### Changed this to log planned errors
                actions = agent.act(obs, t0=t == 0, task=task_idx, eval_mode=True)

                # obs, reward, done, info = env.step(actions[0].clamp(-1,1))
                num_actions.append(len(actions))
                for t in range(1):
                    obs, reward, done, info = env.step(actions[t].clamp(-1,1))
                    ep_reward += reward
                    t += 1
                    if done:
                        break

                # ep_reward += reward
                # t += 1

                if cfg.save_video:
                    frames.append(env.render())

            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            total_num_actions.append(num_actions)

            if cfg.save_video:
                imageio.mimsave(
                    os.path.join(video_dir, f'{task}-{i}-{cfg.planner}.mp4'),
                    frames,
                    fps=15
                )
        for i in range(len(total_num_actions)):
            print(f"Actions: {total_num_actions[i]} ---- Reward: {ep_rewards[i]} ---- Success: {ep_successes[i]}")
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)
        total_time = time.time() - start_time


        if cfg.multitask:
            scores.append(ep_successes * 100 if task.startswith('mw-') else ep_rewards / 10)

        print(colored(
            f'  {task:<22}'
            f'\tR: {ep_rewards:.01f}  '
            f'\tS: {ep_successes:.02f}'
            f'\tTotal Time: {total_time:.02f}'
            f'\tAvg.Time per episode: {total_time / cfg.eval_episodes:.02f}',
            'yellow'
        ))

    if cfg.multitask:
        print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))

def eval_actions(agent, env, cfg, penalty):
    
    errors_per_step = np.zeros((100, cfg.horizon))
    rewards_per_step_actual = np.zeros((100, cfg.horizon))
    rewards_per_step_imagined = np.zeros((100, cfg.horizon))
    action_counts = np.zeros(cfg.horizon)

    for i in range(100):
        obs, done, ep_reward, t = env.reset(task_idx=None), False, 0, 0
        z = agent.model.encode(obs.to(agent.device), task=None)
        errors_episode = []
        
        actions = agent.act(obs, t0=True, task=None, eval_mode=True)
        for t, a in enumerate(actions):
            action_counts[t] += 1

            # Compute imagined reward before updating state
            rewards_per_step_imagined[i, t] = math.two_hot_inv(agent.model.reward(z.to(agent.device), a.clamp(-1, 1).to(agent.device), task=None), agent.cfg)
            
            z = agent.model.next(z.to(agent.device), a.clamp(-1, 1).to(agent.device), task=None)
            
            new_obs, reward, done, info = env.step(a.clamp(-1, 1))

            encoded_new_obs = agent.model.encode(new_obs.to(agent.device), task=None)
            errors_per_step[i, t] = torch.norm(z - encoded_new_obs, dim=-1).cpu().item()

            rewards_per_step_actual[i, t] = reward
            if done:
                break
    return errors_per_step, rewards_per_step_actual, rewards_per_step_imagined, action_counts

def plot_results(results, results_corrected, cfg):
    plt.plot(results, label='Planned Errors')
    plt.plot(results_corrected, label='Corrected Planned Errors')
    plt.title('Error Model Performance Over Time')
    plt.xlabel('Horizon')
    plt.ylabel('Mean Planned Error')
    ax = plt.gca()
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    plt.legend()
    plt.savefig(os.path.join(f'{cfg.error_model_plot_dir}/{cfg.task}/plots/', 'planned_errors_diffs_key_turn.png'))
    plt.close()

def plot_rewards(rewards_actual, rewards_imagined, cfg, file_name):
    plt.plot(rewards_actual, label='Actual Rewards')
    plt.plot(rewards_imagined, label='Imagined Rewards')
    plt.title('Actual vs Imagined Rewards Over Time')
    plt.xlabel('Horizon')
    plt.ylabel('Mean Reward')
    plt.legend()
    ax = plt.gca()
    # ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    os.makedirs(f'{cfg.error_model_plot_dir}/{cfg.task}/plots/', exist_ok=True)
    plt.savefig(os.path.join(f'{cfg.error_model_plot_dir}/{cfg.task}/plots/', file_name))
    plt.close()

if __name__ == '__main__':
    evaluate()
