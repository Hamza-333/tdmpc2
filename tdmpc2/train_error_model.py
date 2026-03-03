import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from trainer.train_estimator import TrainEstimator
from error_estimator.replay_buffer import ReplayBuffer
from tdmpc2 import TDMPC2
from error_estimator.estimator import ErrorEstimator
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):

	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)


	os.makedirs(f'{cfg.error_model_save_path}/{cfg.task}/', exist_ok=True)
	os.makedirs(f'{cfg.error_model_log_dir}/{cfg.task}/', exist_ok=True)
	os.makedirs(f'{cfg.error_model_plot_dir}/{cfg.task}/plots', exist_ok=True)
	
	trainer_cls = TrainEstimator
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		error_model=ErrorEstimator(cfg),
		replay_buffer=ReplayBuffer(max_size=cfg.error_model_replay_buffer_size),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
