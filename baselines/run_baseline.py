import time
from os.path import exists

import torch
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker

from argparse_pokemon import *
from red_gym_env import RedGymEnv

sess_path = f'session_{str(uuid.uuid4())[:8]}'

run_steps = 1000
runs_per_update = 6
updates_per_checkpoint = 4

args = get_args('run_baseline.py', ep_length=run_steps, headless=False, sess_path=sess_path)

env_config = {
    'headless': False, 'save_final_state': True, 'early_stop': False,
    'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': run_steps,
    'print_rewards': True, 'save_video': True, 'session_path': sess_path,
    'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
}

env_config = change_env(env_config, args)
env = RedGymEnv(config=env_config)

env_checker.check_env(env)

learn_steps = 40
file_name = 'poke_'  # 'best_12-7/poke_12_b'
inference_only = True

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Check your installation.")

# Set default device to CUDA
if torch.cuda.is_available():
    print("Setting default device to CUDA")
    torch.set_default_device('cuda')

if exists(file_name + '.zip'):
    print('\nloading checkpoint')
    custom_objects = None
    if inference_only:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "n_steps": 10 * 20 * 2048
        }
    model = PPO.load(file_name, env=env, custom_objects=custom_objects)
else:
    t1 = time.time()
    model = PPO('MultiInputPolicy', env, verbose=1, n_steps=run_steps * runs_per_update, batch_size=6000, n_epochs=3,
                gamma=0.98, device="cuda")

for i in range(learn_steps):
    model.learn(total_timesteps=run_steps * runs_per_update * updates_per_checkpoint)
    print(f"Time with cuda : {time.time() - t1:.2f}s")
    model.save(sess_path / Path(file_name + str(i)))
