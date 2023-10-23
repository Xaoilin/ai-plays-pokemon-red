import time
from os.path import exists

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from argparse_pokemon import *
from red_gym_env import RedGymEnv

headless = True


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


# Function to create environments with modified 'headless' parameter based on 'i'
def make_envs(num_cpu, env_config):
    envs = []
    for i in range(num_cpu):
        # Modify 'headless' based on the value of 'i'
        # if i == 0:
        #     env_config['headless'] = False
        # else:
        env_config['headless'] = headless

        # Use a copy of the configuration for each environment to avoid side effects
        env = make_env(i, env_config.copy())
        envs.append(env)
    return envs


if __name__ == '__main__':
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

    number_of_steps = 1024 * 4
    sess_path = f'ssj_{str(uuid.uuid4())[:8]}'
    args = get_args('run_baseline_parallel.py', headless=headless, ep_length=number_of_steps, sess_path=sess_path)

    env_config = {
        'headless': headless, 'save_final_state': True, 'early_stop': False,
        'action_freq': 16, 'init_state': '../has_pokedex_nballs.state', 'max_steps': number_of_steps,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
    }

    env_config = change_env(env_config, args)

    num_cpu = 16  # 64 #46  # Also sets the number of episodes per training iteration
    # env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    # env = [make_env(i, env_config) for i in range(num_cpu)]
    # env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # Create a list of environments
    env_list = make_envs(num_cpu, env_config)

    # Pass the list of environments to SubprocVecEnv
    env = SubprocVecEnv(env_list)

    checkpoint_callback = CheckpointCallback(save_freq=number_of_steps, save_path=sess_path,
                                             name_prefix='poke')
    # env_checker.check_env(env)
    learn_steps = 2348974787  # Not really used tbh, the model keeps iterating even after this max limit specified. Can be removed.
    # file_name = 'ssj_d29ef1db/poke_229376_steps' #'session_e41c9eff/poke_250871808_steps'
    file_name = 'do_not_detect_file_name'  # 'session_e41c9eff/poke_250871808_steps'

    # 'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = number_of_steps
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = number_of_steps
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        t1 = time.time()
        # model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)
        model = PPO('MultiInputPolicy', env, verbose=1, n_steps=number_of_steps, batch_size=512, n_epochs=1,
                    gamma=0.999, device='cuda')

    for i in range(learn_steps):
        model.learn(total_timesteps=(number_of_steps) * num_cpu * 1000, callback=checkpoint_callback)
