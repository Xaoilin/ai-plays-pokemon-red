import json
import uuid
from collections import Counter
from math import floor
from pathlib import Path

import hnswlib
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pandas as pd
from einops import rearrange
from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from skimage.transform import resize

from memory_addresses import *


class RedGymEnv(Env):

    def __init__(
            self, config=None):

        self.memory_addresses = MemoryAddresses(config)

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320  # 1000
        self.headless = config['headless']
        self.num_elements = 20000  # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.one_time_badge_reward = 0
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.all_runs = []
        self.visited_maps = []
        self.visited_audio = []
        self.recent_maps_queue = []
        self.recent_audio_queue = []
        self.last_party_exp = 0
        self.total_party_exp_gained: float = 0
        self.total_exploration_reward = 0
        self.unique_map_instances = set()
        self.unique_caught_pokemon = set()

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PASS
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8),
            'player_coords': self.create_box(0, 255, (2,), np.int32),
            'bag': self.create_box(0, 255, (42,), np.int32),
            'pokemart_items': self.create_box(0, 255, (10,), np.int32),
            'party_pokemon': self.create_box(0, 255, (7,), np.int32),
            'player_first_pokemon': self.create_box(0, 255, (30,), np.int32),
            'player_second_pokemon': self.create_box(0, 255, (30,), np.int32),
            'player_third_pokemon': self.create_box(0, 255, (44,), np.int32),
            'player_fourth_pokemon': self.create_box(0, 255, (44,), np.int32),
            'player_fifth_pokemon': self.create_box(0, 255, (44,), np.int32),
            'player_sixth_pokemon': self.create_box(0, 255, (44,), np.int32),
            'map_details': self.create_box(0, 255, (7,), np.int32),
            'events_progress_flags': self.create_box(0, 255, (61,), np.int32),
            'player_battle_status': self.create_box(0, 255, (15,), np.int32),
            'audio_information': self.create_box(0, 255, (6,), np.int32),
            'opponent_battle_status': self.create_box(0, 255, (11,), np.int32),
            'pokedex_owned': self.create_box(0, 255, (19,), np.int32),
            'pokedex_seen': self.create_box(0, 255, (19,), np.int32),
            'battle_modifiers': self.create_box(0, 255, (13,), np.int32),
        })

        self.pyboy = MemoryAddresses.pyboy

        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0)
        self.reset()

    def create_box(self, low, high, shape, dtype):
        return spaces.Box(low=low,  # or appropriate min value
                          high=high,  # or appropriate max value
                          shape=shape,
                          dtype=dtype)

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_knn()

        self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)

        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0],
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.visited_maps = []
        self.visited_audio = []
        self.total_party_exp_gained = 0
        self.total_exploration_reward = 0
        # state = {'screen': self.render(), 'player_coords_x': 0, 'player_coords_y': 0,
        #          'pokemon_items': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}

        state = {
            'screen': self.render(),
            'player_coords': np.array([i for i in range(0, 2)]),
            'bag': np.array([i for i in range(0, 42)]),
            'pokemart_items': np.array([i for i in range(0, 10)]),
            'party_pokemon': np.array([i for i in range(0, 7)]),
            'player_first_pokemon': np.array([i for i in range(0, 30)]),
            'player_second_pokemon': np.array([i for i in range(0, 30)]),
            'player_third_pokemon': np.array([i for i in range(0, 44)]),
            'player_fourth_pokemon': np.array([i for i in range(0, 44)]),
            'player_fifth_pokemon': np.array([i for i in range(0, 44)]),
            'player_sixth_pokemon': np.array([i for i in range(0, 44)]),
            'map_details': np.array([i for i in range(0, 7)]),
            'events_progress_flags': np.array([i for i in range(0, 61)]),
            'player_battle_status': np.array([i for i in range(0, 15)]),
            'audio_information': np.array([i for i in range(0, 6)]),
            'opponent_battle_status': np.array([i for i in range(0, 11)]),
            'pokedex_owned': np.array([i for i in range(0, 19)]),
            'pokedex_seen': np.array([i for i in range(0, 19)]),
            'battle_modifiers': np.array([i for i in range(0, 13)]),
        }

        return state, {}

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim)  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3),
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(),
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render

    def step(self, action):
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
                   frame_start:frame_start + self.output_shape[0], ...].flatten().astype(np.float32)

        self.update_frame_knn_index(obs_flat)

        self.update_heal_reward()

        new_reward, new_prog = self.update_reward()

        self.last_health = self.memory_addresses.read_current_hp_as_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()

        self.save_and_print_info(step_limit_reached, obs_memory)

        self.step_count += 1

        current_map_info = self.memory_addresses.read_map_info()
        current_audio_info = self.memory_addresses.read_audio_information()

        self.recent_maps_queue.append(current_map_info)

        if len(self.recent_maps_queue) >= 10:
            self.recent_maps_queue.pop(0)

        # print(self.recent_maps_queue)
        # Convert the inner lists to tuples so they can be counted
        tuples = [tuple(lst) for lst in self.recent_maps_queue]

        # Create a Counter from the tuples
        counter = Counter(tuples)

        # Find the most common tuple
        most_common_tuple = counter.most_common(
            1)  # Returns a list with a tuple of the most common element and its count

        # print(most_common_tuple)  # Output will be [((1, 2, 3), 3)] because [1, 2, 3] appears most

        # If you want just the most common list and not its count, you can do:
        most_common_map = list(most_common_tuple[0][0])  # [1, 2, 3]
        # print(most_common_map)

        # Check if the current map has already been visited
        if most_common_map not in self.visited_maps:
            # If it's a new map, add its information to the visited_maps list
            self.visited_maps.append(most_common_map)

        if current_audio_info not in self.visited_audio:
            # If it's a new map, add its information to the visited_maps list
            self.visited_audio.append(current_audio_info)

        state = {
            'screen': obs_memory,
            'player_coords': np.array(self.memory_addresses.read_player_coordinates()),
            'bag': np.array(self.memory_addresses.read_bag_items()),
            'pokemart_items': np.array(self.memory_addresses.read_pokemart_items()),
            'party_pokemon': np.array(self.memory_addresses.read_pokemon_in_party()),
            'player_first_pokemon': np.array(self.memory_addresses.read_pokemon1_stats()),
            'player_second_pokemon': np.array(self.memory_addresses.read_pokemon2_stats()),
            'player_third_pokemon': np.array(self.memory_addresses.read_pokemon3_stats()),
            'player_fourth_pokemon': np.array(self.memory_addresses.read_pokemon4_stats()),
            'player_fifth_pokemon': np.array(self.memory_addresses.read_pokemon5_stats()),
            'player_sixth_pokemon': np.array(self.memory_addresses.read_pokemon6_stats()),
            'map_details': np.array(self.memory_addresses.read_map_info()),
            'events_progress_flags': np.array(self.memory_addresses.read_event_flags()),
            'player_battle_status': np.array(self.memory_addresses.read_player_battle_statuses()),
            'audio_information': np.array(self.memory_addresses.read_audio_information()),
            'opponent_battle_status': np.array(self.memory_addresses.read_opponent_battle_status()),
            'pokedex_owned': np.array(self.memory_addresses.read_owned_pokedex_entries()),
            'pokedex_seen': np.array(self.memory_addresses.read_seen_pokedex_entries()),
            'battle_modifiers': np.array(self.memory_addresses.read_battle_modifiers()),
        }

        return state, new_reward * 0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def append_agent_stats(self, action):
        x_pos = self.memory_addresses.read_m(0xD362)
        y_pos = self.memory_addresses.read_m(0xD361)
        map_n = self.memory_addresses.read_m(0xD35E)
        levels = [self.memory_addresses.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        self.agent_stats.append({
            'step': self.step_count,
            'x': x_pos,
            'y': y_pos,
            'map': map_n,
            'last_action': action,
            'pcount': self.memory_addresses.read_m(0xD163),
            'levels': levels,
            'ptypes': self.memory_addresses.read_pokemon_in_party(),
            'hp': self.memory_addresses.read_current_hp_as_fraction(),
            'frames': self.knn_index.get_current_count(),
            'deaths': self.died_count,
            'badge': self.memory_addresses.read_badges(),
            'event': self.progress_reward['event'],
            'unique_catching': self.progress_reward['unique_catching'],
            'visited_maps': self.progress_reward['visited_maps'],
            'visited_audio': self.progress_reward['visited_audio'],
            'party_exp': self.progress_reward['party_exp'],
            'healr': self.total_healing_rew,
            'player_first_pokemon': np.array(self.memory_addresses.read_pokemon1_stats()),
            'player_second_pokemon': np.array(self.memory_addresses.read_pokemon2_stats()),
            'bag': np.array(self.memory_addresses.read_bag_items()),
            'map_details': np.array(self.memory_addresses.read_map_info()),
            'audio_information': np.array(self.memory_addresses.read_audio_information()),
            'events_progress_flags': np.array(self.memory_addresses.read_event_flags()),
            'player_coords': np.array(self.memory_addresses.read_player_coordinates()),
            'time': self.memory_addresses.read_game_time()
        })

    def update_frame_knn_index(self, frame_vec):

        if self.memory_addresses.read_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k=1)
            if distances[0] > self.similar_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(
            [val for _, val in self.progress_reward.items()])  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.memory_addresses.read_current_hp_as_fraction() > 0:
            # print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')

        self.total_reward = new_total
        return (new_step,
                (new_prog[0] - old_prog[0],
                 new_prog[1] - old_prog[1],
                 new_prog[2] - old_prog[2])
                )

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog['level'] * 100, self.memory_addresses.read_current_hp_as_fraction() * 2000, prog['visited_maps'] * 1)

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val):
            col_steps = self.col_steps
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory

        level, hp, visited_maps = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(visited_maps)
        ), axis=-1)

        if self.memory_addresses.read_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory,
            '(w h) c -> h w c',
            h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'),
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'),
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'),
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

    # self.last_health is essentially live health
    # the below function does this: check if current health is greater than previously recorded 'live' health
    # if it is, it means we healed, reward them
    def update_heal_reward(self):
        cur_health = self.memory_addresses.read_current_hp_as_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1

    def get_all_events_reward(self):
        return max(
            sum([self.memory_addresses.bit_count(self.memory_addresses.read_m(i)) for i in range(0xD747, 0xD886)]) - 13,
            0)

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(0xD163)
        poke_xps = [self.read_triple(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(0xD747, 0xD886)])
        oak_parcel = self.read_bit(0xD74E, 1) 
        oak_pokedex = self.read_bit(0xD74B, 5)
        opponent_level = self.read_m(0xCFF3)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(0xD89C)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''

        # if self.get_badges() == 1 and self.one_time_badge_reward == 0:
        #     steps_range = (self.max_steps - self.step_count)
        #     if steps_range <= 1500:
        #         self.one_time_badge_reward = 6
        #     else:
        #         self.one_time_badge_reward = steps_range / 250

        # check if player is in battle
        # if not, check both x and y coords, and if they haven't changed in x steps (e.g. 30)
        #

        # print(self.get_pokemon1_exp())
        current_party_exp = sum(self.memory_addresses.read_party_exp())

        if current_party_exp != self.last_party_exp:
            exp_gained_reward = 0.25
            self.total_party_exp_gained += exp_gained_reward

        current_map_instance = tuple(self.memory_addresses.read_current_map_instance())

        if current_map_instance not in self.unique_map_instances:
            self.unique_map_instances.add(current_map_instance)
            exploration_gain_reward = 0.01
            self.total_exploration_reward += exploration_gain_reward

        state_scores = {
            # 'reset_count': self.reset_count,
            'event': self.update_max_event_rew() * 2,
            # 'party_xp': 0.1*sum(poke_xps),
            'unique_catching': (sum(self.memory_addresses.read_owned_pokedex_entries()) * 1.5) - 1.5,
            'level': self.memory_addresses.read_levels_reward() * 0,
            # 'unique_catching': self.get_levels_reward() * 1,
            'heal': self.total_healing_rew,
            'op_lvl': self.update_max_op_level(),
            'dead': 0 * self.died_count,
            'badge': (self.memory_addresses.read_badges() * 3),
            'visited_maps': len(self.visited_maps) * 0.4,  # Scaled Equation to make maps worth more as game progresses
            'visited_audio': len(self.visited_audio) * 0,
            # Scaled Equation to make maps worth more as game progresses
            'party_exp': self.total_party_exp_gained,
            'explore': self.total_exploration_reward,
            # 'badge': self.get_badges() * 2,
            # 'op_poke': self.max_opponent_poke * 800,
            # 'money': money * 3,
            # 'seen_poke': seen_poke_count * 400,
            # 'explore': self.get_knn_reward() * 0.1
        }

        self.last_party_exp = current_party_exp

        return state_scores

    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'),
            self.render(reduce_res=False))

    def update_max_op_level(self):
        opponent_level = max(
            [self.memory_addresses.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]) - 5
        # if opponent_level >= 7:
        #    self.save_screenshot('highlevelop')
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
