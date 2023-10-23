import sys
from pyboy import PyBoy


class MemoryAddresses:
    pyboy = None

    def __init__(self, config):
        head = 'headless' if config['headless'] else 'SDL2'

        MemoryAddresses.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

    def read_current_map_instance(self):
        map_instance_addresses = [
            0xD362,  # Player's X Coords
            0xD361,  # Player's Y Coords
            0xD35E,  # Current Map Number
        ]

        current_map_instance = [self.read_m(addr) for addr in map_instance_addresses]
        return current_map_instance

    def read_current_hp_as_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256 * 256 * self.read_m(start_add) + 256 * self.read_m(start_add + 1) + self.read_m(start_add + 2)

    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) +
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == '1'

    def read_player_coordinates(self):
        x_coord = self.read_m(0xD361)
        y_coord = self.read_m(0xD362)
        return x_coord, y_coord

    def read_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0)  # subtract starting pokemon level

    def read_bag_items(self):
        # Specific memory addresses for each item and quantity in the bag
        item_addresses = [
            0xD31D,  # Total Items
            0xD31E,  # Item 1
            0xD31F,  # Item 1 Quantity
            0xD320,  # Item 2
            0xD321,  # Item 2 Quantity
            0xD322,  # Item 3
            0xD323,  # Item 3 Quantity
            0xD324,  # Item 4
            0xD325,  # Item 4 Quantity
            0xD326,  # Item 5
            0xD327,  # Item 5 Quantity
            0xD328,  # Item 6
            0xD329,  # Item 6 Quantity
            0xD32A,  # Item 7
            0xD32B,  # Item 7 Quantity
            0xD32C,  # Item 8
            0xD32D,  # Item 8 Quantity
            0xD32E,  # Item 9
            0xD32F,  # Item 9 Quantity
            0xD330,  # Item 10
            0xD331,  # Item 10 Quantity
            0xD332,  # Item 11
            0xD333,  # Item 11 Quantity
            0xD334,  # Item 12
            0xD335,  # Item 12 Quantity
            0xD336,  # Item 13
            0xD337,  # Item 13 Quantity
            0xD338,  # Item 14
            0xD339,  # Item 14 Quantity
            0xD33A,  # Item 15
            0xD33B,  # Item 15 Quantity
            0xD33C,  # Item 16
            0xD33D,  # Item 16 Quantity
            0xD33E,  # Item 17
            0xD33F,  # Item 17 Quantity
            0xD340,  # Item 18
            0xD341,  # Item 18 Quantity
            0xD342,  # Item 19
            0xD343,  # Item 19 Quantity
            0xD344,  # Item 20
            0xD345,  # Item 20 Quantity
            0xD346  # Item End of List
        ]

        bag_items = [self.read_m(addr) for addr in item_addresses]
        return bag_items

    def read_levels_reward(self):
        level_sum = self.read_levels_sum()
        return level_sum

    def read_knn_reward(self):
        pre_rew = 0.004
        post_rew = 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def read_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_pokemart_items(self):
        # Specific memory addresses for each item in the Pokemart
        item_addresses = [
            0xCF7C,  # Item 1
            0xCF7D,  # Item 2
            0xCF7E,  # Item 3
            0xCF7F,  # Item 4
            0xCF80,  # Item 5
            0xCF81,  # Item 6
            0xCF82,  # Item 7
            0xCF83,  # Item 8
            0xCF84,  # Item 9
            0xCF85  # Item 10
        ]

        pokemart_items = [self.read_m(addr) for addr in item_addresses]
        return pokemart_items

    def read_pokemon_in_party(self):
        # Specific memory addresses for each Pokémon in the party
        party_addresses = [
            0xD163,  # Number of Pokémon in party
            0xD164,  # Pokémon 1
            0xD165,  # Pokémon 2
            0xD166,  # Pokémon 3
            0xD167,  # Pokémon 4
            0xD168,  # Pokémon 5
            0xD169  # Pokémon 6
        ]

        pokemon_party = [self.read_m(addr) for addr in party_addresses]
        return pokemon_party

    def read_party_exp(self):
        pokemon_exp = [
            0xD17B, 0xD1A7, 0xD1D3, 0xD1FF, 0xD22B, 0xD257
        ]

        return [(self.read_m(addr)) for addr in pokemon_exp]

    def read_pokemon1_stats(self):
        # Specific memory addresses for various stats of PK1
        pk1_addresses = [
            0xD16B,  # Pokémon (Again)
            0xD16C, 0xD16D,  # Current HP
            0xD16F,  # Status (Poisoned, Paralyzed, etc.)
            0xD170,  # Type 1
            0xD171,  # Type 2
            0xD173,  # Move 1
            0xD174,  # Move 2
            0xD175,  # Move 3
            0xD176,  # Move 4
            0xD179, 0xD17A, 0xD17B,  # Experience
            0xD17C, 0xD17D,  # HP EV
            0xD17E, 0xD17F,  # Attack EV
            0xD180, 0xD181,  # Defense EV
            0xD182, 0xD183,  # Speed EV
            0xD184, 0xD185,  # Special EV
            0xD186,  # Attack/Defense IV
            0xD187,  # Speed/Special IV
            0xD188,  # PP Move 1
            0xD189,  # PP Move 2
            0xD18A,  # PP Move 3
            0xD18B,  # PP Move 4
            0xD18C  # Level (actual level)
        ]

        pk1_stats = [self.read_m(addr) for addr in pk1_addresses]
        return pk1_stats

    def read_pokemon2_stats(self):
        # Specific memory addresses for various stats of PK2
        pk2_addresses = [
            0xD197,  # Pokémon
            0xD198, 0xD199,  # Current HP
            0xD19B,  # Status
            0xD19C,  # Type 1
            0xD19D,  # Type 2
            0xD19F,  # Move 1
            0xD1A0,  # Move 2
            0xD1A1,  # Move 3
            0xD1A2,  # Move 4
            0xD1A5, 0xD1A6, 0xD1A7,  # Experience
            0xD1A8, 0xD1A9,  # HP EV
            0xD1AA, 0xD1AB,  # Attack EV
            0xD1AC, 0xD1AD,  # Defense EV
            0xD1AE, 0xD1AF,  # Speed EV
            0xD1B0, 0xD1B1,  # Special EV
            0xD1B2,  # Attack/Defense IV
            0xD1B3,  # Speed/Special IV
            0xD1B4,  # PP Move 1
            0xD1B5,  # PP Move 2
            0xD1B6,  # PP Move 3
            0xD1B7,  # PP Move 4
            0xD1B8  # Level (actual)
        ]

        pk2_stats = [self.read_m(addr) for addr in pk2_addresses]
        return pk2_stats

    def read_pokemon3_stats(self):
        # Specific memory addresses for various stats of Pokémon 3
        pokemon3_addresses = [
            0xD1C3,  # Pokémon
            0xD1C4, 0xD1C5,  # Current HP
            0xD1C6,  # 'Level' (not the actual level, see the notes article)
            0xD1C7,  # Status
            0xD1C8,  # Type 1
            0xD1C9,  # Type 2
            0xD1CA,  # Catch rate/Held item (When traded to Generation II)
            0xD1CB,  # Move 1
            0xD1CC,  # Move 2
            0xD1CD,  # Move 3
            0xD1CE,  # Move 4
            0xD1CF, 0xD1D0,  # Trainer ID
            0xD1D1, 0xD1D2, 0xD1D3,  # Experience
            0xD1D4, 0xD1D5,  # HP EV
            0xD1D6, 0xD1D7,  # Attack EV
            0xD1D8, 0xD1D9,  # Defense EV
            0xD1DA, 0xD1DB,  # Speed EV
            0xD1DC, 0xD1DD,  # Special EV
            0xD1DE,  # Attack/Defense IV
            0xD1DF,  # Speed/Special IV
            0xD1E0,  # PP Move 1
            0xD1E1,  # PP Move 2
            0xD1E2,  # PP Move 3
            0xD1E3,  # PP Move 4
            0xD1E4,  # Level (actual level)
            0xD1E5, 0xD1E6,  # Max HP
            0xD1E7, 0xD1E8,  # Attack
            0xD1E9, 0xD1EA,  # Defense
            0xD1EB, 0xD1EC,  # Speed
            0xD1ED, 0xD1EE  # Special
        ]

        pokemon3_stats = [self.read_m(addr) for addr in pokemon3_addresses]
        return pokemon3_stats

    def read_pokemon4_stats(self):
        # Specific memory addresses for various stats of Pokémon 4
        pokemon4_addresses = [
            0xD1EF,  # Pokémon
            0xD1F0, 0xD1F1,  # Current HP
            0xD1F2,  # 'Level' (not the actual level, see the notes article)
            0xD1F3,  # Status
            0xD1F4,  # Type 1
            0xD1F5,  # Type 2
            0xD1F6,  # Catch rate/Held item (When traded to Generation II)
            0xD1F7,  # Move 1
            0xD1F8,  # Move 2
            0xD1F9,  # Move 3
            0xD1FA,  # Move 4
            0xD1FB, 0xD1FC,  # Trainer ID
            0xD1FD, 0xD1FE, 0xD1FF,  # Experience
            0xD200, 0xD201,  # HP EV
            0xD202, 0xD203,  # Attack EV
            0xD204, 0xD205,  # Defense EV
            0xD206, 0xD207,  # Speed EV
            0xD208, 0xD209,  # Special EV
            0xD20A,  # Attack/Defense IV
            0xD20B,  # Speed/Special IV
            0xD20C,  # PP Move 1
            0xD20D,  # PP Move 2
            0xD20E,  # PP Move 3
            0xD20F,  # PP Move 4
            0xD210,  # Level (actual level)
            0xD211, 0xD212,  # Max HP
            0xD213, 0xD214,  # Attack
            0xD215, 0xD216,  # Defense
            0xD217, 0xD218,  # Speed
            0xD219, 0xD21A  # Special
        ]

        pokemon4_stats = [self.read_m(addr) for addr in pokemon4_addresses]
        return pokemon4_stats

    def read_pokemon5_stats(self):
        # Specific memory addresses for various stats of Pokémon 5
        pokemon5_addresses = [
            0xD21B,  # Pokémon
            0xD21C, 0xD21D,  # Current HP
            0xD21E,  # 'Level' (not the actual level, see the notes article)
            0xD21F,  # Status
            0xD220,  # Type 1
            0xD221,  # Type 2
            0xD222,  # Catch rate/Held item (When traded to Generation II)
            0xD223,  # Move 1
            0xD224,  # Move 2
            0xD225,  # Move 3
            0xD226,  # Move 4
            0xD227, 0xD228,  # Trainer ID
            0xD229, 0xD22A, 0xD22B,  # Experience
            0xD22C, 0xD22D,  # HP EV
            0xD22E, 0xD22F,  # Attack EV
            0xD230, 0xD231,  # Defense EV
            0xD232, 0xD233,  # Speed EV
            0xD234, 0xD235,  # Special EV
            0xD236,  # Attack/Defense IV
            0xD237,  # Speed/Special IV
            0xD238,  # PP Move 1
            0xD239,  # PP Move 2
            0xD23A,  # PP Move 3
            0xD23B,  # PP Move 4
            0xD23C,  # Level (actual level)
            0xD23D, 0xD23E,  # Max HP
            0xD23F, 0xD240,  # Attack
            0xD241, 0xD242,  # Defense
            0xD243, 0xD244,  # Speed
            0xD245, 0xD246  # Special
        ]

        pokemon5_stats = [self.read_m(addr) for addr in pokemon5_addresses]
        return pokemon5_stats

    def read_pokemon6_stats(self):
        # Specific memory addresses for various stats of Pokémon 6
        pokemon6_addresses = [
            0xD247,  # Pokémon
            0xD248, 0xD249,  # Current HP
            0xD24A,  # 'Level' (not the actual level, see the notes article)
            0xD24B,  # Status
            0xD24C,  # Type 1
            0xD24D,  # Type 2
            0xD24E,  # Catch rate/Held item (When traded to Generation II)
            0xD24F,  # Move 1
            0xD250,  # Move 2
            0xD251,  # Move 3
            0xD252,  # Move 4
            0xD253, 0xD254,  # Trainer ID
            0xD255, 0xD256, 0xD257,  # Experience
            0xD258, 0xD259,  # HP EV
            0xD25A, 0xD25B,  # Attack EV
            0xD25C, 0xD25D,  # Defense EV
            0xD25E, 0xD25F,  # Speed EV
            0xD260, 0xD261,  # Special EV
            0xD262,  # Attack/Defense IV
            0xD263,  # Speed/Special IV
            0xD264,  # PP Move 1
            0xD265,  # PP Move 2
            0xD266,  # PP Move 3
            0xD267,  # PP Move 4
            0xD268,  # Level (actual level)
            0xD269, 0xD26A,  # Max HP
            0xD26B, 0xD26C,  # Attack
            0xD26D, 0xD26E,  # Defense
            0xD26F, 0xD270,  # Speed
            0xD271, 0xD272  # Special
        ]

        pokemon6_stats = [self.read_m(addr) for addr in pokemon6_addresses]
        return pokemon6_stats

    def read_map_info(self):
        # Specific memory addresses for various map properties
        map_addresses = [
            0xD35B,  # Audio track (See Audio section)
            0xD35C,  # Audio bank (See Audio section)
            0xD35D,  # Controls the map's palette. Usually 0, but is set to 6 when Flash is required.
            0xD35E,  # Current Map Number
            0xD36A, 0xD36B,  # Map's Data
            0xD535  # Grass Tile
        ]

        map_info = [self.read_m(addr) for addr in map_addresses]
        return map_info

    def read_event_flags(self):
        # Specific memory addresses for various event flags
        event_flag_addresses = [
            # Range for Missable Objects Flags
            *range(0xD5A6, 0xD5C6),  # Note: This will include every address from D5A6 to D5C5
            0xD5AB,  # Starters Back?
            0xD5C0,  # 0=Mewtwo appears, 1=Doesn't (See D85F) - specific bit handling needed
            0xD5F3,  # Have Town map?
            0xD60D,  # Have Oak's Parcel?
            0xD700,  # Bike Speed
            0xD70B,  # Fly Anywhere Byte 1
            0xD70C,  # Fly Anywhere Byte 2
            0xD70D,  # Safari Zone Time Byte 1
            0xD70E,  # Safari Zone Time Byte 2
            0xD710,  # Fossilized Pokémon?
            0xD714,  # Position in Air
            0xD72E,  # Did you get Lapras Yet?
            0xD732,  # Debug New Game
            0xD751,  # Fought Giovanni Yet?
            0xD755,  # Fought Brock Yet?
            0xD75E,  # Fought Misty Yet?
            0xD773,  # Fought Lt. Surge Yet?
            0xD77C,  # Fought Erika Yet?
            0xD782,  # Fought Articuno Yet?
            0xD790,  # If bit 7 is set, Safari Game over - specific bit handling needed
            0xD792,  # Fought Koga Yet?
            0xD79A,  # Fought Blaine Yet?
            0xD7B3,  # Fought Sabrina Yet?
            0xD7D4,  # Fought Zapdos Yet?
            0xD7D8,  # Fought Snorlax Yet (Vermilion)
            0xD7E0,  # Fought Snorlax Yet? (Celadon)
            0xD7EE,  # Fought Moltres Yet?
            0xD803,  # Is SS Anne here?
            0xD85F  # Mewtwo can be caught if bit 2 clear - Needs D5C0 bit 1 clear, too
        ]

        event_flags = [self.read_m(addr) for addr in event_flag_addresses]
        return event_flags

    def read_player_battle_statuses(self):
        # Specific memory addresses for battle statuses and related flags
        battle_status_addresses = [
            0xD062,  # Battle Status (Player) - bit-specific statuses
            0xD063,  # Additional battle statuses - bit-specific statuses
            0xD064,  # More battle statuses - bit-specific statuses
            0xD065,  # Stat to double (CPU)
            0xD066,  # Stat to halve (CPU)
            0xD067,  # Battle Status (CPU) - bit-specific statuses, part 1
            0xD068,  # Battle Status (CPU) - bit-specific statuses, part 2
            0xD069,  # Battle Status (CPU) - "Transformed" status, etc.
            0xD06A,  # Multi-Hit Move counter (Player)
            0xD06B,  # Confusion counter (Player)
            0xD06C,  # Toxic counter (Player)
            0xD06D,  # Disable counter (Player) - first part
            0xD06E,  # Disable counter (Player) - second part
            0xD05E,  # Critical Hit / OHKO Flag
            0xD0D8  # Amount of damage attack is about to do
        ]

        battle_statuses = [self.read_m(addr) for addr in battle_status_addresses]
        return battle_statuses

    def read_audio_information(self):
        # Specific memory addresses for audio tracks and channels
        audio_addresses = [
            0xC022,  # Audio track channel 1
            0xC023,  # Audio track channel 2
            0xC024,  # Audio track channel 3
            0xC025,  # Audio track channel 4
            0xD35B,  # Audio track in current map
            0xD35C  # Audio bank in current map
        ]

        audio_status = [self.read_m(addr) for addr in audio_addresses]
        return audio_status

    def read_opponent_battle_status(self):
        # Specific memory addresses for enemy's status in battle
        enemy_battle_addresses = [
            0xCFEA,  # Enemy's Type 1
            0xCFEB,  # Enemy's Type 2
            0xCFF3,  # Enemy's Level
            0xD8A4,  # Pokémon
            0xD8A5,  # Current HP (first part)
            0xD8A6,  # Current HP (second part)
            0xD8A8,  # Status
            0xD8A9,  # Type 1
            0xD8AA,  # Type 2
            0xD8B0,  # Trainer ID (first part)
            0xD8B1  # Trainer ID (second part)
        ]

        enemy_status = [self.read_m(addr) for addr in enemy_battle_addresses]
        return enemy_status

    def read_owned_pokedex_entries(self):
        # Specific memory addresses for Pokedex entries
        pokedex_addresses = [
            # Owned Pokémon entries
            0xD2F7, 0xD2F8, 0xD2F9, 0xD2FA, 0xD2FB, 0xD2FC, 0xD2FD, 0xD2FE, 0xD2FF,
            0xD300, 0xD301, 0xD302, 0xD303, 0xD304, 0xD305, 0xD306, 0xD307, 0xD308,
            0xD309,
        ]

        pokedex_entries = [self.bit_count((self.read_m(addr))) for addr in pokedex_addresses]
        return pokedex_entries

    def read_seen_pokedex_entries(self):
        # Specific memory addresses for Pokedex entries
        pokedex_addresses = [
            # Seen Pokémon entries
            0xD30A, 0xD30B, 0xD30C, 0xD30D, 0xD30E, 0xD30F, 0xD310, 0xD311, 0xD312,
            0xD313, 0xD314, 0xD315, 0xD316, 0xD317, 0xD318, 0xD319, 0xD31A, 0xD31B,
            0xD31C
        ]

        pokedex_entries = [self.read_m(addr) for addr in pokedex_addresses]
        return pokedex_entries

    def read_battle_modifiers(self):
        # Specific memory addresses for Battle stats
        battle_addresses = [
            0xCCD5,  # Number of turns in current battle
            0xCCEE,  # Player move that the enemy disabled
            0xCCEF,  # Enemy move that the player disabled
            0xCD2F,  # Enemy's Pokémon Defense modifier
            0xCD30,  # Enemy's Pokémon Speed modifier
            0xCD31,  # Enemy's Pokémon Special modifier
            0xCD32,  # Enemy's Pokémon Accuracy modifier
            0xCD33,  # Enemy's Pokémon Evasion modifier
            0xCD1B,  # Player's Pokémon Defense modifier
            0xCD1C,  # Player's Pokémon Speed modifier
            0xCD1D,  # Player's Pokémon Special modifier
            0xCD1E,  # Player's Pokémon Accuracy modifier
            0xCD1F  # Player's Pokémon Evasion modifier
        ]

        battle_stats = [self.read_m(addr) for addr in battle_addresses]
        return battle_stats

    def read_game_time(self):
        # Read the memory addresses
        hours = self.read_m(0xDA40) + (self.read_m(0xDA41))  # Assuming read_m reads a single byte and big-endian
        minutes = self.read_m(0xDA42) + (self.read_m(0xDA43))
        seconds = self.read_m(0xDA44)  # Single byte
        frames = self.read_m(0xDA45)  # Single byte

        # Construct a dictionary to hold the time components
        game_time = {
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds,
            'frames': frames,
        }

        return game_time

    def read_current_map_instance(self):
        map_instance_addresses = [
            0xD362,  # Player's X Coords
            0xD361,  # Player's Y Coords
            0xD35E,  # Current Map Number
        ]

        current_map_instance = [self.read_m(addr) for addr in map_instance_addresses]
        return current_map_instance

    def read_m(self, addr):
        return MemoryAddresses.pyboy.get_memory_value(addr)
