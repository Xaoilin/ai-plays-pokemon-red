from pyboy import PyBoy


if __name__ == '__main__':
    # Load PyBoy with the ROM file
    pyboy = PyBoy('PokemonRed.gb',
                  window_type="SDL2")  # Use "headless" for non-interactive mode. Remove for playing the game with GUI.
    pyboy.load_state(open("has_pokedex_nballs.state", "rb"))

    counter = 0

    while True:
        pyboy.tick()  # Runs one tick of the game
        counter += 1
        # Add your game playing code here if you want to reach a certain state in the game before saving.

        # Save the game state

        if counter % 20 == 0:
            pyboy.save_state(open("pallet_town_grass.state", "wb"))

    # Properly stop PyBoy
    # pyboy.stop()