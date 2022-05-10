'''
Author: Alex Reichenbach
Date: May 4, 2022
'''

import click
import gym

def create_gym():
    env_name = "neuron_poker-v0"


@click.command()
def main():
    env = create_environment()
    # play_out_hand(env, n_seats)
    train(env, memory_size)


if __name__ == "__main__":
    main()