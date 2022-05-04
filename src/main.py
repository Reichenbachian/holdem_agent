'''
Author: Alex Reichenbach
Date: May 4, 2022
'''

import click
import gym
import holdem


policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def play_out_hand(env, n_seats):
    # reset environment, gather relevant observations
    (player_states, (community_infos, community_cards)) = env.reset()
    (player_infos, player_hands) = zip(*player_states)

    # display the table, cards and all
    env.render(mode='human')

    terminal = False
    while not terminal:
        # play safe actions, check when noone else has raised, call when raised.
        actions = holdem.safe_actions(community_infos, n_seats=n_seats)
        (player_states, (community_infos, community_cards)), rews, terminal, info = env.step(actions)
        env.render(mode='human')

def create_environment(n_seats, initial_stacks):
    # env = gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(2)
    env = holdem.TexasHoldemEnv(n_seats, max_limit=1e9, debug=False)
    # start with 2 players
    for i in range(n_seats):
        env.add_player(i, stack=initial_stacks) # add a player to seat 0 with 2000 "chips"
    return env

def train(env, num_episodes, memory_size):
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(memory_size)

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        (player_states, (community_infos, community_cards)) = env.reset()
        (player_infos, player_hands) = zip(*player_states)

        terminal = False
        while not terminal:
            # Select and perform an action
            action = select_action(state)

            # play safe actions, check when no one else has raised, call when raised.
            actions = holdem.safe_actions(community_infos, n_seats=n_seats)
            (player_states, (community_infos, community_cards)), rews, terminal, info = env.step(actions)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()

@click.command()
@click.option("--n_seats", type=int, default=2)
@click.option("--initial_stacks", type=int, default=2000)
@click.option("--batch_size", type=int, default=128)
@click.option("--gamma", type=float, default=0.999)
@click.option("--eps_start", type=float, default=0.9)
@click.option("--eps_end", type=float, default=0.05)
@click.option("--eps_decay", type=float, default=200)
@click.option("--target_update", type=float, default=10)
@click.option("--num_episodes", type=float, default=100)
@click.option("--memory_size", type=int, default=10000)
def main(n_seats, initial_stacks, batch_size, eps_start, eps_end, eps_decay,
         target_update, num_episodes):
    env = create_environment(n_seats, initial_stacks)
    # play_out_hand(env, n_seats)
    train(env, memory_size)


if __name__ == "__main__":
    main()