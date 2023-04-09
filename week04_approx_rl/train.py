import argparse
import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from replay_buffer import OptimizedReplayBuffer, ReplayBuffer
import utils
from tqdm import trange
from IPython.display import clear_output
from framebuffer import FrameBuffer
import atari_wrappers
from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2
from dataclasses import dataclass
import json
from pathlib import Path
from pympler import asizeof
import gym.wrappers


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)

    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        m = np.array(channel_weights).reshape((1, 3))
        return cv2.transform(rgb, m)

    def observation(self, img):
        """what happens to each observation"""

        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (Use imresize from any library you want,
        #      e.g. opencv, PIL, keras. Don't use skimage.imresize
        #      because it is extremely slow.)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        cropped = img[31:194, 7:152]
        gray = self._to_gray_scale(cropped) / 255
        resized = cv2.resize(gray, (64, 64))
        clipped = np.clip(resized, 0, 1).astype('float32')
        return np.expand_dims(clipped, axis=0)


def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env


def wait_for_keyboard_interrupt():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        action = agent.sample_actions(qvalues)[0]
        next_s, r, done, _ = env.step(action)
        sum_rewards += r
        exp_replay.add(s, action, r, next_s, done)
        if done:
            s = env.reset()
        else:
            s = next_s

    return sum_rewards, s


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def make_env(clip_rewards=True, seed=None):
    ENV_NAME = "BreakoutNoFrameskip-v4"
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return self.relu(x)


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        self.backbone = nn.Sequential(  # [4, 64, 64]
            ConvBlock(4, 16),  # [16, 32, 32]
            ConvBlock(16, 32),  # [32, 16, 16]
            ConvBlock(32, 64),  # [64, 8, 8]
            nn.Flatten(),  # [4096]
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.backbone(state_t)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert (
                len(qvalues.shape) == 2 and
                qvalues.shape[0] == state_t.shape[0] and
                qvalues.shape[1] == self.n_actions
        )

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


class DuelingDQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        self.backbone = nn.Sequential(  # [4, 64, 64]
            ConvBlock(4, 16),  # [16, 32, 32]
            ConvBlock(16, 32),  # [32, 16, 16]
            ConvBlock(32, 64),  # [64, 8, 8]
            nn.Flatten(),  # [4096]
        )
        self.v = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.a = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        x = self.backbone(state_t)
        v = self.v(x)
        a = self.a(x)
        qvalues = v + (a - a.mean(dim=1, keepdim=True))

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert (
                len(qvalues.shape) == 2 and
                qvalues.shape[0] == state_t.shape[0] and
                qvalues.shape[1] == self.n_actions
        )

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network, device,
                    gamma=0.99,
                    check_shapes=False):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float32)  # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.int64)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    next_state_values = predicted_next_qvalues.amax(dim=-1)  # shape: [batch_size]

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], \
        "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * is_not_done * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, \
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, \
            "there's something wrong with target q-values, they must be a vector"

    return loss


def compute_td_loss_double_dqn(states, actions, rewards, next_states, is_done,
                               agent, target_network, device,
                               gamma=0.99,
                               check_shapes=False):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float32)  # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.int64)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]  # shape: [batch_size]

    predicted_next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]
    index_selection_next_qvalues = agent(next_states)
    max_actions = index_selection_next_qvalues.argmax(-1)
    next_state_values = predicted_next_qvalues[range(len(max_actions)), max_actions]  # shape: [batch_size]

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], \
        "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * is_not_done * next_state_values

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, \
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, \
            "there's something wrong with target q-values, they must be a vector"

    return loss


def train(env, agent, target_network, loss_fn, config, device):
    opt = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    state = env.reset()
    # exp_replay = OptimizedReplayBuffer(config.replay_buffer_size)
    exp_replay = ReplayBuffer(config.replay_buffer_size)
    for i in trange(config.replay_buffer_size // config.n_steps):
        if not utils.is_enough_ram(min_available_gb=0.1):
            print("""
                Less than 100 Mb RAM available. 
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """
                  )
            break
        play_and_record(state, agent, env, exp_replay, n_steps=config.n_steps)
        if len(exp_replay) == config.replay_buffer_size:
            break
    print('Replay size:', len(exp_replay))
    print('Replay memory:', asizeof.asizeof(exp_replay) / 1024 / 1024, 'MB')
    mean_rw_history = []
    td_loss_history = []
    grad_norm_history = []
    initial_state_v_history = []
    step = 0

    state = env.reset()
    with trange(step, config.total_steps + 1) as progress_bar:
        for step in progress_bar:
            if not utils.is_enough_ram():
                print('less that 100 Mb RAM available, freezing')
                print('make sure everything is ok and use KeyboardInterrupt to continue')
                wait_for_keyboard_interrupt()

            agent.epsilon = utils.linear_decay(config.init_epsilon, config.final_epsilon, step, config.decay_steps)

            # play
            _, state = play_and_record(state, agent, env, exp_replay, config.timesteps_per_epoch)

            # train
            states, actions, rewards, next_states, dones = exp_replay.sample(config.batch_size)

            loss = loss_fn(states, actions, rewards, next_states, dones, agent, target_network, device)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            opt.step()
            opt.zero_grad()

            if step % config.loss_freq == 0:
                td_loss_history.append(float(loss.data.cpu().item()))
                grad_norm_history.append(float(grad_norm.cpu().item()))

            if step % config.refresh_target_network_freq == 0:
                # Load agent weights into target_network
                target_network.load_state_dict(agent.state_dict())

            if step % config.eval_freq == 0:
                mean_rw_history.append(float(evaluate(
                    make_env(clip_rewards=True, seed=step), agent, n_games=3 * config.n_lives, greedy=True))
                )
                initial_state_q_values = agent.get_qvalues(
                    [make_env(seed=step).reset()]
                )
                initial_state_v_history.append(float(np.max(initial_state_q_values)))

                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                      (len(exp_replay), agent.epsilon))

                plt.figure(figsize=[16, 9])

                plt.subplot(2, 2, 1)
                plt.title("Mean reward per life")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                # plt.ylim(0, 20)
                plt.plot(utils.smoothen(td_loss_history))
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.title("Grad norm history (smoothened)")
                # plt.ylim(0, 50)
                plt.plot(utils.smoothen(grad_norm_history))
                plt.grid()

                plt.savefig(f'{config.name}/plots.png', bbox_inches='tight')

    torch.save(agent.state_dict(), f'{config.name}/model.pt')
    return {
        'mean_rw_history': mean_rw_history,
        'td_loss_history': td_loss_history,
        'grad_norm_history': grad_norm_history,
        'initial_state_v_history': initial_state_v_history
    }


def plot_monte_carlo(record, name):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(record['v_mc'], record['v_agent'])
    ax.plot(sorted(record['v_mc']), sorted(record['v_mc']),
           'black', linestyle='--', label='x=y')

    ax.grid()
    ax.legend()
    ax.set_title('State Value Estimates')
    ax.set_xlabel('Monte-Carlo')
    ax.set_ylabel('Agent')

    plt.savefig(f'{name}/mc.png', bbox_inches='tight')


def patch_render(env):
    env.metadata['render.modes'] = ['rgb_array']
    return env


@dataclass
class Config:
    name: str
    seed: int

    replay_buffer_size: int
    n_steps: int

    timesteps_per_epoch: int
    batch_size: int
    total_steps: int
    decay_steps: int

    learning_rate: float

    init_epsilon: float
    final_epsilon: float

    loss_freq: int
    refresh_target_network_freq: int
    eval_freq: int
    max_grad_norm: int
    n_lives: int


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replay_buffer_size', type=int, default=10 ** 4)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--timesteps_per_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_steps', type=int, default=int(2 * 10 ** 6))
    parser.add_argument('--decay_steps', type=int, default=1 * 10 ** 6)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--init_epsilon', type=float, default=1)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--loss_freq', type=int, default=50)
    parser.add_argument('--refresh_target_network_freq', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=int, default=50)
    parser.add_argument('--n_lives', type=int, default=5)

    parser.add_argument('--dueling', type=bool, default=True)
    parser.add_argument('--double', type=bool, default=True)

    args = parser.parse_args()
    return args


def parse_config(args):
    config = Config(name=args.name,
                    seed=args.seed,
                    replay_buffer_size=args.replay_buffer_size,
                    n_steps=args.n_steps,
                    timesteps_per_epoch=args.timesteps_per_epoch,
                    batch_size=args.batch_size,
                    total_steps=args.total_steps,
                    decay_steps=args.decay_steps,
                    learning_rate=args.learning_rate,
                    init_epsilon=args.init_epsilon,
                    final_epsilon=args.final_epsilon,
                    loss_freq=args.loss_freq,
                    refresh_target_network_freq=args.refresh_target_network_freq,
                    eval_freq=args.eval_freq,
                    max_grad_norm=args.max_grad_norm,
                    n_lives=args.n_lives
                    )
    return config


def main():
    args = get_args()
    data_dir = Path(args.name)
    data_dir.mkdir(exist_ok=True)
    config = parse_config(args)
    env = make_env(config.seed)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.dueling:
        agent = DuelingDQNAgent(state_shape, n_actions, epsilon=1).to(device)
        target_network = DuelingDQNAgent(state_shape, n_actions).to(device)
    else:
        agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
        target_network = DQNAgent(state_shape, n_actions).to(device)
    target_network.load_state_dict(agent.state_dict())
    loss_fn = compute_td_loss_double_dqn if args.double else compute_td_loss
    metrics = train(env, agent, target_network, loss_fn, config, device)
    eval_env = make_env(clip_rewards=False)
    record = utils.play_and_log_episode(eval_env, agent)
    plot_monte_carlo(record, args.name)
    result = np.sum(record['rewards'])
    for key, value in record.items():
        record[key] = value.tolist()
    record['result'] = float(result)
    print('Total reward for life: ', result)
    with open(data_dir / 'metrics.json', 'w+') as file:
        json.dump(metrics, file)
    with open(data_dir / 'record.json', 'w+') as file:
        json.dump(record, file)
    with gym.wrappers.Monitor(patch_render(make_env()), directory=str(data_dir / "videos"), force=True) as env_monitor:
        sessions = [evaluate(env_monitor, agent, n_games=config.n_lives, greedy=True) for _ in range(10)]


if __name__ == '__main__':
    main()
