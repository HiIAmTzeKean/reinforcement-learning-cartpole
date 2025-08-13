import math
import random
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model, layers, losses, optimizer


# Set up environment
env = gym.make("CartPole-v1")

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# Set device (TensorFlow handles this internally)

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

# Replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# DQN model
class DQN(Model):
    def __init__(self, n_observations, n_actions) -> None:
        super().__init__()
        self.layer1 = layers.Dense(128, activation='relu')
        self.layer2 = layers.Dense(128, activation='relu')
        self.layer3 = layers.Dense(n_actions)
    
    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.set_weights(policy_net.get_weights())

optimizer = optimizer.Adam(learning_rate=LR)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        q_values = policy_net(state)
        return tf.argmax(q_values[0]).numpy()
    else:
        return env.action_space.sample()

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = np.array(episode_durations, dtype=np.float32)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t)
    if len(durations_t) >= 100:
        means = np.convolve(durations_t, np.ones(100)/100, mode='valid')
        means = np.concatenate([np.zeros(99), means])
        plt.plot(means)
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


@tf.function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = np.array([s is not None for s in batch.next_state], dtype=bool)
    non_final_next_states = np.vstack([s for s in batch.next_state if s is not None]) if any(non_final_mask) else np.empty((0, n_observations))
    state_batch = np.vstack(batch.state)
    action_batch = np.array(batch.action)
    reward_batch = np.array(batch.reward, dtype=np.float32)

    with tf.GradientTape() as tape:
        state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_tensor = tf.convert_to_tensor(action_batch, dtype=tf.int32)
        reward_tensor = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        q_values = policy_net(state_tensor)
        indices = tf.stack([tf.range(BATCH_SIZE), action_tensor], axis=1)
        state_action_values = tf.gather_nd(q_values, indices)
        next_state_values = tf.zeros(BATCH_SIZE, dtype=tf.float32)
        if len(non_final_next_states) > 0:
            next_state_tensor = tf.convert_to_tensor(non_final_next_states, dtype=tf.float32)
            next_q = target_net(next_state_tensor)
            next_state_values_np = tf.reduce_max(next_q, axis=1).numpy()
            next_state_values = tf.convert_to_tensor([next_state_values_np[i] if mask else 0.0 for i, mask in enumerate(non_final_mask)], dtype=tf.float32)
        expected_state_action_values = reward_tensor + GAMMA * next_state_values
        loss = losses.Huber()(expected_state_action_values, state_action_values)
    grads = tape.gradient(loss, policy_net.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 100)
    optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

num_episodes = 600 if tf.config.list_physical_devices('GPU') else 50

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = np.array(state, dtype=np.float32)[None, :]
    for t in count():
        action = select_action(tf.convert_to_tensor(state, dtype=tf.float32))
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        done = terminated or truncated
        next_state = None if terminated else np.array(observation, dtype=np.float32)[None, :]
        memory.push(state, action, next_state, reward)
        state = next_state if next_state is not None else state
        optimize_model()
        # Soft update of the target network
        target_weights = target_net.get_weights()
        policy_weights = policy_net.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = policy_weights[i] * TAU + target_weights[i] * (1 - TAU)
        target_net.set_weights(target_weights)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
