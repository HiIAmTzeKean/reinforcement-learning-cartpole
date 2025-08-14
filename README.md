
# Reinforcement Learning: CartPole with DQN (TensorFlow)

This repository contains a Deep Q-Network (DQN) implementation for solving the CartPole-v1 environment using TensorFlow and Keras. The project demonstrates core reinforcement learning concepts, including experience replay, target networks, and epsilon-greedy exploration.

## Features

- **Environment:** CartPole-v1 from OpenAI Gymnasium
- **Algorithm:** Deep Q-Network (DQN)
- **Frameworks:** TensorFlow, Keras
- **Replay Memory:** Experience replay buffer for stable training
- **Target Network:** Soft updates for improved stability
- **Epsilon-Greedy Policy:** Exploration-exploitation trade-off
- **Visualization:** Live training progress with Matplotlib

## Installation

1. Clone the repository:
 ```bash
 git clone https://github.com/HiIAmTzeKean/reinforcement-learning-cartpole.git
 cd reinforcement-learning-cartpole
 ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):
 ```bash
 uv sync
 ```

## Usage

Run the DQN training script:

```bash
python cartpole-tensorflow.py
```

Training progress will be displayed in a live Matplotlib plot. The script automatically detects GPU availability and adjusts the number of training episodes.

## File Structure

- `cartpole-tensorflow.py` — Main DQN implementation using TensorFlow/Keras.
- `cartpole-pytorch.ipynb` — (Optional) PyTorch implementation in Jupyter Notebook.
- `cartpole-pytorch.py` — (Optional) PyTorch script version.
- `README.md` — Project documentation.
- `pyproject.toml`, `uv.lock` — Dependency management files.

## Key Components

- **ReplayMemory:** Stores transitions for experience replay.
- **DQN Model:** Three-layer fully connected neural network.
- **Training Loop:** Handles environment interaction, optimization, and target network updates.
- **Plotting:** Visualizes episode durations and moving averages.

## Hyperparameters

- Batch size: 128
- Discount factor (gamma): 0.99
- Learning rate: 3e-4
- Epsilon decay: 2500 steps
- Replay memory size: 10,000
- Target network update rate (tau): 0.005

## Reproducibility

Random seeds are set for Python, NumPy, TensorFlow, and the environment to ensure reproducible results.

## References

- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [DQN Paper](https://www.nature.com/articles/nature14236)
