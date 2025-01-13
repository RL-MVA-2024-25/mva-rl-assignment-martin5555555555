import numpy as np
import os
os.environ['KMP_WARNINGS'] = 'off'
import warnings
warnings.filterwarnings('ignore')
import itertools



class SumSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def sum_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _sum_helper(self.tree, start, end, node, node_start, node_end)

    def sum(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.sum_helper(start, end, 1, 0, self.capacity - 1)

    def retrieve(self, upperbound: float) -> int:
        return _sum_retrieve_helper(self.tree, 1, self.capacity, upperbound)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _sum_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]



INF = float('inf')
class MinSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.array([INF for _ in range(2 * capacity)], dtype=np.float32)

    def min_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _min_helper(self.tree, start, end, node, node_start, node_end)

    def min(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.min_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _min_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


def _sum_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int):
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _sum_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _sum_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _sum_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _sum_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            return a + b


def _min_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int):
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _min_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _min_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _min_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _min_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            if a < b:
                return a
            else:
                return b


def _sum_setter_helper(tree: np.ndarray, idx: int):
    while idx >= 1:
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
        idx = idx // 2


def _min_setter_helper(tree: np.ndarray, idx: int):
    while idx >= 1:
        a = tree[2 * idx]
        b = tree[2 * idx + 1]
        if a < b:
            tree[idx] = a
        else:
            tree[idx] = b
        idx = idx // 2


def _sum_retrieve_helper(tree: np.ndarray, idx: int, capacity: int, upperbound: float):
    while idx < capacity: # while non-leaf
        left = 2 * idx
        right = left + 1
        if tree[left] > upperbound:
            idx = 2 * idx
        else:
            upperbound -= tree[left]
            idx = right
    return idx - capacity


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step_return: int = 3,
        gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size), dtype=np.float32)
        self.rews_buf = np.zeros((size), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr = 0
        self.size = 0
        self.n_step_return = n_step_return
        self.traj_obs = np.zeros((n_step_return, obs_dim), dtype=np.float32)
        self.traj_actions = np.zeros(n_step_return, dtype=np.float32)
        self.traj_rewards = np.zeros(n_step_return, dtype=np.float32)
        self.traj_next_idx = 0
        self.is_traj_full = False
        self.gamma = gamma

    def reset_traj(self):
        self.traj_obs = np.zeros_like(self.traj_obs)
        self.traj_actions = np.zeros_like(self.traj_actions)
        self.traj_rewards = np.zeros_like(self.traj_rewards)
        self.traj_next_idx = 0
        self.is_traj_full = False

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.traj_obs[self.traj_next_idx] = obs
        self.traj_actions[self.traj_next_idx] = act
        self.traj_rewards[self.traj_next_idx] = rew
        self.traj_next_idx = (self.traj_next_idx + 1) % self.n_step_return
        self.is_traj_full = self.is_traj_full or (self.traj_next_idx == 0)
        if self.is_traj_full:
            n_step_reward = 0
            discounted_gamma = 1
            for r in itertools.chain(
                self.traj_rewards[self.traj_next_idx :], self.traj_rewards[: self.traj_next_idx]
            ):
                n_step_reward += discounted_gamma * r
                discounted_gamma *= self.gamma

            self.obs_buf[self.ptr] = self.traj_obs[self.traj_next_idx]
            self.next_obs_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = self.traj_actions[self.traj_next_idx]
            self.rews_buf[self.ptr] = n_step_reward
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        if done:
            self.reset_traj()
        return self.is_traj_full

    def sample_batch(self) :
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer."""

    def __init__(
        self,
        obs_dim: int,
        size: int,
        batch_size: int = 32,
        n_step_return: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.000005,
    ):
        """Initialization."""
        assert alpha >= 0

        super().__init__(obs_dim, size, batch_size, n_step_return, gamma)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience and priority."""
        transition_added = super().store(obs, act, rew, next_obs, done)
        if transition_added:
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self):
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        indices = self._sample_proportional() % self.size

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = self._calculate_weights(indices, self.beta)

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)
        _update_priorities_helper(indices, priorities, self.sum_tree, self.min_tree, self.alpha)
        self.max_priority = max(self.max_priority, priorities.max())

    def _sample_proportional(self) -> np.ndarray:
        """Sample indices based on proportions."""
        return _sample_proportional_helper(self.sum_tree, len(self), self.batch_size)

    def _calculate_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        """Calculate the weights of the experiences"""
        return _calculate_weights_helper(indices, beta, self.sum_tree, self.min_tree, len(self))


def _sample_proportional_helper(
    sum_tree: SumSegmentTree,
    size: int,
    batch_size: int,
) :
    indices = np.zeros(batch_size, dtype=np.int32)
    p_total = sum_tree.sum(0, size - 1)
    segment = p_total / batch_size

    for i in range(batch_size):
        a = segment * i
        b = segment * (i + 1)
        upperbound = np.random.uniform(a, b)
        idx = sum_tree.retrieve(upperbound)
        indices[i] = idx

    return indices


def _calculate_weights_helper(
    indices: np.ndarray,
    beta: float,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    size: int,
) :

    weights = np.zeros(len(indices), dtype=np.float32)

    for i in range(len(indices)):

        idx = indices[i]

        # get max weight
        p_min = min_tree.min() / sum_tree.sum()
        max_weight = (p_min * size) ** (-beta)

        # calculate weights
        p_sample = sum_tree[idx] / sum_tree.sum()
        weight = (p_sample * size) ** (-beta)
        weight = weight / max_weight

        weights[i] = weight

    return weights


def _update_priorities_helper(
    indices: np.ndarray,
    priorities: np.ndarray,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    alpha: float,
):

    for i in range(len(indices)):
        idx = indices[i]
        priority = priorities[i]
        sum_tree[idx] = priority**alpha
        min_tree[idx] = priority**alpha
