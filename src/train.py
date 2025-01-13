from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import *
import logging
import random
import torch.nn.functional as F
import torch.optim as optim

#@title Network

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, nf),
            nn.LeakyReLU(),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.Linear(nf, out_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
    
        epsilons = torch.zeros(x.shape)+1e-16
        x = torch.log10(x + epsilons.to(x.device))
        return self.layers(x)

class DuelingNetwork(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(DuelingNetwork, self).__init__()
        self.value_net = Network(in_dim, nf, 1)
        self.adv_net = Network(in_dim, nf, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        value = self.value_net(x)
        adv = self.adv_net(x)
        return value + adv - adv.mean(dim=-1, keepdim=True)
class ProjectAgent:
    def __init__(
        self
    ):

        # Make environments
        self.max_days = 1000
        self.treatment_days = 5
        self.reward_scaler = 1e8
        self.max_episode_steps = 200#max_days // treatment_days
        

        # (1) Make Envs
        self.envs = {
            "train": TimeLimit(HIVPatient(), max_episode_steps=200)
,
        }
        obs_dim = 6
        action_dim = 4

        # Parameters
        
        self.ckpt_path = 'src/test_final/ckpt.pt'
        self.batch_size = 2048
        self.grad_clip = 1000.0
        self.target_update = 1000
        self.epsilon = 0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 1/300
        self.decay_option = 'logistic'
        self.discount_factor = 0.99
        self.n_train = 1
        self.n_step_return = 1
        self.alpha = 0.2
        self.beta = 0.6
        self.beta_increment_per_sampling = 3e-6
        self.lr = 2e-4
        self.l2_reg = 0.0
        self.prior_eps = 1e-6
        self.hidden_dim = 1024
        self.double_dqn = True
        self.memory_size = int(1e6)
        self.load_ckpt = True
        self.is_test = True
        # Device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {self.device}")
        print(f"device: {self.device}")

        # PER memory
        self.per = True
        if self.per:
            
            self.memory = PrioritizedReplayBuffer(
                obs_dim,
                self.memory_size,
                2048,
                self.n_step_return,
                self.discount_factor,
                self.alpha,
                self.beta,
                self.beta_increment_per_sampling,
            )
        else:
            self.memory = ReplayBuffer(
                obs_dim, self.memory_size, self.batch_size, self.n_step_return, self.discount_factor
            )

        # Double DQN
        self.double_dqn = self.double_dqn

        # Networks: DQN, DQN_target
        dqn_config = dict(
            in_dim=obs_dim,
            nf=self.hidden_dim,
            out_dim=action_dim,
        )
        # self.dqn = Network(**dqn_config).to(self.device)
        # self.dqn_target = Network(**dqn_config).to(self.device)
        self.dqn = DuelingNetwork(**dqn_config).to(self.device)
        self.dqn_target = DuelingNetwork(**dqn_config).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        # Mode: train / test
        self.max_cum_reward = -1.0

        # Record (archive train / test results)
        self.record = []

        # Initial episode (default: 1)
        self.init_episode = 1

        if self.load_ckpt:
            self.load()

   

    def save(self, path: str):
        if self.per:
            _memory = _gather_per_buffer_attr(self.memory)
        else:
            _memory = _gather_replay_buffer_attr(self.memory)
        ckpt = dict(
            dqn=self.dqn.state_dict(),
            dqn_target=self.dqn_target.state_dict(),
            optim=self.optimizer.state_dict(),
        )
        torch.save(ckpt, path)

    def load(self):
        ckpt = torch.load(self.ckpt_path, map_location_device='cpu')
        self.dqn.load_state_dict(ckpt["dqn"])
        self.dqn_target.load_state_dict(ckpt["dqn_target"])
        self.optimizer.load_state_dict(ckpt["optim"])
        """
        for key, value in ckpt["memory"].items():
            if key not in ["sum_tree", "min_tree"]:
                setattr(self.memory, key, value)
            else:
                tree = getattr(self.memory, key)
                setattr(tree, "capacity", value["capacity"])
                setattr(tree, "tree", value["tree"])
        """
        
        logging.info(f"Success: Checkpoint loaded (start from Episode {self.init_episode})!")
        print(f"Success: Checkpoint loaded (start from Episode {self.init_episode})!")

    def act(self, observation: np.ndarray, use_random=False) -> int:
        """Select an action from the input observation."""
        # epsilon greedy policy (only for training)
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = random.randint(0,3)#self.envs["train"].action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(observation).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        if self.per:
            # PER needs beta to calculate weights
            samples = self.memory.sample_batch()
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
            indices = samples["indices"]
        else:
            # Vanilla DQN does not require any weights
            samples = self.memory.sample_batch()

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        if self.per:
            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.per:
            # PER: update priorities
            loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def train(
        self, max_episodes: int, train_mode:str, log_freq: int, test_freq: int, save_freq: int, img_dir: str
    ):
        """Train the agent."""
        self.is_test = False

        max_steps = self.max_episode_steps
        update_cnt = 0
        start = datetime.now()
        env = self.envs["train"]
        if train_mode=='random':
                env.unwrapped.domain_randomization = True
        for episode in range(self.init_episode, max_episodes + 1):
           
            state = env.reset()[0]
            #state = (state-state.mean())/state.std()
            losses = []
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, truncated, _ = self.step(env, action)
                transition = [state, action, reward, next_state, done]
                self.memory.store(*transition)
                state = next_state

                # If training is available:
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.n_train):
                        loss = self.update_model()
                    losses.append(loss)
                    self.writer.add_scalar("loss", loss, update_cnt)
                    update_cnt += 1

                    # # epsilon decaying
                    # if self.decay_option == "linear":
                    #     self.epsilon = max(
                    #         self.min_epsilon,
                    #         self.epsilon
                    #         - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    #     )
                    # elif self.decay_option == "logistic":
                    #     self.epsilon = self.min_epsilon + (
                    #         self.max_epsilon - self.min_epsilon
                    #     ) * sigmoid(1 / self.epsilon_decay - episode)
                    if episode < 1 / self.epsilon_decay:
                        self.epsilon = self.max_epsilon
                    else:
                        self.epsilon = max(
                            self.min_epsilon,
                            self.epsilon
                            - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                        )

                    # Target network update
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # If episode ends:
                if done:
                    break

            avg_step_train_loss = np.array(losses).sum() * self.batch_size / max_steps

            # Test
            if test_freq > 0 and episode % test_freq == 0:
                last_treatment_day, max_E, last_E, cum_reward = self.test(episode, img_dir)
                self.record.append(
                    {
                        "episode": episode,
                        "last_treatment_day": last_treatment_day,
                        "max_E": max_E,
                        "last_E": last_E,
                        "cum_reward": cum_reward,
                        "train_loss": avg_step_train_loss,
                        }
                )
                self._save_record_df()

                # Logging
                if log_freq > 0 and episode % log_freq == 0:
                    self._track_results(
                        episode,
                        datetime.now() - start,
                        train_loss=avg_step_train_loss,
                        max_E=max_E,
                        last_E=last_E,
                        cum_reward=cum_reward
                  )

            # Save
            if save_freq > 0 and episode % save_freq == 0:
                path = os.path.join(self.ckpt_dir, "ckpt.pt")
                self.save_ckpt(episode, path)

        env.close()


    def step(self, env, action):
        next_state, reward, done, truncated,  _ = env.step(action)
        #next_state = (next_state-next_state.mean())/next_state.std()
        return  next_state, reward/self.reward_scaler, done, truncated,  _ 
    
    
    def _compute_dqn_loss(self, samples):
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)
        if not self.double_dqn:
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_value = (
                self.dqn_target(next_state)
                .gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True))
                .detach()
            )
        mask = 1 - done
        #print(reward)
        target = (reward + self.discount_factor * next_q_value * mask).to(self.device)
        #print(target)
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    
  

    
def _gather_replay_buffer_attr(memory) :
    if memory is None:
        return {}
    replay_buffer_keys = [
        "obs_buf",
        "next_obs_buf",
        "acts_buf",
        "rews_buf",
        "done_buf",
        "max_size",
        "batch_size",
        "ptr",
        "size",
    ]
    result = {key: getattr(memory, key) for key in replay_buffer_keys}
    return result


def _gather_per_buffer_attr(memory) :
    if memory is None:
        return {}
    per_buffer_keys = [
        "obs_buf",
        "next_obs_buf",
        "acts_buf",
        "rews_buf",
        "done_buf",
        "max_size",
        "batch_size",
        "ptr",
        "size",
        "max_priority",
        "tree_ptr",
        "alpha",
    ]
    result = {key: getattr(memory, key) for key in per_buffer_keys}
    result["sum_tree"] = dict(
        capacity=memory.sum_tree.capacity,
        tree=memory.sum_tree.tree,
    )
    result["min_tree"] = dict(
        capacity=memory.min_tree.capacity,
        tree=memory.min_tree.tree,
    )
    return result


def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))


def get_last_treatment_day(action: np.ndarray) -> int:
    """Find the last treatment day (i.e, nonzero actions) for a given action sequence."""
    n = len(action)
    for i in range(n - 1, -1, -1):
        if action[i] != 0:
            return i + 1
    return 0


