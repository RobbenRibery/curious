import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GRPOPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        # Converts raw output (logits) to probabilities via softmax
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

def compute_returns_and_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Example advantage computation (GAE-like). 
    For each state, compute the discounted return and advantage estimate.
    values: estimated state-values from a value function (if you have one). 
    If not, you can omit advantage and just use reward-to-go.
    """
    advantages = []
    gae = 0
    returns = []
    running_return = 0

    # Process from last step to first
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + (gamma * values[step + 1] if step < len(rewards) - 1 else 0) - values[step]
        gae = delta + gamma * lam * gae
        advantages.append(gae)
        running_return = rewards[step] + (gamma * running_return)
        returns.append(running_return)
    
    advantages.reverse()
    returns.reverse()
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    return returns, advantages

def train_grpo(env, policy_net, optimizer, num_episodes=1000, gamma=0.99, lam=0.95):
    """
    Main training loop for the GRPO-like algorithm.
    env: a gym-like environment
    policy_net: policy network
    optimizer: optimizer for policy network
    """
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        values = []
        states = []
        actions = []
        
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            dist = policy_net(state_tensor)
            action = dist.sample()
            
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            actions.append(action)

            # If you also have a separate value function:
            # value_estimate = value_net(state_tensor)
            # Store value_estimate.item() or similar.
            
            next_state, reward, done, _info = env.step(action.item())
            rewards.append(reward)
            states.append(state)
            
            state = next_state

        # For simplicity, use zero for values. If you have a critic, use it here.
        values = [0]*len(rewards)
        values.append(0)  # next state value, we set it to 0 for last step
        returns, advantages = compute_returns_and_advantages(rewards, values, gamma, lam)

        # Combine log_probs and advantages
        log_probs_tensor = torch.stack(log_probs)
        
        # Policy gradient objective: sum of log_probs * advantage
        loss = -(log_probs_tensor * advantages).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item():.4f}")

def main():
    # Example usage:
    import reasoning_gym  # imaginary environment
    env = reasoning_gym.make("SomeEnvironment-v0")  # hypothetical environment
    
    # Dummy state_dim, action_dim - adapt to your real env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = GRPOPolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    
    train_grpo(env, policy_net, optimizer)

if __name__ == "__main__":
    main()
