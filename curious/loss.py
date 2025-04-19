from typing import Optional, Tuple
import torch
import torch.nn as nn

import numpy as np
from curious.buffer import Experience

@torch.compile(dynamic=True)
def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    Args:
        log_probs (torch.Tensor): The log probabilities of the current model.
        log_probs_ref (torch.Tensor): The log probabilities of the reference model.
        action_mask (Optional[torch.Tensor]): The action mask.

    Returns:
        torch.Tensor: The KL divergence.
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1

@torch.compile(dynamic=True)
def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = None,
    use_fixed_response_length: bool = False,
) -> torch.Tensor:
    """
    Compute the mean of the tensor over the specified dimension.
    If mask is None, return the mean of the whole tensor.
    If dim is None, return the mean of the whole tensor.

    Args:
        tensor (torch.Tensor): The tensor to compute the mean of. (batch_size, seq_len)
        mask (Optional[torch.Tensor]): The mask to compute the mean of. (batch_size, seq_len)
        dim (Optional[int]): The dimension to compute the mean of. default to None

    Returns:
        torch.Tensor: The mean of the tensor.
    """
    if mask is None:
        return tensor.mean(axis=dim)
    # when dim == None, return the mean of the whole tensor
    if use_fixed_response_length:
        return (tensor * mask).sum(axis=dim) / torch.ones_like(mask).sum(axis=dim)
    else:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class AdaptiveKLController:
    """
    Adaptive KL controller.
    """
    def __init__(self, init_kl_coef:float, target_kl:float = 0.5, horizon:int = 100):
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon

    def update(self, current_kl:float, n_steps:int):
        """
        Update the KL coefficient.

        Args:
            current_kl (float): The current KL divergence.
            n_steps (int): The number of steps.
        """
        # reduce kl penalty if kl is below target, increase if kl is above target
        proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class ConstantKLController:
    """
    Constant KL controller.
    """
    def __init__(self, init_kl_coef:float):
        self.value = init_kl_coef

    def update(self, *args, **kwargs):
        pass

class ActorLoss(nn.Module):

    def __init__(
        self, 
        epsilon:float = 0.2, 
        epsilon_high:float = 0.28,
        kl_weight:float = 0.001, 
        use_clip_high:bool = False, 
        use_token_level_loss:bool = False,
        use_fixed_response_length:bool = False,
        use_surrogate_loss:bool = True,
    ) -> None:
        """
        Args:
            epsilon: float, default to 0.2
            epsilon_high: float, default to 0.28
            kl_weight: float, default to 0.001
            use_clip_high: bool, default to False
            use_token_level_loss: bool, default to False
            use_fixed_response_length: bool, default to False
        """
        super().__init__()

        # kl weight
        self.kl_weight = kl_weight

        # surrogate loss clipping
        self.use_clip_high = use_clip_high
        self.epsilon_low = epsilon

        if self.use_clip_high:
            self.epsilon_high = epsilon_high
        else:
            self.epsilon_high = epsilon

        # token-level loss vs group-level loss
        self.use_token_level_loss = use_token_level_loss
        self.aggregation_dim = None if use_token_level_loss else -1
        self.use_fixed_response_length = False if use_token_level_loss else use_fixed_response_length
        self.use_surrogate_loss = use_surrogate_loss

    @torch.compile(dynamic=True)
    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the actor loss.

        Args:
            log_probs (torch.Tensor): The log probabilities of the current model. (batch_size, seq_len)
            experience (Experience): The experience.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The actor loss and the KL divergence.
        """
        old_log_probs = experience.action_log_probs
        action_mask = experience.action_mask
        advantages = experience.advantages

        if self.kl_weight > 0:
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=experience.log_probs_ref,
                action_mask=action_mask,
            )
            kl_loss = self.kl_weight * kl
        else:
            kl_loss = kl = torch.tensor(0.0)

        if self.use_surrogate_loss:
            # importance sampling ratio
            ratio = (log_probs - old_log_probs).exp()
            #print(f"ratio: {ratio.shape}")
            #print(f"advantages: {advantages.shape}")
            
            # surrogate loss 
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.epsilon_low, 1 + self.epsilon_high) * advantages

            policy_loss = -torch.min(surr1, surr2)
            loss = policy_loss + kl_loss
        else:
            policy_loss = -log_probs * advantages
            loss = policy_loss + kl_loss

        # token-level loss vs group-level loss
        loss = masked_mean(loss, action_mask, dim=self.aggregation_dim, use_fixed_response_length=self.use_fixed_response_length).mean()

        return loss, kl.mean(), policy_loss.mean()
