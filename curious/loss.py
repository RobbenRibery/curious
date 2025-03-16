from typing import Optional, Tuple
import torch
import torch.nn as nn

from curious.buffer import Experience


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


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = None,
) -> torch.Tensor:
    """
    Compute the mean of the tensor over the specified dimension.
    If mask is None, return the mean of the whole tensor.
    If dim is None, return the mean of the whole tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to compute the mean of.
        mask (Optional[torch.Tensor]): The mask to compute the mean of.
        dim (Optional[int]): The dimension to compute the mean of.

    Returns:
        torch.Tensor: The mean of the tensor.
    """
    if mask is None:
        return tensor.mean(axis=dim)
    # when dim == None, return the mean of the whole tensor
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()