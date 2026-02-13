"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute numerically stable softmax along the specified dimension.
    
    Uses the formula:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    result = exp_x / sum_exp_x
    return torch.nan_to_num(result, nan=0.0)


def cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute cross-entropy loss with numerically stable log-softmax.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
        ignore_index: Target value to ignore when computing loss (default: -100)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    # Numerically stable log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - log_sum_exp  # (N, C)

    # Create mask for valid (non-ignored) targets
    mask = targets != ignore_index
    # Clamp targets so we can safely gather (ignored positions use index 0, won't matter)
    valid_targets = targets.clamp(min=0)

    # Gather log-probabilities for target classes: -log(p_target)
    nll = -log_probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)

    # Apply mask and compute mean over valid tokens
    nll = nll * mask.float()
    return nll.sum() / mask.float().sum()


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global L2 norm.
    
    If the total gradient norm exceeds max_norm, all gradients are scaled
    down proportionally so the total norm equals max_norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    parameters = list(parameters)
    max_norm = float(max_norm)

    # Collect gradients from parameters that have them
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)

    # Compute per-parameter L2 norms and total L2 norm
    norms = torch.stack([g.detach().norm(2.0) for g in grads])
    total_norm = norms.norm(2.0)

    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    # Scale gradients in-place
    for g in grads:
        g.detach().mul_(clip_coef_clamped)

    return total_norm


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    mask = targets != ignore_index
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    return correct.sum().float() / mask.sum().float()


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    return torch.exp(cross_entropy(logits, targets, ignore_index=ignore_index))
