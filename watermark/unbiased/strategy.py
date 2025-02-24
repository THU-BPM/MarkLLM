import torch
import torch.nn.functional as F
from typing import Union

class WatermarkStrategy:
    """Base class for watermark strategies."""
    def from_random(self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int) -> torch.LongTensor:
        raise NotImplementedError
        
    def reweight_logits(self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor, alpha: float) -> torch.FloatTensor:
        raise NotImplementedError

class DeltaStrategy(WatermarkStrategy):
    """Strategy for delta-reweight."""
    def from_random(self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int) -> torch.LongTensor:
        if isinstance(rng, list):
            batch_size = len(rng)
            u = torch.stack(
                [
                    torch.rand((), generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            u = torch.rand((), generator=rng, device=rng.device)
        return u
    
    def reweight_logits(self, u: torch.LongTensor, p_logits: torch.FloatTensor) -> torch.FloatTensor:
        """Reweight the logits using the u."""
        cumsum = torch.cumsum(F.softmax(p_logits, dim=-1), dim=-1)
        index = torch.searchsorted(cumsum, u[..., None], right=True)
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device) == index,
            torch.full_like(p_logits, 0),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits
    
class GammaStrategy(WatermarkStrategy):
    """Strategy for gamma-reweight."""
    def from_random(self, rng: Union[torch.Generator, list[torch.Generator]], vocab_size: int) -> torch.LongTensor:
        """Generate a permutation from the random number generator."""
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return shuffle
        
    def reweight_logits(self, shuffle: torch.LongTensor, p_logits: torch.FloatTensor, alpha: float = 0.5) -> torch.FloatTensor:
        """Reweight the logits using the shuffle and alpha."""
        unshuffle = torch.argsort(shuffle, dim=-1)
        
        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        
        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax((s_cumsum > (1-alpha)).to(torch.int), dim=-1, keepdim=True)
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (torch.gather(s_cumsum, -1, boundary_2) - (1-alpha)) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1-alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2/2 + s_all_portion_in_right_1/2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits