import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ks: torch.Tensor = None):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))

        # Apply per-sequence top-k filtering if top_ks is provided
        if top_ks is not None and (top_ks > 0).any():
            batch_size = logits.size(0)
            vocab_size = logits.size(-1)

            # Apply top-k filtering to each sequence
            for i in range(batch_size):
                if top_ks[i] > 0:
                    top_k = min(int(top_ks[i].item()), vocab_size)  # Safety check
                    # Get the logits of the sequence
                    seq_logits = logits[i]
                    # Find the token with the lowest probability in the sequence
                    topk_values = torch.topk(seq_logits, top_k)[0]
                    # Set the threshold
                    threshold = topk_values[-1]
                    # Set the tokens below the threshold to negative infinity
                    seq_logits[seq_logits < threshold] = float('-inf')

        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
