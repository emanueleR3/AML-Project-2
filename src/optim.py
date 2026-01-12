"""
Sparse optimizers for Federated Learning.

SparseSGDM: SGD with Momentum that applies mask to gradients.

La mask viene applicata ai gradienti PRIMA dell'update:
    d_p = d_p * mask
    
Ciò permette di allenare solo i pesi "attivi" (mask==1) senza modificare
i pesi mascherati (mask==0) durante il training.

Policy documentate:
- Momentum: applicato normalmente a tutti i gradienti (sia attivi che inattivi)
  nel buffer. Questo è corretto perché momentum è storia dell'ottimizzazione.
- Weight decay: applicato solo ai parametri attivi (flag apply_wd_to_masked_only=True).
  Se un peso è mascherato, non deve nemmeno "decadere" per non inquinare l'update.
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Optional, Iterable


class SparseSGDM(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        mask: Optional[Dict[str, torch.Tensor]] = None,
        apply_wd_to_masked_only: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        
        self.mask = mask or {}
        self.apply_wd_to_masked_only = apply_wd_to_masked_only
    
    def set_mask(self, mask: Dict[str, torch.Tensor]):
        self.mask = mask
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                mask_tensor = None
                param_id = id(p)
                
                if len(self.mask) > 0:
                    for mask_key, mask_val in self.mask.items():
                        if mask_val.shape == d_p.shape:
                            mask_tensor = mask_val
                            break
                
                if mask_tensor is not None:
                    mask_tensor = mask_tensor.to(d_p.device)
                    d_p = d_p * mask_tensor
                
                if weight_decay != 0:
                    if self.apply_wd_to_masked_only and mask_tensor is not None:
                        d_p.add_(p * mask_tensor, alpha=weight_decay)
                    else:
                        d_p.add_(p, alpha=weight_decay)
                
                param_state = self.state[p]
                if len(param_state) == 0:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1.0)
                    d_p = buf
                
                p.add_(d_p, alpha=-lr)
        
        return loss
