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
    r"""
    Sparse SGD with Momentum.
    
    Applica una mask ai gradienti prima dell'update, consentendo di
    allenare solo i parametri "attivi" (mask == 1).
    
    Algoritmo:
        1. d_p = d_p * mask          # applica mask
        2. d_p += weight_decay * p   # weight decay (solo attivi se flag=True)
        3. buf = momentum * buf + d_p
        4. p -= lr * buf
    
    Args:
        params (iterable): Parametri del modello
        lr (float): Learning rate (default: 0.01)
        momentum (float): Momentum coefficient (default: 0.9)
        weight_decay (float): L2 penalty coefficient (default: 1e-4)
        mask (Dict[str, Tensor], optional): Mask per parametro.
            Se None, nessuna mask viene applicata (backward compatible).
        apply_wd_to_masked_only (bool): Se True, weight decay solo su parametri
            attivi. Se False, su tutti. (default: True)
    
    Esempio:
        >>> model = nn.Linear(10, 5)
        >>> mask = {'weight': torch.ones(5, 10), 'bias': torch.ones(5)}
        >>> mask['weight'][:, :5] = 0  # maschiare prima 5 colonne
        >>> optimizer = SparseSGDM(model.parameters(), lr=0.01, mask=mask)
    """
    
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
        """Aggiorna la mask durante il training."""
        self.mask = mask
    
    @torch.no_grad()
    def step(self, closure=None):
        """Esegui un step dell'optimizer."""
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
                
                # Applica mask al gradiente
                # Cerca la mask per questo parametro per shape/posizione
                mask_tensor = None
                param_id = id(p)
                
                # Heuristica: cerca nella mask dict per shape match
                if len(self.mask) > 0:
                    for mask_key, mask_val in self.mask.items():
                        if mask_val.shape == d_p.shape:
                            mask_tensor = mask_val
                            break
                
                if mask_tensor is not None:
                    mask_tensor = mask_tensor.to(d_p.device)
                    d_p = d_p * mask_tensor
                
                # Weight decay
                if weight_decay != 0:
                    if self.apply_wd_to_masked_only and mask_tensor is not None:
                        # Weight decay solo su parametri attivi
                        d_p.add_(p * mask_tensor, alpha=weight_decay)
                    else:
                        # Weight decay su tutti
                        d_p.add_(p, alpha=weight_decay)
                
                # Momentum
                param_state = self.state[p]
                if len(param_state) == 0:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1.0)
                    d_p = buf
                
                # Update
                p.add_(d_p, alpha=-lr)
        
        return loss
