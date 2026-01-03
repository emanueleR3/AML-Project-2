#!/usr/bin/env python3
"""
Test rapido per SparseSGDM (Milestone M7).

Verifica:
1. Parametro mascherato (mask=0) rimane INVARIATO dopo N step
2. Parametro attivo (mask=1) cambia dopo N step
3. Nessun errore durante backward/step
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Import diretto per evitare import di src/__init__.py
sys.path.insert(0, str(Path(__file__).parent / "src"))
from optim import SparseSGDM


def test_sparse_sgdm_basic():
    """Test minimalissimo: 2 parametri, uno mascherato, uno attivo."""
    print("\n" + "="*70)
    print("TEST: SparseSGDM - Parametri mascherati vs attivi")
    print("="*70)
    
    # Modello: Linear(2, 1) => weight shape [1, 2]
    model = nn.Linear(2, 1, bias=False)
    
    # Mask: colonna 0 mascherata (0), colonna 1 attiva (1)
    mask = {
        'weight': torch.tensor([[0.0, 1.0]])  # shape [1, 2]
    }
    
    # Optimizer con mask
    optimizer = SparseSGDM(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
        mask=mask,
        apply_wd_to_masked_only=True
    )
    
    # Dummy data
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    # Salva pesi iniziali
    w_init = model.weight.data.clone()
    print(f"\n[INIT] Peso iniziale:\n{w_init}")
    print(f"[INIT] Mask:\n{mask['weight']}")
    
    # Training loop: N step
    num_steps = 10
    print(f"\n[TRAIN] Esecuzione {num_steps} step...")
    for step in range(num_steps):
        optimizer.zero_grad()
        
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1:2d}: loss={loss.item():.6f}")
    
    w_final = model.weight.data.clone()
    print(f"\n[FINAL] Peso finale:\n{w_final}")
    
    # === VERIFICHE ===
    print("\n" + "-"*70)
    print("VERIFICHE")
    print("-"*70)
    
    # Check 1: Parametro mascherato invariato?
    w_masked_init = w_init[0, 0].item()
    w_masked_final = w_final[0, 0].item()
    diff_masked = abs(w_masked_init - w_masked_final)
    
    print(f"\n✓ Parametro MASCHERATO (mask=0):")
    print(f"  Valore iniziale:  {w_masked_init:.8f}")
    print(f"  Valore finale:    {w_masked_final:.8f}")
    print(f"  Differenza:       {diff_masked:.2e}")
    
    if diff_masked > 1e-6:
        print(f"  ❌ FAIL: È CAMBIATO quando non dovrebbe!")
        return False
    else:
        print(f"  ✓ PASS: Rimane invariato ✓")
    
    # Check 2: Parametro attivo cambiato?
    w_active_init = w_init[0, 1].item()
    w_active_final = w_final[0, 1].item()
    diff_active = abs(w_active_init - w_active_final)
    
    print(f"\n✓ Parametro ATTIVO (mask=1):")
    print(f"  Valore iniziale:  {w_active_init:.8f}")
    print(f"  Valore finale:    {w_active_final:.8f}")
    print(f"  Differenza:       {diff_active:.8f}")
    
    if diff_active < 1e-6:
        print(f"  ❌ FAIL: NON è CAMBIATO quando dovrebbe!")
        return False
    else:
        print(f"  ✓ PASS: È stato aggiornato ✓")
    
    return True


def test_sparse_sgdm_no_mask():
    """Test backward compatibility: optimizer senza mask aggiorna tutto."""
    print("\n" + "="*70)
    print("TEST: SparseSGDM senza mask (backward compatibility)")
    print("="*70)
    
    # Modello
    model = nn.Linear(2, 1, bias=False)
    
    # Optimizer SENZA mask
    optimizer = SparseSGDM(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0
    )
    
    # Dummy data
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    w_init = model.weight.data.clone()
    
    # Training
    for step in range(5):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    w_final = model.weight.data.clone()
    diff = (w_init - w_final).abs().max().item()
    
    print(f"\n[RESULT] Differenza massima: {diff:.8f}")
    
    if diff < 1e-6:
        print(f"❌ FAIL: Nessun parametro è stato aggiornato!")
        return False
    else:
        print(f"✓ PASS: Tutti i parametri sono stati aggiornati ✓")
        return True


def test_sparse_sgdm_all_masked():
    """Test case: tutti i parametri mascherati => nulla cambia."""
    print("\n" + "="*70)
    print("TEST: SparseSGDM con tutti parametri mascherati")
    print("="*70)
    
    model = nn.Linear(2, 1, bias=False)
    
    # Mask: TUTTO a 0 (tutti mascherati)
    mask = {
        'weight': torch.zeros(1, 2)
    }
    
    optimizer = SparseSGDM(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        mask=mask
    )
    
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()
    
    w_init = model.weight.data.clone()
    
    # Training
    for step in range(5):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    w_final = model.weight.data.clone()
    diff = (w_init - w_final).abs().max().item()
    
    print(f"\n[RESULT] Differenza massima: {diff:.2e}")
    
    if diff > 1e-6:
        print(f"❌ FAIL: Parametri sono cambiati quando tutto era mascherato!")
        return False
    else:
        print(f"✓ PASS: Nulla è cambiato (corretto) ✓")
        return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MILESTONE M7 - SparseSGDM Tests")
    print("="*70)
    
    all_pass = True
    
    try:
        # Test 1
        if not test_sparse_sgdm_basic():
            all_pass = False
        
        # Test 2
        if not test_sparse_sgdm_no_mask():
            all_pass = False
        
        # Test 3
        if not test_sparse_sgdm_all_masked():
            all_pass = False
        
        print("\n" + "="*70)
        if all_pass:
            print("✅ TUTTI I TEST PASSED!")
            print("="*70 + "\n")
            sys.exit(0)
        else:
            print("❌ ALCUNI TEST FALLITI!")
            print("="*70 + "\n")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
