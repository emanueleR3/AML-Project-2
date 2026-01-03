# Milestone M7 — SparseSGDM

## ✅ Completato

### Implementazione: `src/optim.py`

Creato optimizer **SparseSGDM** (Sparse SGD with Momentum) che:

1. **Applica una mask ai gradienti** prima dell'update:
   ```python
   d_p = d_p * mask  # Solo parametri attivi (mask==1) vengono aggiornati
   ```

2. **Policy documentate**:
   - **Momentum**: Applicato normalmente a tutti i gradienti nel buffer (storia dell'ottimizzazione)
   - **Weight decay**: 
     - Se `apply_wd_to_masked_only=True` (default): peso L2 solo su parametri attivi
     - Se `apply_wd_to_masked_only=False`: peso L2 su tutti i parametri

3. **Backward compatibility**: Se nessuna mask viene passata, l'optimizer funziona come standard SGD+Momentum

### Test: `test_sparse_optim.py`

Tutti i test passano ✅:

#### Test 1: Parametri mascherati vs attivi
```
[RESULT]
✓ Parametro MASCHERATO (mask=0):
  - Rimane INVARIATO dopo 10 step
  - Differenza: 0.00e+00 ✓

✓ Parametro ATTIVO (mask=1):
  - Cambia dopo 10 step
  - Differenza: 0.38 ✓
```

#### Test 2: Backward compatibility (senza mask)
```
[RESULT]
✓ TUTTI i parametri vengono aggiornati normalmente ✓
```

#### Test 3: Tutti i parametri mascherati
```
[RESULT]
✓ Nulla cambia (corretto) ✓
```

### Export: `src/__init__.py`

`SparseSGDM` è stato aggiunto agli export del modulo `src`:
```python
from src import SparseSGDM
```

---

## 🎯 Stop condition (M7)

✅ **Test passa**  
✅ **Training su 1 batch non errora**  
✅ **Optimizer pronto per FL**  

---

## 📋 Prossimi step (M8)

La Milestone M8 richiede di:

1. Creare `scripts/run_sparse_fedavg.py` con:
   - `--stage calibrate_mask`: calcola e salva la mask
   - `--stage train_sparse`: allena con SparseSGDM + mask

2. Integrare SparseSGDM nel loop di FedAvg

3. Confrontare diverse regole di mask:
   - `least_sensitive`, `most_sensitive`, `random`, `magnitude-based`

4. Esperimenti:
   - iid vs non-iid
   - FedAvg standard vs sparse-FedAvg

---

## 📚 Riferimenti

- **Mask computation**: `src/masking.py` (Fisher diagonal, sensitivity scores)
- **Optimizer standard**: `torch.optim.SGD`
- **Policy ML**: SparseSGDM aggiorna solo parametri attivi, momentum applicato normalmente
