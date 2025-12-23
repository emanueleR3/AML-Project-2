# AML Project 2 — Federated DINO on CIFAR-100

## Convenzioni usate qui
- **K**: numero totale di client (target: 100)
- **C**: frazione di client selezionati per round (target: 0.1)
- **J**: local steps per client (target iniziale: 4; poi 8, 16)
- **Nc**: numero di classi per client in scenario non‑iid (target: {1,5,10,50})
- **Round**: iterazione FL (client selection → local train → FedAvg)
- **Output**:
  - checkpoint: `outputs/checkpoints/`
  - figure: `outputs/figures/`
  - log: `outputs/logs/` (tensorboard/wandb/csv/json)

---

# Milestone M0 — Repo runnable (setup, logging, checkpoint)
### Obiettivo
Avere la repo eseguibile con CLI, con output coerenti, senza ancora fare training “vero”.

### Task
1. **Dipendenze**
   - Definisci `requirements.txt` (minimo): `torch`, `torchvision`, `pyyaml`, `tqdm`, `numpy`, `matplotlib`
   - (opzionale) `tensorboard` o `wandb`

2. **Utility di base** (`src/utils.py`)
   - `set_seed(seed: int)`
   - `get_device()` (cuda se disponibile)
   - `ensure_dir(path)`
   - `save_checkpoint(path, payload_dict)`
   - `load_checkpoint(path) -> payload_dict`
   - `save_metrics_json(path, dict)` / `append_metrics_csv(path, row_dict)`
   - logger minimale (stampa + salva su file)

3. **Interfaccia CLI** (`scripts/run_*.py`)
   - Lettura YAML da `--config`
   - Output directory per esperimento (es. `outputs/logs/<exp_name>/`)
   - Flag `--dry_run 1` (se vuoi) per testare pipeline senza training lungo

### Stop condition
- `python scripts/run_central.py --help` non errora
- `python scripts/run_central.py --config configs/central.yaml --dry_run 1` crea cartelle in `outputs/` e scrive un file metriche/log.

### Deliverable
- utilities funzionanti
- un “esempio di run” in README (1 comando)

---

# Milestone M1 — Data pipeline + split + sharding FL
### Obiettivo
Gestire CIFAR‑100 con split train/val/test e sharding per client (iid e non‑iid).

### Task (in `src/data.py`)
1. **Load dataset**
   - `torchvision.datasets.CIFAR100(root=..., train=True/False, download=True)`
2. **Split**
   - Crea split `train` e `val` partendo dal train (es. 90/10) + `test` dal test ufficiale.
   - Salva indici split (per riproducibilità) in `outputs/logs/.../splits.json`.
3. **Transforms**
   - `train_transform`: augmentation (crop, flip, normalization)
   - `eval_transform`: solo normalization
4. **Sharding IID**
   - Distribuisci esempi uniformemente tra K client.
5. **Sharding non‑IID (Nc)**
   - Ogni client riceve solo **Nc classi** (con esempi di quelle classi).
   - Vincolo: client disgiunti (ogni esempio a un solo client).
6. **Dataloader per client**
   - `get_client_loader(client_id, split='train', batch_size=...)`
7. **Sanity checks**
   - Stampa distribuzione classi per 2–3 client e verifica che:
     - iid ≈ uniforme
     - non‑iid rispetta Nc

### Stop condition
- `scripts/run_central.py --dry_run 1` riesce a caricare un batch da train/val/test.
- Per non‑iid: un client ha classi <= Nc (idealmente esattamente Nc).

### Deliverable
- funzione `build_federated_datasets(K, sharding, Nc, seed)` o equivalente
- output “class histogram per client” salvato (anche solo json)

---

# Milestone M2 — Modello DINO ViT‑S/16 + head CIFAR‑100
### Obiettivo
Costruire modello: backbone DINO + head per 100 classi, con controllo su freeze/unfreeze.

### Task (in `src/model.py`)
1. **Loader backbone DINO**
   - Scegli una via:
     - `torch.hub` (repo facebookresearch/dino) **oppure**
     - `timm`/HF (se preferisci)
   - Documenta la scelta nel README.
2. **Head classificazione**
   - Layer lineare: embedding_dim → 100
   - (opzionale) dropout
3. **Freeze policy**
   - Modalità:
     - `head_only` (backbone frozen)
     - `finetune_all` (tutto trainabile)
     - (opzionale) `last_blocks_only`
4. **Utility**
   - `get_trainable_params(model)` per optimizer
   - `count_params(model)` per logging

### Stop condition
- Forward su batch finto ritorna logits shape `[B, 100]`
- Training step singolo (loss backward) non errora

### Deliverable
- `build_model(config)` restituisce modello pronto

---

# Milestone M3 — Baseline centralizzata (benchmark)
### Obiettivo
Ottenere baseline centralizzata solida e riproducibile.

### Task
1. **Training loop** (`src/train.py`)
   - `train_one_epoch(model, loader, optimizer, scheduler, device)`
   - `evaluate(model, loader, device)`
   - metriche: `loss`, `accuracy`
2. **Script** (`scripts/run_central.py`)
   - Setup run folder (exp name da config)
   - Train per N epoche
   - Best checkpoint su val
   - Eval su test finale
   - Salva:
     - `metrics.json` (per epoca)
     - `best.pt` checkpoint
     - figure loss/acc in `outputs/figures/`
3. **Hyperparam sanity**
   - LR, wd, batch size, epochs, scheduler (cosine)
   - (opzionale) gradient clipping

### Stop condition
- Run completa produce:
  - `outputs/checkpoints/central_best.pt`
  - `outputs/figures/central_*.png`
  - log metriche per epoca

### Deliverable
- baseline pronta per confronto con FL

---

# Milestone M4 — FedAvg IID (baseline FL)
### Obiettivo
Implementare FedAvg con K=100, C=0.1, J=4 su iid.

### Task (in `src/fedavg.py` + `scripts/run_fedavg.py`)
1. **Client selection**
   - Campiona `m = max(1, int(C*K))` client per round
2. **Local training (client)**
   - Per ciascun client selezionato:
     - carica pesi globali
     - esegui J local steps (o mini‑epoch equivalente)
     - ritorna aggiornamento (state_dict o delta)
3. **Aggregazione FedAvg**
   - Media pesata per numero di esempi locali:  
     `w_global = Σ (n_k / n_total) * w_k`
4. **Logging per round**
   - Val/test loss/acc per round (non ogni step)
   - Salva checkpoint globali (es. ogni N round + best)
5. **Output pulito**
   - `outputs/logs/fedavg_iid/<run_id>/...`
   - figure curve round→metriche

### Stop condition
- Curva (val/test) migliora col tempo
- Nessun “collapse” immediato (loss esplode)
- Round completati senza OOM (riduci batch se serve)

### Deliverable
- FedAvg iid replicabile e confrontabile

---

# Milestone M5 — Non‑IID + sweep su Nc e J
### Obiettivo
Studiare eterogeneità non‑iid e effetto di J.

### Task
1. **Non‑iid setup**
   - Usa sharding non‑iid con `Nc ∈ {1,5,10,50}`
2. **Sweep local steps**
   - Per ogni Nc, esegui `J ∈ {4,8,16}`
3. **Normalizzazione del confronto**
   - Decidi un criterio:
     - **stesso numero di round** per tutti **oppure**
     - **stesso compute budget** (es. round * J costante)
   - Documentalo nel report.
4. **Tabella risultati**
   - Acc test finale (e magari best val) per ogni (Nc, J)
   - (opzionale) 3 seed per stimare varianza

### Stop condition
- Hai una matrice di risultati completa (Nc × J)
- Hai 1–2 figure comparabili (stesso asse round)

### Deliverable
- plot e tabella pronti per report

---

# Milestone M6 — Sensitivity/Fisher diag + mask calibration
### Obiettivo
Costruire una mask di sparsità per selezionare quali pesi aggiornare.

### Task (in `src/masking.py`)
1. **Stima sensibilità (Fisher diagonale o proxy)**
   - Stima per parametro: accumula statistiche su batch
   - Output: tensori “score” stessa shape dei parametri
2. **Calibrazione multi‑round**
   - Ripeti per più round/batch e aggiorna stima (media/EMA)
   - Salva `scores.pt` per riproducibilità
3. **Costruzione mask**
   - Input: `scores`, `sparsity_ratio`
   - Output: `mask` (0/1 o bool) per ciascun parametro
4. **Regole (per estensione)**
   - least_sensitive (baseline)
   - most_sensitive
   - random
   - highest_magnitude / lowest_magnitude
5. **Export**
   - salva `mask_<rule>_<ratio>.pt` in `outputs/checkpoints/` o `outputs/logs/...`

### Stop condition
- La mask ha la sparsità attesa (es. 10% pesi attivi)
- Coerenza: stessi parametri → stessa mask dato seed/config

### Deliverable
- pipeline di calibrazione ripetibile

---

# Milestone M7 — SparseSGDM
### Obiettivo
Allenare aggiornando solo i pesi “attivi” (mask) con un optimizer mascherato.

### Task (in `src/optim.py`)
1. **Implementazione**
   - Variante semplice: prima dell’update fai `p.grad *= mask`
   - Gestisci:
     - momentum (decidi policy e documentala)
     - weight decay (solo su attivi o su tutti? documenta)
2. **Test rapido**
   - Caso minimale: 2 parametri, uno mascherato (0), uno attivo (1)
   - Dopo N step:
     - param mascherato invariato
     - param attivo cambia

### Stop condition
- test passa
- training su 1 batch non errora

### Deliverable
- optimizer pronto per FL

---

# Milestone M8 — FedAvg + sparse fine‑tuning + extension
### Obiettivo
Integrare mask + SparseSGDM nel training FL e confrontare regole di mask.

### Task (in `scripts/run_sparse_fedavg.py`)
1. **Modalità stage**
   - `--stage calibrate_mask`: calcola e salva mask
   - `--stage train_sparse`: esegue FedAvg con SparseSGDM usando mask
2. **Integrazione in FL**
   - Decidi dove si calcola mask:
     - globale (una sola mask condivisa) **oppure**
     - per‑client (mask specifica)
   - Mantieni coerenza e spiegalo nel report.
3. **Esperimenti minimi**
   - iid: FedAvg vs sparse-FedAvg
   - non‑iid severo (es. Nc=1 o 5): FedAvg vs sparse-FedAvg
4. **Guided extension**
   - Ripeti training sparse per diverse regole mask:
     - least_sensitive, most_sensitive, random, magnitude-based
5. **Reporting**
   - Tabella riassuntiva + plot curve migliori

### Stop condition
- Hai risultati comparabili per almeno:
  - 1 config iid
  - 1 config non‑iid
  - ≥ 3 regole di mask
- Figure “report‑ready”

### Deliverable
- sezione extension forte (core del progetto)

---

# Milestone M9 — Report finale + pulizia consegna
### Obiettivo
Chiudere la consegna: report (8 pagine max + refs) + repo riproducibile.

### Task
1. **Report** (`report/main.tex`)
   - Setup: DINO, CIFAR‑100, split, transforms
   - Baseline central
   - FedAvg iid/non‑iid (Nc, J)
   - Masking + SparseSGDM
   - Extension (focus principale)
   - Limiti + possibili miglioramenti
2. **References** (`report/references.bib`)
   - FedAvg, survey FL, DINO, task arithmetic / sparse fine-tuning (quelli richiesti dalla traccia)
3. **README**
   - Install
   - Comandi esatti:
     - central
     - fedavg iid
     - fedavg non‑iid (con Nc e J)
     - calibrate mask
     - train sparse + extension rules
   - Dove trovare output (logs/figures/checkpoints)

### Stop condition
- Un TA può clonare la repo e lanciare almeno 1 run per pipeline senza “tribal knowledge”
- Report include risultati e figure chiave

### Deliverable
- repo consegnabile
- report compilabile (pdf) e referenze complete

---

## Checklist finale (quick)
- [ ] Central baseline ok
- [ ] FedAvg iid ok
- [ ] non‑iid sweep (Nc × J) completato
- [ ] mask calibration + mask saved
- [ ] SparseSGDM testato
- [ ] sparse-FedAvg + extension (regole mask) completato
- [ ] report + README finali

