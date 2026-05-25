Training Stability Review - SSTZIP-GNN Model
==============================================

## 1. ISSUES IDENTIFIED

### A. Negative Loss Values (HIGH PRIORITY)
**Problem**: Training logs show `train_loss_step=-28.8` (negative losses)
**Impact**: Loss should always be positive (it's negative log-likelihood)
**Root Cause**: Potential sign error in loss computation or incorrect scaling
**Status**: NEEDS INVESTIGATION

### B. Dataset Size vs. Compute (MEDIUM PRIORITY)
**Problem**: 1.9M training sequences on CPU only
**Impact**: Training takes ~42 minutes per epoch, very slow iteration
**Recommendation**: 
  - Use demo mode (50 zones, 7 days) for testing: ~350 sequences ✓ (implemented)
  - Scale to full data only after validation

### C. Feature Dimension Mismatch (RESOLVED)
**Problem**: Raw features (4-dim) → TCN expects (spatial_dim=64)
**Solution**: Added feature_projection layer (4→64) in trainer ✓

### D. Loss Function Signature (RESOLVED)
**Problem**: ZIPLoss expects tuple (pi, lambda), not separate args
**Solution**: Updated all loss calls to use tuple format ✓

---

## 2. RECOMMENDATIONS

### Phase 1: Validation (Today) - Demo Mode
```
- Use stable trainer script: scripts/04_train_model_stable.py
- Dataset: 50 zones × 7 days = 336 training sequences
- Epochs: 3-5 (completes in <2 minutes)
- Check: Loss values are POSITIVE
- Check: Model learns (validation loss decreases)
- Check: No NaN/Inf values
```

### Phase 2: Scaling (After validation)
```
- Increase to 261 zones × 30 days = ~2,790 sequences
- Epochs: 10-20
- Monitor loss behavior carefully
- Validate convergence
```

### Phase 3: Full Training
```
- Use complete dataset (261 zones, 90 days, 1.9M sequences)
- Consider GPU acceleration (10x speedup)
- Run for 50-100 epochs with early stopping
```

---

## 3. KEY HYPERPARAMETERS (CONSERVATIVE)

### Current Config (Stable):
```yaml
Model:
  spatial_dim: 32          # Reduced from 64
  temporal_dim: 32         # Reduced from 64
  num_spatial_layers: 1    # Reduced from 2
  num_temporal_layers: 1   # Reduced from 3
  dropout: 0.1            # Reduced from 0.2

Training:
  batch_size: 16          # Reduced from 32
  learning_rate: 0.001
  weight_decay: 1e-6      # Reduced from 1e-5
  gradient_clip_val: 1.0  # NEW: Prevents exploding gradients
  early_stopping_patience: 3
```

---

## 4. MONITORING CHECKLIST

During training, verify:

- [ ] Loss values are POSITIVE (not negative)
- [ ] Loss decreases over epochs (learning)
- [ ] Validation loss >= training loss (not underfitting)
- [ ] No NaN/Inf in logs
- [ ] No model parameter explosion
- [ ] GPU memory stable (if using GPU)
- [ ] Validation loss plateaus (convergence)

---

## 5. NEXT STEPS

### Step 1: Run Stable Trainer
```bash
python scripts/04_train_model_stable.py
```
Expected output:
```
Training complete in ~120 seconds
Test loss: 3.2456 (POSITIVE)
Status: SUCCESS
```

### Step 2: Check Loss Values
If test loss is negative:
  → Debug ZIPLoss.forward() implementation
  → Check loss sign/scaling
  → Validate probability bounds

If test loss is positive:
  → Proceed to Phase 2 scaling
  → Run with 30-day dataset
  → Monitor convergence

### Step 3: Visualize Results
After validation:
```bash
python scripts/visualize_model.py
```
Review:
  - Training history (loss curve)
  - Predictions vs actual
  - Zone-level performance

---

## 6. FILES INVOLVED

### Training Code:
- `scripts/04_train_model.py` (full version - use after validation)
- `scripts/04_train_model_stable.py` (demo version - for testing NOW)
- `src/training/trainer.py` (SSTZIPGNNLightning)

### Model Components:
- `src/models/sstzip_gnn/sstzip_gnn.py`
- `src/models/sstzip_gnn/tcn.py`
- `src/models/sstzip_gnn/zip_head.py` (ZIPLoss implementation)

### Configuration:
- `config/config.yaml`

---

## 7. EXPECTED BEHAVIOR

### Stable Training:
```
Epoch 0: train_loss=2.845, val_loss=2.932
Epoch 1: train_loss=2.723, val_loss=2.821
Epoch 2: train_loss=2.634, val_loss=2.754
Training complete in 120 seconds
Test loss: 2.698
Status: SUCCESS ✓
```

### Signs of Instability:
- Losses stay negative → Implementation error
- Validation loss >> training loss → Underfitting
- Exploding gradients → Need gradient clipping
- No improvement → Learning rate too small

---

## 8. DECISION: DEMO VS FULL

**RECOMMENDATION: Start with Demo Mode**

Reasons:
1. Complete in <2 minutes (vs 42 min/epoch full)
2. Validates entire pipeline
3. Catches bugs early
4. Saves 39 minutes per failed attempt
5. Allows quick iteration

Approval criteria to scale:
- [ ] Loss values are POSITIVE
- [ ] Validation loss decreases
- [ ] No errors in logs
- [ ] Model checkpoints save correctly
- [ ] Loss values are reasonable (2-5 range)

---

Generated: 2024-03-15
Status: READY FOR TESTING
Next: Run stable trainer and check loss behavior
