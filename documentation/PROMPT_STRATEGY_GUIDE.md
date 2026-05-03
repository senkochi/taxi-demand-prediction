# 💬 Guide Prompt Hiệu Quả Với Documentation System

**Purpose:** Giúp bạn prompt AI/Copilot hiệu quả bằng cách attach đúng file, context phù hợp  
**Audience:** Person A, Person B, AI assistants

---

## 1. Nguyên Tắc Cơ Bản

### ❌ **Làm Sao KHÔNG NÊN**
```
❌ "Viết code Spark"
   → Quá mơ hồ, AI không biết bạn cần cái gì

❌ "Implement SSTZIP-GNN"
   → Quá to, không có context, output sẽ lòng vòng

❌ Attach cả 10 files
   → AI confuse, input quá nhiều, token lãng phí
```

### ✅ **Làm Sao NÊN**
```
✅ "Viết PySpark script cho Method 2 clustering. 
    Dùng KMeans++ với 5 features: trip_count, avg_distance, avg_fare, 
    avg_passenger_count, avg_trip_duration. 
    Include silhouette score analysis."
   → Cụ thể, có context, dễ theo dõi

✅ Attach 2-3 files liên quan
   → Đủ context, không overload

✅ Nêu acceptance criteria
   → AI biết khi nào xong
```

---

## 2. Matrix: File Nào Attach Khi?

### **For Development & Implementation**

| Khi Làm | Primary File | Secondary Files | Optional |
|---------|--------------|-----------------|----------|
| **Setup env** | DEVELOPMENT_SETUP.md | - | - |
| **Data pipeline** | DATA_PIPELINE_SPECIFICATION.md | MODULE_BIG_DATA_ENGINEERING.md | - |
| **Spark ETL** | MODULE_BIG_DATA_ENGINEERING.md | DATA_PIPELINE_SPECIFICATION.md | DEVELOPMENT_SETUP.md |
| **Model impl** | MODULE_DEEP_LEARNING_MODELING.md | AI_CONFIG_SYSTEM_PROMPT.md | - |
| **Training** | MODULE_DEEP_LEARNING_MODELING.md | EXPERIMENT_TRACKING.md | - |
| **Evaluation** | MODULE_EXPERIMENT_REPORTING.md | EXPERIMENT_TRACKING.md | - |
| **Code review** | COLLABORATION_GUIDELINES.md | - | - |
| **Debugging** | MODULE_*.md (relevant) | DEVELOPMENT_SETUP.md | - |

### **For Planning & Coordination**

| Khi Cần | Primary File | Notes |
|--------|--------------|-------|
| Know this week's task | PROJECT_ROADMAP.md | Thường không cần attach |
| Track progress | PROJECT_ROADMAP.md | Attach khi update status |
| Set sprint goals | COLLABORATION_GUIDELINES.md | Section: Sprint Planning |
| Review code | COLLABORATION_GUIDELINES.md | Section: Code Review Checklist |
| Resolve blocker | COLLABORATION_GUIDELINES.md | Section: Conflict Resolution |

### **For Writing & Reporting**

| Khi Cần | Primary File | Secondary Files |
|--------|--------------|-----------------|
| Draft thesis | MODULE_EXPERIMENT_REPORTING.md | Relevant MODULE file |
| Create figures | MODULE_EXPERIMENT_REPORTING.md | Section 3: Visualizations |
| Compare methods | EXPERIMENT_TRACKING.md | MODULE_EXPERIMENT_REPORTING.md |
| Write presentation | MODULE_EXPERIMENT_REPORTING.md | Section 4.3 |

---

## 3. Prompt Patterns Theo Giai Đoạn

### **Week 1-2: Setup & Data**

#### Pattern 1: Setup Environment
```
"Tôi cần setup development environment cho taxi prediction project.
 Dùng Python 3.11, PySpark 3.3, PyTorch. 
 Chỉ mình setup local venv trước, sau đó dùng Spark standalone."

Attach: DEVELOPMENT_SETUP.md (section 2-3)
Optional: docker-compose.yml nếu muốn Docker option

Expected: Step-by-step commands, folder structure
```

#### Pattern 2: Data Validation
```
"Viết PySpark script để validate NYC Taxi data.
 Kiểm tra:
 1. Schema validation (17+ required columns)
 2. Outlier detection: fare_amount (Z-score > 3)
 3. Null handling
 4. Date range [2022-01-01, 2024-01-01]
 5. Output: validation report + cleaned Parquet
 
 Expected execution time: 10-15 min on 5M records"

Attach: 
  - DATA_PIPELINE_SPECIFICATION.md (section 3: Validation)
  - MODULE_BIG_DATA_ENGINEERING.md (section 1.2: Outlier Detection)

Expected: Full PySpark script, error handling, logging
```

---

### **Week 3-5: Implementation**

#### Pattern 3: Feature Extraction (Specific Method)
```
"Implement Method 2: Mobility Pattern Clustering features.

Requirements from spec:
- Extract 5 features: trip_count, avg_distance, avg_fare, 
  avg_passenger_count, avg_trip_duration
- Per zone (PULocationID)
- Normalize: StandardScaler (zero mean, unit variance)
- Handle nulls: impute with zone median
- Output: Parquet at data/processed/method2_features/

Code should include:
1. Comments explaining each step
2. Error handling
3. Logging (record count before/after)
4. Performance timing
5. Save schema to metadata.json"

Attach:
  - DATA_PIPELINE_SPECIFICATION.md (section 4.3: Method 2 Specification)
  - MODULE_BIG_DATA_ENGINEERING.md (section 2.3: Method 2 Features)

Expected: 
  - Complete PySpark script (100-150 lines)
  - Docstring with parameters
  - MLflow logging ready
```

#### Pattern 4: Model Architecture
```
"Implement DGCN (Diffusion Graph Convolution) layer in PyTorch.

Specifications:
- Input: node features [num_nodes, in_channels]
- Adjacency: normalized A (degree-normalized)
- Formula: H = Σ(w_k,1*(D_out^-1*A)^k + w_k,2*(D_in^-1*A^T)^k)*X
- Multi-hop: K hops, learnable weights
- Output: [num_nodes, out_channels]

Implementation requirements:
1. Handle both outflow (A) and inflow (A^T)
2. Numerical stability (avoid NaN for large λ)
3. Proper gradient flow
4. Comments on mathematical meaning
5. Include __repr__ for debugging"

Attach:
  - MODULE_DEEP_LEARNING_MODELING.md (section 2: DGCN Implementation)
  - AI_CONFIG_SYSTEM_PROMPT.md (context on project)

Expected:
  - PyTorch class (80-120 lines)
  - Docstring with LaTeX formula
  - Example usage
```

---

### **Week 6: Training**

#### Pattern 5: Training Loop
```
"Setup PyTorch Lightning trainer cho SSTZIP-GNN.

Cần:
1. Custom training_step: 
   - Compute π (zero prob) and λ (Poisson mean)
   - Calculate ZIP loss
   - Log loss metrics
   
2. validation_step:
   - Same loss + MAE metric
   - Log per-cluster MAE
   
3. Callbacks:
   - EarlyStopping (monitor val_loss, patience=15)
   - ModelCheckpoint (save best 3 models)
   - MLflow logger (log params, metrics, models)

4. Configure optimizer:
   - Adam (lr=0.001)
   - ReduceLROnPlateau (factor=0.5, patience=10)

Cần handle:
- Different datasets per method (baseline, method1, method2, method3)
- Different time buckets (15min, 30min, 60min)
- Save checkpoint + model state"

Attach:
  - MODULE_DEEP_LEARNING_MODELING.md (section 6: Training Loop)
  - EXPERIMENT_TRACKING.md (section 2.2: MLflow Integration)

Expected:
  - Complete PyTorch Lightning trainer (150-200 lines)
  - Ready to use with train/val loaders
```

---

### **Week 7: Evaluation**

#### Pattern 6: Metrics & Comparison
```
"Compute evaluation metrics so sánh 4 methods.

Cho 12 experiments (4 methods × 3 time buckets), tính:

1. Core metrics:
   - MAE, RMSE, MAPE
   - Zero-Inflation Score (classification accuracy on zeros/non-zeros)
   - MASE (scale-independent)

2. Per-cluster breakdown:
   - MAE per cluster
   - Sample count per cluster
   - Mean & std demand

3. Statistical tests:
   - ANOVA: tất cả methods khác nhau?
   - Pairwise t-tests: Method X vs Method Y
   - Confidence intervals (95%)

4. Output:
   - CSV: results_comparison.csv
   - JSON: detailed metrics per method
   - Plots: method_comparison.png, cluster_performance_*.png

Acceptance criteria:
- Method 2 MAE ~3.12 (vs baseline 3.82)
- Statistical significance p < 0.05
- All plots publication-ready (300 DPI)"

Attach:
  - EXPERIMENT_TRACKING.md (section 2: Metrics Collection)
  - MODULE_EXPERIMENT_REPORTING.md (section 1: Evaluation Metrics)

Expected:
  - Python script với functions (200+ lines)
  - Generates all outputs
  - Ready for thesis figures
```

---

### **Week 8: Reporting**

#### Pattern 7: Thesis Writing
```
"Draft 'Comparative Analysis' section cho ML report.

Context:
- Method 2 (Mobility clustering) outperforms baseline by 18%
- Per-cluster analysis: downtown zones +25%, residential -5%
- Zero-Inflation Score improved 0.75 → 0.82
- Training time: 55 min (vs baseline 25 min)

Requirements:
1. Explain why Method 2 is best:
   - Spatial homogeneity benefit
   - Feature relevance (mobility > demand timing)
   
2. Discuss trade-offs:
   - Accuracy vs interpretability vs training cost
   
3. Analyze limitations:
   - Why underperforms in residential zones?
   - Cold-start problem?
   
4. Statistical rigor:
   - Reference p-values, confidence intervals
   
5. Propose next steps

Target: 500-800 words, ready for thesis"

Attach:
  - MODULE_EXPERIMENT_REPORTING.md (section 4.2: ML Module Structure)
  - Results CSV/JSON files (optional, for reference)

Expected:
  - Professional 2-3 paragraph section
  - Proper citations/references
  - Thesis-ready quality
```

---

## 4. Context Chaining: Khi Cần Multiple Steps

### **Scenario: Implement → Train → Evaluate**

```
Step 1: Implement (Day 1-2)
"Implement Method 2 clustering + feature extraction.
 [Include DATA_PIPELINE_SPECIFICATION.md + MODULE_BIG_DATA_ENGINEERING.md]"

↓ After code review ↓

Step 2: Integrate to training (Day 3)
"Connect Method 2 features to SSTZIP-GNN training.
 Features: [output from Step 1]
 Adjacency: cluster-based graph
 [Include MODULE_DEEP_LEARNING_MODELING.md + EXPERIMENT_TRACKING.md]"

↓ After training ↓

Step 3: Evaluate (Day 4-5)
"Compare Method 2 results với baseline.
 Metrics template: [reference EXPERIMENT_TRACKING.md]
 Visualizations: [include MODULE_EXPERIMENT_REPORTING.md section 3]"

↓ After analysis ↓

Step 4: Write (Day 6)
"Draft comparison section cho thesis.
 [Include MODULE_EXPERIMENT_REPORTING.md section 4.2]"
```

---

## 5. Document Attachment Quick Reference

### **Minimal Prompts** (không cần attach file)
```
✓ "What's this week's task?"
  → PROJECT_ROADMAP.md (already know từ memory)

✓ "How do I push code?"
  → COLLABORATION_GUIDELINES.md (already know)

✓ "What's the project goal?"
  → AI_CONFIG_SYSTEM_PROMPT.md (in context)
```

### **Standard Prompts** (attach 1-2 files)
```
✓ "Write Spark code for feature extraction"
  → Attach: DATA_PIPELINE_SPECIFICATION.md
  
✓ "Implement loss function"
  → Attach: MODULE_DEEP_LEARNING_MODELING.md

✓ "Compute metrics"
  → Attach: MODULE_EXPERIMENT_REPORTING.md
```

### **Complex Prompts** (attach 2-3 files)
```
✓ "End-to-end pipeline: ingest → extract → train → evaluate"
  → Attach: 
    1. DATA_PIPELINE_SPECIFICATION.md
    2. MODULE_BIG_DATA_ENGINEERING.md
    3. MODULE_DEEP_LEARNING_MODELING.md
    4. EXPERIMENT_TRACKING.md
```

---

## 6. Anti-Patterns: Cái Nên Tránh

### ❌ **Anti-Pattern 1: Attach Quá Nhiều File**
```
❌ BAD: Attach tất cả 10 files cho 1 câu hỏi
→ AI confuse, output quality giảm, token lãng phí

✓ GOOD: Attach 2-3 files liên quan trực tiếp
```

### ❌ **Anti-Pattern 2: Quá Mơ Hồ**
```
❌ BAD: "Viết code cho project"
✓ GOOD: "Viết PySpark script để aggregate taxi demand 
          theo zone + time bucket. 
          Input: Parquet, Output: CSV with columns [zone, time, demand]"
```

### ❌ **Anti-Pattern 3: Không Đủ Context**
```
❌ BAD: "Implement K-Means" (no info on data/features/K)
✓ GOOD: "Implement K-Means++ on Method 2 features (5 dimensions).
         Find optimal K [3-8] using silhouette score.
         Output: cluster assignments + silhouette plot"
```

### ❌ **Anti-Pattern 4: Yêu Cầu Quá To**
```
❌ BAD: "Viết toàn bộ model training pipeline"
       (quá dài, dễ lỗi, khó debug)

✓ GOOD: "Viết training_step cho PyTorch Lightning.
         Input: batch of [X, adj, y]
         Output: ZIP loss (binary + Poisson)"
        (focused, dễ verify, debug nhanh)
```

### ❌ **Anti-Pattern 5: Không Nêu Success Criteria**
```
❌ BAD: "Debug why model not training"
✓ GOOD: "Debug why model not training.
         Expected: val_loss should decrease
         Actual: val_loss stays at 2.1 (should be ~0.5)
         Error: RuntimeError at line 87 [show traceback]"
```

---

## 7. Practical Examples: Real Prompts

### **Example 1: Spark Feature Extraction**
```
PROMPT:
"Implement Method 1: Demand-Based Clustering feature extraction.

Data: NYC Taxi trips (Parquet)
Target: Extract demand patterns per zone

Required:
1. Group by PULocationID + hour of day
2. Compute demand (trip count) for each hour
3. Calculate statistics:
   - demand_mean
   - demand_std
   - peak_hour (which hour has most demand)
   - night_ratio (demand 0-6 / total demand)

Output: Parquet with schema [zone_id, demand_mean, demand_std, ...]

Include:
- Null handling (drop if > 5%)
- Logging (before/after record counts)
- Error handling (invalid zone IDs)
- Performance timing"

FILES TO ATTACH:
- DATA_PIPELINE_SPECIFICATION.md (section 4.2)
- MODULE_BIG_DATA_ENGINEERING.md (section 2.2)

WHAT I EXPECT:
- 100-120 line PySpark script
- Ready to run locally or on cluster
- Docstring with example
- Include config file usage
```

### **Example 2: Model Implementation**
```
PROMPT:
"Implement TCN (Temporal Convolution Network) encoder for SSTZIP-GNN.

Architecture:
- Stack of causal dilated convolutions
- 3 blocks with dilation=[1, 2, 4]
- Residual connections
- Dropout=0.2
- ReLU activation

Input: [batch, channels, time_steps]
Output: [batch, hidden_dim, time_steps]

Constraints:
1. Causal: no information from future
2. Proper gradient flow
3. Handle variable sequence lengths
4. Include __init__, forward, docstring"

FILES TO ATTACH:
- MODULE_DEEP_LEARNING_MODELING.md (section 3)

WHAT I EXPECT:
- PyTorch nn.Module class (80-100 lines)
- Includes comments on causal padding
- Example usage
- Test case
```

### **Example 3: Metrics Computation**
```
PROMPT:
"Compute evaluation metrics so sánh 4 clustering methods.

Input:
- 12 trained models (4 methods × 3 time buckets)
- Test predictions: y_pred [num_samples, num_clusters]
- Ground truth: y_true [num_samples, num_clusters]
- Cluster IDs: cluster_id [num_samples]

Output metrics:
1. Overall: MAE, RMSE, MAPE, Zero-Inflation Score
2. Per-cluster: MAE + sample count per cluster
3. Statistical: ANOVA p-value, t-test results

Acceptance criteria:
- Method 2 MAE = 3.12 ± 0.05
- p-value < 0.05 (significance)
- All plots 300 DPI

Create:
- results_comparison.csv
- figures/method_comparison.png
- figures/cluster_performance.png"

FILES TO ATTACH:
- MODULE_EXPERIMENT_REPORTING.md (section 1-2)
- EXPERIMENT_TRACKING.md (section 6)

WHAT I EXPECT:
- Python script (150-200 lines)
- Generates all outputs
- Ready for thesis
```

---

## 8. When To Ask For Help (Specific Prompts)

### **Debugging Issues**
```
"Error: 'NoneType' object has no attribute 'shape'
File: src/models/dgcn.py, line 45
Context: Computing (D_out^-1 * A)

Code:
  deg_out = adj.sum(dim=1, keepdim=True)
  adj_norm = adj / deg_out  # ERROR HERE

Expected: Normalized adjacency matrix
Actual: Getting None

Attach: Full traceback + relevant code"
```

### **Performance Issues**
```
"Spark job taking 45 min (expected: 5 min)
Data: 5M taxi records
Operation: Group by zone + hour, then K-Means

Profiling shows:
- Ingestion: 1 min ✓
- Groupby: 30 min ✗ (SLOW)
- K-Means: 1 min ✓

How to optimize the shuffle operation?

Attach: Data volume info + current code snippet"
```

### **Architecture Questions**
```
"Should I use DGCN for all methods or create different graphs?

Options:
A) One DGCN, 4 different adjacency matrices
   Pros: Code reuse. Cons: May not optimize for each method
   
B) 4 separate models
   Pros: Optimize per method. Cons: Code duplication
   
Which is better for comparison thesis?
Also reference: time/accuracy trade-offs

Attach: PROJECT_ROADMAP.md (to understand thesis goal)"
```

---

## 9. Document Versions & Updates

### **When to Refresh Prompts with Latest Docs**

✅ **Refresh after:**
- Making changes to config/parameters
- Updating feature list
- Changing model architecture
- Modifying evaluation metrics

❌ **No need to refresh:**
- Just using existing documents
- Asking clarifications on same topic
- Following established procedures

### **Tip: Save Your Good Prompts**
```
In your project:
├── prompts/
│   ├── 01_spark_etl.prompt
│   ├── 02_model_impl.prompt
│   ├── 03_metrics.prompt
│   └── 04_reporting.prompt

Then reuse with small modifications.
Example:
"Use this prompt template + update feature list to [...]"
```

---

## 10. Checklist: Before Hitting "Send"

- [ ] Prompt is specific & not mơ hồ?
- [ ] Attached 1-3 relevant files (not 0, not 10)?
- [ ] Nêu acceptance criteria / success metrics?
- [ ] Đã tìm answer trong documents rồi không?
- [ ] Prompt < 500 words (concise)?
- [ ] Nêu context nếu cần (data size, constraints)?
- [ ] Mention expected output format?
- [ ] Include error traces nếu debugging?

---

## 11. Quick Prompt Templates Copy-Paste

### **Template 1: Feature Extraction**
```
[METHOD NAME] feature extraction

Data source: [INPUT]
Target features: [LIST 3-5 features]
Per unit: [zone/cluster/etc]
Normalization: [yes/no + type]
Output format: [Parquet/CSV/etc]

Include: error handling, logging, performance timing

Attach: DATA_PIPELINE_SPECIFICATION.md (section 4.[X])
        MODULE_BIG_DATA_ENGINEERING.md (section 2.[X])
```

### **Template 2: Model Component**
```
Implement [LAYER NAME] for SSTZIP-GNN

Architecture: [INPUT SIZE] → [OUTPUT SIZE]
Requirements: [LIST 3-5 requirements]

Constraints:
- [numerical stability / gradient flow / etc]
- [performance requirement]

Include: docstring, example, test case

Attach: MODULE_DEEP_LEARNING_MODELING.md (section [X])
```

### **Template 3: Evaluation**
```
Compute metrics for [METHOD/ALL METHODS]

Metrics: [LIST: MAE, RMSE, ...]
Granularity: [overall/per-cluster/per-timebucket]
Statistical tests: [ANOVA/t-test/confidence intervals]
Outputs: [CSV/plots/JSON]

Acceptance criteria:
- [Performance target]
- [Statistical significance]
- [Output format]

Attach: MODULE_EXPERIMENT_REPORTING.md (section [X])
        EXPERIMENT_TRACKING.md (section [X])
```

### **Template 4: Debugging**
```
[ERROR NAME]: [ERROR MESSAGE]

Location: [FILE:LINE]
Context: [WHAT WERE YOU DOING]

Code causing error:
[CODE SNIPPET]

Expected: [WHAT SHOULD HAPPEN]
Actual: [WHAT ACTUALLY HAPPENED]

Attach: Relevant document + full traceback
```

---

## Summary: The 80/20 Rule

**80% hiệu quả đạt từ:**

1. ✅ **Attach đúng 2-3 files** (không quá nhiều)
2. ✅ **Nêu cụ thể requirements** (không mơ hồ)
3. ✅ **Mention acceptance criteria** (when is it done?)
4. ✅ **Provide context** (data size, constraints, goals)
5. ✅ **Reference specific sections** (section 2.3 of file X)

**Remaining 20%** từ fine-tuning + iterations

---

**Good luck with your prompts!** 🚀

Lưu bộ prompt templates này và reuse cho mỗi task.
