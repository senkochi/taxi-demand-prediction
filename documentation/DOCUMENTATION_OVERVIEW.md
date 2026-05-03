# 📚 Documentation System Overview

**Last Updated:** May 3, 2026  
**Project:** Taxi Demand Prediction with Clustering Methods Comparison  
**Team:** 2 people | **Duration:** < 2 months

---

## 📋 Files in This Documentation System

### Core Documents (Read First)
1. **PROJECT_ROADMAP.md** ⭐
   - 8-week timeline with weekly milestones
   - Task allocation (Person A & B)
   - Critical dependencies & risk management
   - Success criteria & checkpoints

2. **COLLABORATION_GUIDELINES.md** 👥
   - Team roles & responsibilities
   - Communication protocol
   - Git workflow & code review process
   - Testing & validation procedures
   - Conflict resolution strategies

3. **DEVELOPMENT_SETUP.md** 🛠️
   - Repository structure & organization
   - Environment setup (local/Docker/cloud)
   - Dependencies & requirements
   - Configuration management
   - Development workflows & quick commands

### Technical Specifications
4. **DATA_PIPELINE_SPECIFICATION.md** 🔄
   - End-to-end data flow
   - Input data schema & validation
   - Feature extraction for all 4 methods
   - Temporal aggregation (15min, 30min, 60min)
   - Train/val/test split strategy
   - Output formats & storage options

5. **MODULE_BIG_DATA_ENGINEERING.md** 📦
   - Spark ETL implementation (Weeks 1-2)
   - Data quality checks & outlier detection
   - Feature extraction code (all 4 methods)
   - K-Means clustering with silhouette analysis
   - Performance optimization & benchmarking
   - Database storage recommendations

6. **MODULE_DEEP_LEARNING_MODELING.md** 🧠
   - SSTZIP-GNN architecture overview
   - DGCN spatial layer implementation
   - TCN temporal layer implementation
   - Zero-Inflated Poisson (ZIP) loss function
   - Full model integration with PyTorch Lightning
   - Training loop & inference pipeline

7. **EXPERIMENT_TRACKING.md** 📊
   - MLflow tracking setup
   - Metrics collection & logging framework
   - Experiment naming conventions
   - Per-cluster metrics computation
   - Comparison framework & visualization
   - Results aggregation template

### System Prompts (AI Configuration)
8. **AI_CONFIG_SYSTEM_PROMPT.md** 🤖
   - Role definition: Senior Spatiotemporal AI Researcher
   - Project identity & core tasks
   - Theoretical anchors & constraints
   - Communication protocols

---

## 🗺️ Quick Navigation by Role

### For Person A (Data Engineering Lead)
Start here:
1. **PROJECT_ROADMAP.md** → Week 1-6 tasks
2. **DEVELOPMENT_SETUP.md** → Environment setup
3. **DATA_PIPELINE_SPECIFICATION.md** → Data flow understanding
4. **MODULE_BIG_DATA_ENGINEERING.md** → Implementation details

**Key files to create/modify:**
- `scripts/01_data_ingestion.py`
- `scripts/02_data_validation.py`
- `scripts/03-06_method_clustering.py` (all 4 methods)
- `src/features/` module (all feature extractors)
- `config/spark_config.yaml`

**Deliverables:**
- Week 2: Data ingestion & baseline features ✓
- Week 5: All feature extraction complete ✓
- Week 6: Trained clustering models ✓
- Week 8: Big Data report + documentation ✓

---

### For Person B (ML/Analytics Lead)
Start here:
1. **PROJECT_ROADMAP.md** → Week 1-8 tasks
2. **DEVELOPMENT_SETUP.md** → Environment setup
3. **MODULE_DEEP_LEARNING_MODELING.md** → Model implementation
4. **EXPERIMENT_TRACKING.md** → Evaluation & tracking
5. **MODULE_EXPERIMENT_REPORTING.md** → Reporting & visualization

**Key files to create/modify:**
- `src/models/dgcn.py`, `tcn.py`, `zip_head.py`
- `src/models/sstzip_gnn.py` (full model)
- `src/training/trainer.py` (PyTorch Lightning)
- `src/evaluation/metrics.py`, `visualization.py`
- `scripts/08_model_training.py`
- `scripts/09_evaluation.py`
- `notebooks/01_eda.ipynb`, `03_results_analysis.ipynb`

**Deliverables:**
- Week 2: EDA & data profiling ✓
- Week 5: Model architecture complete ✓
- Week 6: All 4 models trained ✓
- Week 7: Evaluation & comparison complete ✓
- Week 8: ML report + presentation slides ✓

---

### For Both Team Members
- **COLLABORATION_GUIDELINES.md** → Communication & workflows
- **EXPERIMENT_TRACKING.md** → Logging & comparing results
- **PROJECT_ROADMAP.md** → Sync-ups & milestones

---

## 🎯 Document Purposes & Key Sections

### PROJECT_ROADMAP.md
```
Purpose: Single source of truth for timeline, milestones, and dependencies
Use when: Planning sprint, tracking progress, identifying blockers
Key sections:
  - 8-week timeline (phases 1-6)
  - Task allocation with durations
  - Critical dependencies
  - Risk mitigation strategies
  - Success criteria
```

### COLLABORATION_GUIDELINES.md
```
Purpose: Define team structure, communication, and workflows
Use when: Setting up workflows, resolving conflicts, onboarding
Key sections:
  - Team roles (Person A vs Person B)
  - Communication protocol (sync, async, response times)
  - Git workflow (branch strategy, PR process, code review)
  - Task allocation & sprint planning
  - Testing & validation responsibilities
  - Knowledge handoff procedures
```

### DEVELOPMENT_SETUP.md
```
Purpose: Environment setup and development infrastructure
Use when: Setting up local dev environment, onboarding new member
Key sections:
  - Repository structure (folder layout)
  - Environment setup (Python venv, Spark, Docker)
  - Dependencies (requirements.txt, environment.yml)
  - Configuration management (config.yaml)
  - Development workflows (running scripts, testing)
```

### DATA_PIPELINE_SPECIFICATION.md
```
Purpose: Define data formats, transformations, and specifications
Use when: Working with data, understanding schema, debugging data issues
Key sections:
  - End-to-end data flow diagram
  - Input schema & validation rules
  - Feature extraction methodology (all 4 methods)
  - Temporal aggregation specifications
  - Output formats & storage options
  - Data versioning & tracking
```

### MODULE_BIG_DATA_ENGINEERING.md
```
Purpose: Detailed implementation guide for data engineering tasks
Use when: Implementing Spark ETL, clustering algorithms
Key sections:
  - Spark ETL specifications with code
  - Feature extraction for all 4 methods
  - K-Means clustering implementation
  - Performance optimization & benchmarking
  - Database design recommendations
```

### MODULE_DEEP_LEARNING_MODELING.md
```
Purpose: Deep learning architecture and implementation
Use when: Implementing SSTZIP-GNN, training models
Key sections:
  - Architecture overview (DGCN, TCN, ZIP)
  - Layer implementations with PyTorch code
  - Loss function derivation & implementation
  - Full model integration
  - Training loop with PyTorch Lightning
  - Evaluation & inference procedures
```

### EXPERIMENT_TRACKING.md
```
Purpose: Define experiment logging, metrics, and comparison
Use when: Running experiments, tracking results, comparing methods
Key sections:
  - MLflow setup & integration
  - Metrics collection & logging
  - Naming conventions & tags
  - Comparison framework
  - Visualization utilities
  - Results aggregation template
```

### MODULE_EXPERIMENT_REPORTING.md
```
Purpose: Evaluation methodology and thesis reporting structure
Use when: Evaluating models, writing reports, creating visualizations
Key sections:
  - Evaluation metrics (MAE, RMSE, Zero-Inflation Score)
  - Per-cluster analysis
  - Comparative analysis with statistical testing
  - Visualization templates
  - Thesis report structure (20-30 pages each)
  - Presentation outline
```

---

## 📁 File Dependencies (Read Order)

```
Level 1: Overall Understanding
├─ PROJECT_ROADMAP.md (Timeline & milestones)
└─ COLLABORATION_GUIDELINES.md (Team structure)

Level 2: Setup & Infrastructure
├─ DEVELOPMENT_SETUP.md (Environment)
└─ DATA_PIPELINE_SPECIFICATION.md (Data understanding)

Level 3: Implementation (Parallel tracks)
├─ MODULE_BIG_DATA_ENGINEERING.md (Person A focus)
├─ MODULE_DEEP_LEARNING_MODELING.md (Person B focus)
└─ AI_CONFIG_SYSTEM_PROMPT.md (AI context)

Level 4: Evaluation & Reporting
├─ EXPERIMENT_TRACKING.md (Experiment management)
└─ MODULE_EXPERIMENT_REPORTING.md (Results & thesis)
```

---

## 🚀 Implementation Workflow

### Week 1-2: Setup Phase
1. Read: **PROJECT_ROADMAP.md** + **COLLABORATION_GUIDELINES.md**
2. Setup: **DEVELOPMENT_SETUP.md**
3. Understand: **DATA_PIPELINE_SPECIFICATION.md**
4. Both: Initial sync-up (30 min)

### Week 3-5: Implementation Phase
**Person A:**
- Implement ETL using **MODULE_BIG_DATA_ENGINEERING.md**
- Create scripts 01-06
- Log progress in **EXPERIMENT_TRACKING.md**

**Person B:**
- Implement model using **MODULE_DEEP_LEARNING_MODELING.md**
- Prepare evaluation framework
- Both: Weekly syncs (PROJECT_ROADMAP.md schedule)

### Week 6: Modeling Phase
- Person A: Model training infrastructure
- Person B: Training & optimization
- Both: Use **EXPERIMENT_TRACKING.md** for logging

### Week 7: Evaluation Phase
**Person B focus:**
- Compute metrics (**MODULE_EXPERIMENT_REPORTING.md**)
- Generate visualizations
- Statistical analysis
- Create comparison tables

### Week 8: Reporting Phase
**Person A:**
- Draft Big Data report (20-30 pages)
- Code documentation

**Person B:**
- Draft ML/DL report (20-30 pages)
- Create presentation slides

**Both:**
- Final review using **COLLABORATION_GUIDELINES.md**
- Repository cleanup using **DEVELOPMENT_SETUP.md**

---

## 📊 Status Tracking

### Document Completion
- [x] PROJECT_ROADMAP.md (8-week detailed plan)
- [x] COLLABORATION_GUIDELINES.md (Team workflows)
- [x] DEVELOPMENT_SETUP.md (Env & repo structure)
- [x] DATA_PIPELINE_SPECIFICATION.md (Data flow & schema)
- [x] MODULE_BIG_DATA_ENGINEERING.md (Spark implementation)
- [x] MODULE_DEEP_LEARNING_MODELING.md (Model architecture)
- [x] EXPERIMENT_TRACKING.md (Metrics & comparison)
- [x] MODULE_EXPERIMENT_REPORTING.md (Evaluation & reporting)
- [x] AI_CONFIG_SYSTEM_PROMPT.md (AI context)

### Documentation Quality
- ✅ Each document has clear purpose & scope
- ✅ Code examples included (PySpark, PyTorch)
- ✅ Implementation templates provided
- ✅ Prompt templates for AI assistance
- ✅ Cross-references between documents
- ✅ Practical, action-oriented (not just theory)

---

## 💡 How to Use This System Effectively

### For Daily Work
1. Consult **PROJECT_ROADMAP.md** for current week's tasks
2. Refer to specific MODULE documents for implementation
3. Use **EXPERIMENT_TRACKING.md** for logging results
4. Check **COLLABORATION_GUIDELINES.md** for Git/PR procedures

### For Debugging / Troubleshooting
1. **Data issues** → **DATA_PIPELINE_SPECIFICATION.md** (section 8: QA Checklist)
2. **Spark errors** → **MODULE_BIG_DATA_ENGINEERING.md** (section 7: Troubleshooting)
3. **Model training issues** → **MODULE_DEEP_LEARNING_MODELING.md** + **DEVELOPMENT_SETUP.md**
4. **Metrics/results problems** → **EXPERIMENT_TRACKING.md** + **MODULE_EXPERIMENT_REPORTING.md**
5. **Team/communication issues** → **COLLABORATION_GUIDELINES.md** (section 7: Conflict Resolution)

### For Reporting Progress
1. Weekly: Update **PROJECT_ROADMAP.md** checklist
2. Create progress summary using **EXPERIMENT_TRACKING.md**
3. Review team sync notes using **COLLABORATION_GUIDELINES.md**
4. Final: Use **MODULE_EXPERIMENT_REPORTING.md** structure for thesis

### For Onboarding (if team member joins)
1. Read **README.md** (project overview)
2. Read **PROJECT_ROADMAP.md** (understand timeline)
3. Read **COLLABORATION_GUIDELINES.md** (team practices)
4. Follow **DEVELOPMENT_SETUP.md** (setup environment)
5. Review last 5 PRs to see current state
6. 30-min sync with existing team member

---

## 🎓 Practical Examples

### Example 1: Week 3 Tuesday, 3PM - "What should I work on?"
```
1. Open PROJECT_ROADMAP.md
2. Check "Week 3: Method 1 - Demand-Based Clustering" section
3. See task 3.2: "K-Means clustering on demand patterns"
4. Open MODULE_BIG_DATA_ENGINEERING.md
5. Go to section 2.2 "Method 1: Demand-Based Clustering Features"
6. Use the code template provided
7. Log progress/issues in EXPERIMENT_TRACKING.md
```

### Example 2: Code Review Comment - "This function crashes with nulls"
```
1. Check DATA_PIPELINE_SPECIFICATION.md section 3.2 (validation rules)
2. Add null handling based on strategy defined there
3. Refer to MODULE_BIG_DATA_ENGINEERING.md code examples for Spark null handling
4. Update PR with fix
5. Reference DATA_PIPELINE_SPECIFICATION.md section 10 (QA Checklist) in PR description
```

### Example 3: Week 7 - "Model X is underperforming, why?"
```
1. Check MODULE_EXPERIMENT_REPORTING.md section 2 (Comparative Analysis)
2. Compute per-cluster metrics (section 2.2)
3. Compare against other methods (section 4: Results Summary)
4. Use visualization templates (section 3) to understand patterns
5. Draft explanation for discussion section (section 4.2, ML Module)
```

### Example 4: "Need to write Big Data report, where do I start?"
```
1. Read MODULE_EXPERIMENT_REPORTING.md section 4.1 (Big Data Module structure)
2. Open MODULE_BIG_DATA_ENGINEERING.md to gather technical details
3. Use DATA_PIPELINE_SPECIFICATION.md for data architecture sections
4. Refer to PROJECT_ROADMAP.md section "Phase 1-5" for insights on each phase
5. Use EXPERIMENT_TRACKING.md results for performance analysis
6. Create visualizations using templates in MODULE_EXPERIMENT_REPORTING.md section 3
```

---

## 🔗 Cross-Reference Map

### Common Questions → Go To:

| Question | Document(s) | Section |
|----------|------------|---------|
| What should I work on this week? | PROJECT_ROADMAP.md | Current week phase |
| How do I set up my environment? | DEVELOPMENT_SETUP.md | Section 2-3 |
| What's the data schema? | DATA_PIPELINE_SPECIFICATION.md | Section 2 |
| How do I implement feature extraction? | MODULE_BIG_DATA_ENGINEERING.md | Section 2 |
| How do I implement the model? | MODULE_DEEP_LEARNING_MODELING.md | Section 2-5 |
| How do I track experiments? | EXPERIMENT_TRACKING.md | Section 2-3 |
| What metrics should I compute? | MODULE_EXPERIMENT_REPORTING.md | Section 1 |
| How do I write the thesis report? | MODULE_EXPERIMENT_REPORTING.md | Section 4 |
| How do I push code & PRs? | COLLABORATION_GUIDELINES.md | Section 3 |
| What's our team communication protocol? | COLLABORATION_GUIDELINES.md | Section 2 |
| What if I find a bug/blocker? | COLLABORATION_GUIDELINES.md | Section 7 |

---

## ✅ Pre-Submission Checklist

Use this when approaching final deadline:

**Code & Repository:**
- [ ] All scripts in `scripts/` folder complete
- [ ] All modules in `src/` folder organized
- [ ] Unit tests in `tests/` folder (>80% coverage)
- [ ] `README.md` updated with project summary
- [ ] No secrets, API keys, or hardcoded paths in code
- [ ] `.gitignore` properly configured

**Data & Experiments:**
- [ ] All 12 experiments logged in MLflow (4 methods × 3 time buckets)
- [ ] Results exported to CSV (EXPERIMENT_TRACKING.md section 6.1)
- [ ] Statistical significance tested (section 2.1)
- [ ] All visualizations saved as high-res PNG (300 DPI)

**Documentation:**
- [ ] This overview document (DOCUMENTATION_OVERVIEW.md) ✓
- [ ] PROJECT_ROADMAP.md milestones completed
- [ ] COLLABORATION_GUIDELINES.md procedures followed
- [ ] DEVELOPMENT_SETUP.md setup instructions verified
- [ ] DATA_PIPELINE_SPECIFICATION.md specifications documented
- [ ] MODULE files complete with code examples
- [ ] EXPERIMENT_TRACKING.md results compiled

**Reports:**
- [ ] Big Data Module report (20-30 pages, Person A)
- [ ] ML/DL Module report (20-30 pages, Person B)
- [ ] Presentation slides (15-20 minutes)
- [ ] Figures in reports are publication-ready

**Final Review:**
- [ ] Both team members reviewed reports
- [ ] Results reproducible (same seed → same results)
- [ ] No merge conflicts
- [ ] All dependencies documented in requirements.txt
- [ ] Final git commit with "Release v1.0" tag

---

## 📞 Support & Resources

**Within This System:**
- Troubleshooting guides in MODULE documents (section 7-8)
- Prompt templates for AI assistance (each module has section 7-8)
- Code examples in all technical modules

**External Resources:**
- PySpark docs: https://spark.apache.org/docs/latest/
- PyTorch: https://pytorch.org/docs/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/

---

## 🎯 Success Definition

**Project successful when:**
1. ✅ All 4 clustering methods implemented & compared
2. ✅ SSTZIP-GNN model trained on 4 variants
3. ✅ Results show clear method rankings (Method 2 preferred)
4. ✅ 18%+ improvement over baseline demonstrated
5. ✅ Statistical significance established (p < 0.05)
6. ✅ Two independent 20-30 page thesis modules completed
7. ✅ 15-20 minute presentation ready
8. ✅ All code reproducible & documented
9. ✅ Submitted on time with clean repository

---

**Last Updated:** May 3, 2026  
**Maintained By:** Both team members  
**Next Review:** Every week (PROJECT_ROADMAP.md sync)
