# 📋 Project Roadmap: Taxi Demand Prediction

**Duration:** < 2 months (8 weeks)  
**Team:** 2 people  
**Objective:** Compare 4 clustering methods for taxi demand prediction (Baseline + 3 Methods)

---

## 🎯 Phase 1: Setup & Data Pipeline (Week 1-2)

### Week 1: Environment & Infrastructure
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **1.1** Set up Git repo, Spark cluster, Cassandra cluster | Person A | 1.5 days | Docker Cassandra 3-node cluster running |
| **1.2** Data ingestion: NYC Taxi dataset (raw → Parquet) | Person A | 2 days | 5M+ records in Parquet format |
| **1.3** Exploratory Data Analysis (EDA) & data profiling | Person B | 2 days | EDA report + data quality assessment |
| **1.4** Define feature schema & naming conventions | Both | 0.5 day | FEATURE_REGISTRY.md |
| **1.5** Cassandra schema creation & validation | Person A | 1 day | taxi_db keyspace with taxi_demand & cluster_metadata tables |

### Week 2: Baseline Method & Feature Engineering
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **2.1** Baseline: Zone-based aggregation (existing zones) | Person A | 2 days | `baseline_features.parquet` |
| **2.2** Outlier detection (Z-score) & data cleaning | Person B | 2 days | Cleaned dataset |
| **2.3** Feature validation & distribution analysis | Both | 1 day | Baseline features ready for modeling |

**Milestone 1 ✅:** Baseline features ready, EDA complete

---

## 📊 Phase 2: Method 1 & 2 Implementation (Week 3-4)

### Week 3: Method 1 - Demand-Based Clustering
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **3.1** Extract demand patterns (demand_count by hour, zone) | Person A | 1.5 days | Time-series demand matrix |
| **3.2** K-Means clustering on demand patterns | Person A | 1.5 days | Optimal K via Silhouette analysis |
| **3.3** Generate clustered features & validate | Person B | 1 day | `method1_clusters.pkl` + cluster quality report |

### Week 4: Method 2 - Mobility Pattern Clustering ⭐ (Main)
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **4.1** Extract 5 features: trip_count, avg_distance, avg_fare, avg_passenger, duration | Person A | 1.5 days | Feature matrix for all zones |
| **4.2** Feature normalization & dimension reduction (PCA if needed) | Person B | 1 day | Normalized feature matrix |
| **4.3** K-Means++ clustering + Silhouette analysis | Person A | 1 day | `method2_clusters.pkl` + cluster interpretation |
| **4.4** **Write Method 2 features to Cassandra** | Person A | 1 day | Features persisted with RF=3, QUORUM consistency |
| **4.5** Label clusters (downtown, airport, residential, etc.) | Person B | 0.5 day | Cluster semantics document |

**Milestone 2 ✅:** Method 1 & 2 clusters ready

---

## 🔄 Phase 3: Method 3 & Aggregation (Week 5)

### Week 5: Method 3 - OD-Flow Based Clustering
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **5.1** Extract inflow/outflow per zone + top destinations | Person A | 1.5 days | OD-Flow matrix |
| **5.2** Combine with Method 2 features (enriched feature set) | Person B | 1 day | Enhanced feature matrix |
| **5.3** K-Means clustering + validation | Both | 1 day | `method3_clusters.pkl` + analysis |
| **5.4** Create unified feature dataset for all 4 methods | Person A | 0.5 day | `all_methods_features.parquet` |

**Milestone 3 ✅:** All clustering methods complete

---

## 🧠 Phase 4: Deep Learning Modeling (Week 5-6)

### Week 5-6: SSTZIP-GNN Implementation & Training
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **6.1** Implement DGCN (Spatial layer) + TCN (Temporal layer) | Person A | 2 days | PyTorch model skeleton |
| **6.2** Implement ZIP head (binary + Poisson) & custom loss | Person B | 1.5 days | Loss function tested |
| **6.3** Setup PyTorch Lightning trainer + callbacks | Person A | 1 day | Training pipeline ready |
| **6.4** Train on all 4 methods (parallel training) | Both | 3 days | 4 trained models + logs |

**Milestone 4 ✅:** All models trained

---

## 📈 Phase 5: Evaluation & Comparison (Week 7)

### Week 7: Experimental Analysis
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **7.1** Compute MAE, RMSE, Zero-Inflation Score for all 4 methods | Person A | 1.5 days | Metrics comparison table |
| **7.2** Analyze computational efficiency (training time, inference speed) | Person B | 1 day | Performance report |
| **7.3** Statistical significance testing (confidence intervals) | Both | 1 day | Statistical analysis |
| **7.4** Generate comparison visualizations & insights | Person B | 1 day | Figures & interpretations |

**Milestone 5 ✅:** Results ready for thesis

---

## 📝 Phase 6: Distributed Database Analysis (Week 6)

### Week 6: Cassandra Distributed System Evaluation
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **6.1** Test fault tolerance: kill nodes 1-at-a-time | Person A | 1 day | Fault tolerance report + write/read availability data |
| **6.2** Analyze consistency levels: latency vs safety trade-offs | Person A | 1 day | Performance benchmarks (ONE vs QUORUM vs ALL) |
| **6.3** Read queries: test time-range queries, scalability | Person B | 1 day | Query performance analysis + optimization recommendations |
| **6.4** Distributed system analysis: CAP theorem, replication, recovery | Both | 1 day | Architecture analysis document |

**Milestone 6 ✅:** Cassandra analysis complete & thesis ready

**Milestone 7 ✅:** Thesis complete & submitted (2 courses: Big Data + Distributed Database)

---

## ⚠️ Critical Path & Dependencies

```
Week 1-2 (Setup) 
    ↓
Week 3 (Method 1) ← Week 4 (Method 2) 
    ↓
Week 5 (Method 3) + Week 5-6 (Modeling)
    ↓
Week 7 (Evaluation)
    ↓
Week 7-8 (Reporting)
```

**Critical Dependencies:**
- ✅ Week 1-2 must complete before Week 3
- ✅ Methods 1-3 can run in parallel (Week 3-5)
- ✅ Modeling can start when any 2 methods are ready
- ⚠️ If Method 2 delays, use Method 1 + Baseline as fallback

---

## 👥 Team Allocation Strategy

### Person A (Data Engineer Focus)
- Weeks 1-2: Spark ETL, data pipelines, infrastructure
- Weeks 3-5: Feature extraction (all methods), Spark jobs
- Week 6: Model training & logging infrastructure
- Week 8: Big Data report + code documentation

### Person B (ML/Analytics Focus)
- Weeks 1-2: EDA, data quality, feature validation
- Weeks 3-5: Clustering validation, cluster interpretation
- Week 6: Model architecture implementation, optimization
- Week 7-8: Evaluation, analysis, thesis writing

### Shared Tasks
- Weekly sync-ups (30 min) every Monday
- Code reviews (GitHub PRs)
- Experimental results sharing

---

## 📊 Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| All 4 methods implemented | 100% | ⏳ |
| Models trained (4 variants) | 4/4 | ⏳ |
| Metrics comparison complete | ✓ | ⏳ |
| Big Data report (20-30 pages) | ✓ | ⏳ |
| ML report (20-30 pages) | ✓ | ⏳ |
| Presentation ready | ✓ | ⏳ |
| Code reproducible & documented | ✓ | ⏳ |

---

## 📌 Key Assumptions & Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Spark cluster setup delays | High | Pre-configure cloud VM (AWS/GCP) by Day 1 |
| Dataset too large/corrupted | High | Sample 1M records first, validate schema |
| Model training slow | Medium | Use distributed training, reduce epochs initially |
| Clustering methods underperform | Medium | Have ensemble approach ready as backup |
| Time zone/aggregation bugs | High | Unit tests + validation logic in Week 2 |

---

## 📞 Communication & Checkpoints

- **Every Monday:** Team sync (current progress, blockers, next week plan)
- **Every Thursday:** Code review & experiment logging
- **Milestone reviews:** After each major phase completion
- **Emergency escalation:** Use Slack/Email for blocking issues
