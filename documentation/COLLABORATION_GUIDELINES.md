# 👥 Team Collaboration Guidelines

**Purpose:** Define team structure, communication, code review process, and best practices for 2-person project

---

## 1. Team Structure & Roles

### Overview
- **Duration:** < 2 months (8 weeks)
- **Team Size:** 2 people
- **Coordination:** Weekly syncs + asynchronous collaboration via Git

### Role Definition

#### Person A: **Data Engineering Lead**
**Primary Focus:** Infrastructure, data pipelines, scalability

**Responsibilities:**
- Week 1-2: Spark cluster setup, data ingestion, ETL pipelines
- Week 3-5: Feature extraction (all methods)
- Week 6: Model training infrastructure, distributed training setup
- Week 8: Big Data module report, code documentation

**Skills:** PySpark, Hadoop, distributed systems, database design

**Deliverables:**
- Data ingestion scripts
- Feature extraction modules (for all 4 methods)
- Model training framework
- Distributed database setup
- Big Data report (20-30 pages)

#### Person B: **ML/Analytics Lead**
**Primary Focus:** Modeling, analysis, visualization

**Responsibilities:**
- Week 1-2: EDA, data profiling, quality checks
- Week 3-5: Clustering analysis, cluster interpretation, validation
- Week 6: Model architecture, loss functions, optimization
- Week 7-8: Evaluation, comparison, visualization, ML report

**Skills:** Machine Learning, Deep Learning, Data Science, Statistics

**Deliverables:**
- EDA report & analysis
- Cluster interpretation & semantics
- SSTZIP-GNN implementation
- Evaluation metrics & comparison
- ML/DL module report (20-30 pages)
- Publication-ready visualizations

### Shared Responsibilities
- Code review (PRs)
- Weekly sync-ups
- Problem-solving & blockers
- Testing & validation

---

## 2. Communication Protocol

### 2.1 Synchronous Communication (Meetings)

**Weekly Sync (Monday, 30 min)**
- **Time:** Every Monday, 09:00
- **Duration:** 30 minutes
- **Format:** Video call (Zoom/Teams)
- **Agenda:**
  1. Progress since last week (5 min per person)
  2. Blockers & challenges (5 min)
  3. Next week priorities (5 min)
  4. Knowledge sharing / technical discussion (10 min)

**Example Agenda:**
```
[Week 3 Sync]
- Person A: Completed data ingestion, 5M records loaded. Starting clustering code.
- Person B: Finished EDA, identified 3 seasonal patterns. Ready for Method 1 validation.
- Blocker: Spark memory overflow on raw data → use sampling approach
- Next: Both start clustering implementations in parallel
```

**Ad-hoc Standups (as needed)**
- For blocking issues that need immediate resolution
- Use Slack/Teams for quick resolution

### 2.2 Asynchronous Communication (Chat)

**Slack/Teams Channel: #taxi-prediction**

**Channel Conventions:**
- **#general:** Project announcements, timeline updates
- **#questions:** Technical Q&A
- **#blockers:** Issues blocking progress (needs response within 2 hours)
- **#results:** Share intermediate results, plots, findings
- **#review:** Code review requests

**Message Protocol:**
```
Good async message:
❌ "Hey can you check my code?"
✅ "@Person_A I've pushed feature/method2-optimization to review branch.
   Main changes: normalized features, added silhouette analysis.
   PR: https://github.com/repo/pull/5"

Quick question:
❌ "How do I compute MAE?"
✅ "@Person_B Quick Q: For zero-inflated data, should MAE include or exclude structural zeros?"
```

**Response Time Expectations:**
- Blockers: 2 hours
- Questions: 4-6 hours (same day if possible)
- Code reviews: 24 hours
- General messages: best effort

### 2.3 Documentation & Knowledge Sharing

**Shared Google Doc / Notion Page:** Project Notes
- Weekly summaries
- Key learnings & tips
- Dataset insights
- Hyperparameter tuning results

**GitHub Wiki:** For permanent documentation
- Project architecture
- Data schema explanations
- Tutorial links

---

## 3. Version Control & Git Workflow

### 3.1 Branch Strategy

```
main (stable, always tested)
├── develop (integration branch)
│   ├── feature/data-ingestion (Person A)
│   ├── feature/method1-clustering (Person B)
│   ├── feature/method2-clustering (Person B)
│   ├── feature/model-architecture (Person A)
│   └── bugfix/spark-memory-issue
└── hotfix/urgent-fix (if needed)
```

### 3.2 Commit Message Convention

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Add/modify tests
- `docs`: Documentation changes
- `chore`: Build, dependencies, tooling

**Examples:**
```
feat(clustering): implement k-means for method1
  - Extract demand patterns by hour
  - Add silhouette score analysis
  - Tested on 2M records, completed in 3min

fix(features): handle null values in mobility features
  - Previous: crashes if trip_distance is null
  - Now: impute with zone median

perf(spark): optimize groupby operation
  - Reduced execution time from 8min to 2min
  - Used broadcast join instead of shuffle

docs(readme): add quick start guide
```

### 3.3 Pull Request (PR) Process

**Step 1: Push feature branch**
```bash
git checkout -b feature/method2-clustering
git add src/features/mobility_features.py
git commit -m "feat(features): implement mobility pattern extraction"
git push origin feature/method2-clustering
```

**Step 2: Create PR with template**
```markdown
## Description
Implements Method 2: Mobility Pattern Clustering
Extracts 5 features: trip_count, avg_distance, avg_fare, avg_passenger, duration

## Changes
- [ ] New module: src/features/mobility_features.py
- [ ] Tests: tests/unit/test_mobility_features.py
- [ ] Docs: Update DATA_PIPELINE_SPECIFICATION.md

## Type of change
- [ ] New feature
- [ ] Bug fix
- [ ] Performance improvement

## Testing
Tested on 2M records, silhouette score: 0.68
Execution time: 3.2 minutes

## Checklist
- [x] Code follows PEP 8 (`black`, `isort` clean)
- [x] No new linting errors (`flake8` clean)
- [x] All tests pass
- [x] Docstrings added
- [x] Config-driven (no hardcoded paths)
- [ ] Performance impact analyzed
```

**Step 3: Code Review**
- Reviewer: The other team member
- Turnaround: within 24 hours
- Use GitHub comments for discussion

**Step 4: Address feedback**
```bash
# Make changes based on review
git add src/features/mobility_features.py
git commit -m "refactor(features): clarify docstring per review feedback"
git push origin feature/method2-clustering
# PR auto-updates
```

**Step 5: Merge**
```bash
# After approval
git checkout develop
git pull origin develop
git merge --no-ff feature/method2-clustering
git push origin develop

# Delete branch
git push origin --delete feature/method2-clustering
git branch -d feature/method2-clustering
```

### 3.4 Code Review Checklist

Reviewer should verify:

- [ ] **Functionality**: Does code do what it claims?
- [ ] **Style**: Follows PEP 8, naming conventions
- [ ] **Tests**: New tests added, all pass
- [ ] **Performance**: No obvious inefficiencies (e.g., nested loops, memory leaks)
- [ ] **Documentation**: Docstrings, comments for complex logic
- [ ] **Configuration**: Uses config.yaml, no hardcoded values
- [ ] **Error Handling**: Handles edge cases, null values
- [ ] **Data Integrity**: No data loss or corruption
- [ ] **Logging**: Appropriate log statements for debugging

**Review Comment Examples:**
```
👍 Good:
"Clean implementation! The silhouette analysis is well-structured."

💭 Suggestion:
"This loop could be vectorized with NumPy for better performance.
 Example: np.array(list) instead of for loop"

🐛 Issue:
"This will crash if trip_distance is null. Add validation or imputation."
```

---

## 4. Task Allocation & Scheduling

### 4.1 Task Board (GitHub Projects or Notion)

```
┌─────────────────────────────────────────────────────────┐
│ BACKLOG    │ TODO      │ IN PROGRESS │ REVIEW │ DONE   │
├─────────────┼───────────┼─────────────┼────────┼────────┤
│             │ [Week 1]  │ [Week 3]    │        │ ✅     │
│             │ Spark     │ Method1     │        │ EDA    │
│             │ Setup     │ (Person B)  │        │        │
│             │           │             │        │        │
│ [Week 6]    │ [Week 5]  │ [Week 5]    │        │ ✅     │
│ Evaluation  │ Method3   │ Method2     │        │ Method │
│             │           │ (Person B)  │        │ 1      │
└─────────────┴───────────┴─────────────┴────────┴────────┘
```

### 4.2 Sprint Planning (Weekly)

**Every Monday after sync:**

```
Week 3 Sprint
│
├─ Person A
│  ├─ Task: Method 1 feature extraction
│  ├─ Effort: 3 days
│  ├─ Dependency: Cleaned data (✅ ready)
│  └─ Deliverable: method1_features.parquet
│
├─ Person B
│  ├─ Task: Method 1 clustering + validation
│  ├─ Effort: 3.5 days
│  ├─ Dependency: method1_features.parquet (from Person A)
│  └─ Deliverable: method1_clusters.pkl + report
│
└─ Shared
   ├─ Code review (30 min)
   └─ Integration test (1 day)
```

### 4.3 Dependency Management

**Critical Path Visualization:**
```
Data Ingestion (A) [2 days]
        ↓
Feature Eng (A) [2 days]
        ↓
      ┌─────────────────────────────────┐
      ↓                                 ↓
Method1 (B) [3 days]        Method2 (A) [3 days]
      ├─ Clustering                    ├─ Clustering
      ├─ Validation                    ├─ Validation
      └─ Semantics                     └─ Semantics
      ↓                                 ↓
      └─────────────┬───────────────────┘
                    ↓
            Modeling (both) [4 days]
                    ↓
            Evaluation (B) [2 days]
                    ↓
            Reporting (both) [3 days]
```

**Blocking Scenarios & Backup Plans:**
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Person A: Spark setup delays | High | Start with local single-machine Spark, scale later |
| Data corrupted | High | Validate schema in Day 1, have backup dataset |
| Method 1 underperforms | Medium | Have Method 2 + Baseline ready as alternative |
| Model training slow | Medium | Use distributed GPU training if available |

---

## 5. Testing & Validation

### 5.1 Unit Testing Responsibility

| Component | Owner | Test Coverage |
|-----------|-------|----------------|
| Data validation (validators.py) | Person B | 90%+ |
| Feature extraction (all methods) | Person A | 80%+ |
| Clustering logic | Person B | 85%+ |
| Model layers (DGCN, TCN, ZIP) | Person A | 90%+ |
| Metrics computation | Person B | 95%+ |

### 5.2 Integration Testing (Joint Responsibility)

```bash
# Weekly integration test
pytest tests/integration/ -v

# Checklist:
- [ ] Data ingestion → Feature extraction pipeline works
- [ ] All 4 methods produce output in expected format
- [ ] Model can train on generated features
- [ ] Evaluation metrics compute without errors
```

### 5.3 Code Quality Gates

**Before pushing to develop:**
```bash
make lint    # PEP 8 compliance
make test    # All tests pass
make build   # No build errors
```

**CI/CD Pipeline (GitHub Actions):**
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run linting
        run: make lint
      - name: Run tests
        run: make test
      - name: Build
        run: make build
```

---

## 6. Documentation Responsibilities

| Document | Owner | Update Frequency |
|----------|-------|------------------|
| README.md | Both | Per week |
| DATA_PIPELINE_SPECIFICATION.md | Person A | As features added |
| MODULE_*.md | Both | As implemented |
| EXPERIMENT_TRACKING.md | Person B | Per experiment |
| Code docstrings | Each feature owner | At merge time |
| Project status (Notion) | Both | Every sync |

---

## 7. Conflict Resolution

### Issue Resolution Process

**Level 1: Direct Discussion (5 min)**
- Talk to other team member
- Explain perspective

**Level 2: Async Decision Making (1 hour)**
- Document decision on Slack with reasoning
- Other team member can comment

**Level 3: Sync Call (if needed)**
- Quick 15-min video call to resolve
- Document decision in Slack

**Level 4: Escalation (rare)**
- Involve advisor/supervisor
- Follow their guidance

### Example Conflict Scenarios

**Scenario 1: Disagreement on feature design**
```
Person A: "We should aggregate at zone level for simplicity"
Person B: "We need sub-zone clusters to capture heterogeneity"

→ Resolution: Implement both, compare performance, choose winner
```

**Scenario 2: Time pressure, incomplete testing**
```
Person A: "We're behind schedule, need to ship Method 3 untested"
Person B: "Risk: untested code breaks evaluation"

→ Resolution: Implement core logic + basic unit tests,
   full integration tests in parallel
```

---

## 8. Knowledge Handoff & Documentation

### 8.1 If Someone Gets Sick/Unavailable

**Current State Documentation (Slack pin):**
```
Week X Status (auto-updated every Monday):
- Person A currently working on: [task name]
- Person A blockers: [list]
- Person B currently working on: [task name]  
- Person B blockers: [list]
- Next critical deadline: [date]
```

**Code Documentation:**
- Every function has docstring + type hints
- Complex algorithms have comments explaining logic
- README has "Architecture Overview" section

### 8.2 Onboarding New Team Member (if needed)

1. Read README.md + PROJECT_ROADMAP.md
2. Run `make setup` to get environment ready
3. Read DATA_PIPELINE_SPECIFICATION.md
4. Review last 3 PRs to understand current state
5. 30-min sync with existing team to get oriented

---

## 9. Final Submission Handoff

### 8 weeks before deadline:

**Week 7.5 (Merge all code to main)**
- All features merged to `main` branch
- All tests passing
- Documentation complete

**Week 8 (Final Report)**
- Person A: Big Data module report (20-30 pages)
- Person B: ML/DL module report (20-30 pages)
- Both: Presentation slides (15-20 min)
- Both: Code repository cleanup & final documentation

**Submission Checklist:**
- [ ] Code reproducible (clear setup instructions)
- [ ] All results documented (MLflow runs saved)
- [ ] Figures publication-ready (high resolution)
- [ ] Thesis reports complete
- [ ] Presentation rehearsed
- [ ] No merge conflicts
- [ ] All secrets removed from repo

---

## 10. Team Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Code review turnaround | < 24 hours | GitHub PR timestamps |
| Meeting attendance | 100% | Weekly sync attendance |
| Test coverage | > 80% | Coverage reports |
| Blocker resolution | < 4 hours | Slack response times |
| Deliverable on-time | 100% | vs. PROJECT_ROADMAP |
| Code quality (lint) | 0 errors | `make lint` output |

---

## Quick Reference

**Emergency Contacts:**
- Person A: [phone/email]
- Person B: [phone/email]

**Important Dates:**
- Week 1-2: Setup phase
- Week 3-5: Implementation phase
- Week 6: Modeling phase
- Week 7: Evaluation phase
- Week 8: Reporting phase
- **Final Deadline:** [date]

**Tools Used:**
- GitHub: Version control
- Slack/Teams: Communication
- Notion/Google Docs: Documentation
- MLflow: Experiment tracking
- Local/Cloud: Compute

**Success is when we deliver:**
✅ 4 clustering methods implemented & compared
✅ Reproducible results with clear documentation
✅ Two independent but coherent thesis modules
✅ Publication-quality visualizations
✅ On time & under budget (time-wise!)
