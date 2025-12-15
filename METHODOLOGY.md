# Methodological Breakdown: Founder Success Prediction GNN

## Overview

This system uses **Heterogeneous Graph Neural Networks (HeteroGNN)** to predict startup founder success by learning from the network structure of founders' educational and professional backgrounds.

---

## Pipeline Architecture

The system follows a 5-stage pipeline:

```
1. Data Loading & Preprocessing
   ↓
2. Graph Construction
   ↓
3. GNN Training (Multi-seed Ensemble)
   ↓
4. Clustering (Founder Archetypes)
   ↓
5. Evaluation (Precision@K)
```

---

## Stage 1: Data Loading & Preprocessing

**Location**: `src/data/preprocessing.py`, `src/data/loader.py`

### Input Data Format
- **CSV files** with founder records
- **JSON columns**: `educations_json`, `jobs_json`
- **Binary label**: `success` (1 = successful founder, 0 = not successful)

### Feature Extraction

#### Education Features (13 features)
Extracted from `educations_json`:
- **Degree metrics**: Number of degrees, highest degree score (1-4)
- **University prestige**: Best QS ranking, top-10/50/100 flags
- **Degree types**: PhD, MBA, Masters flags
- **Field of study**: STEM, Business, combined flags

**Example**:
```json
[{
  "degree": "PhD",
  "field": "Computer Science",
  "qs_ranking": 5
}]
```
→ Produces: `edu_has_phd=1`, `edu_is_stem=1`, `edu_is_top10_school=1`

#### Job Features (17 features)
Extracted from `jobs_json`:
- **Career counts**: Total jobs, senior roles, CXO roles
- **Role types**: Technical, product, business roles
- **Company size**: Big company experience, startup experience
- **Experience**: Total years, number of industries
- **Seniority**: Repeat founder, technical senior flags

**Example**:
```json
[{
  "role": "CTO",
  "company_size": "1001-5000",
  "industry": "Technology",
  "duration": "4-5 years"
}]
```
→ Produces: `job_num_cxo_roles=1`, `job_has_big_company_exp=1`, `job_total_experience_years=4.5`

#### Feature Redundancy Removal
- Computes **Jaccard similarity** between all feature pairs
- Removes features with >60% similarity to higher-lift features
- Reduces ~30 features → ~20-25 features (typical)

**Algorithm**:
1. Compute lift (precision/base_rate) for each feature
2. Sort features by lift (descending)
3. Keep feature only if Jaccard similarity < 0.6 with all kept features

---

## Stage 2: Graph Construction

**Location**: `src/graph/builder.py`, `src/graph/heterodata.py`

### Heterogeneous Graph Structure

#### Node Types (5 types)

1. **Founder nodes** (n = number of founders, e.g., 2000)
   - Features: 20-25 baseline features (education + job)
   - Labels: Binary success labels (for train/val only)

2. **University nodes** (n = 5, fixed)
   - `uni_top10` (QS rank 1-10)
   - `uni_top50` (QS rank 11-50)
   - `uni_top100` (QS rank 51-100)
   - `uni_other` (QS rank 100+)
   - `uni_unknown` (no ranking)
   - Features: One-hot encoded (5-dim)

3. **Company size nodes** (n = variable, e.g., 8)
   - `company_1_10`, `company_11_50`, `company_51_200`, etc.
   - Features: One-hot encoded

4. **Industry nodes** (n = variable, e.g., 15)
   - `industry_Technology`, `industry_Finance`, etc.
   - Features: One-hot encoded

5. **Role type nodes** (n = 8, fixed)
   - `role_cxo`, `role_founder`, `role_vp`, `role_director`
   - `role_engineer`, `role_product`, `role_business`, `role_other`
   - Features: One-hot encoded (8-dim)

#### Edge Types (4 types + 4 reverse)

**Forward edges** (founder → attribute):
1. `(founder, studied_at, university)` - Education connections
2. `(founder, worked_at, company_size)` - Job company size
3. `(founder, in, industry)` - Industry connections
4. `(founder, had, role_type)` - Job role types

**Reverse edges** (attribute → founder):
- Created automatically for bidirectional message passing
- `(university, rev_studied_at, founder)`
- `(company_size, rev_worked_at, founder)`
- `(industry, rev_in, founder)`
- `(role_type, rev_had, founder)`

**Example founder**:
```
Alice (founder) →
  - studied_at → uni_top10 (2 edges for 2 degrees)
  - worked_at → company_1001_5000, company_51_200
  - in → industry_Technology
  - had → role_cxo, role_engineer (3 edges for 3 jobs)
```

### Graph Statistics
Typical graph:
- **Nodes**: 2,000 founders + 5 universities + 8 company sizes + 15 industries + 8 roles = **2,036 nodes**
- **Edges**: ~10,000 founder→attribute edges + ~10,000 reverse edges = **~20,000 edges**

---

## Stage 3: GNN Training

**Location**: `src/models/gnn.py`, `src/models/trainer.py`, `src/models/losses.py`

### Model Architecture: FounderGNNAdvanced

#### Layer-by-layer breakdown:

**Step 1: Node Projection** (`self.projections`)
- Projects each node type to 256-dimensional space
- Architecture per node type:
  ```
  Input (dim varies by node type)
    → Linear(input_dim → 256)
    → LayerNorm(256)
    → ReLU
    → Dropout(0.4)
    → Output: 256-dim embeddings
  ```

**Step 2: Message Passing Layers** (6 layers)
- Uses **SAGEConv** (GraphSAGE convolution) per edge type
- Each layer:
  ```
  For each edge type:
    1. Aggregate neighbor messages (mean aggregation)
    2. Combine with self features
    3. Apply linear transformation

  Update:
    h_i^(l+1) = σ(W · CONCAT(h_i^(l), AGG({h_j^(l) : j ∈ N(i)})))
  ```

- After convolution:
  ```
  → GELU activation
  → Dropout(0.4)
  → Residual connection: h^(l+1) = h^(l+1) + h^(l) (after layer 1)
  ```

**Step 3: Classification Head** (`self.classifier`)
- Deep 3-layer classifier for founder nodes:
  ```
  Founder embedding (256-dim)
    → Linear(256 → 256)
    → LayerNorm(256)
    → ReLU
    → Dropout(0.4)
    → Linear(256 → 128)
    → LayerNorm(128)
    → ReLU
    → Dropout(0.4)
    → Linear(128 → 1)
    → Sigmoid
    → Success probability [0, 1]
  ```

### Message Passing Example

**Iteration 0**: Initial embeddings
```
Alice: [baseline features projected to 256-dim]
uni_top10: [one-hot → 256-dim]
role_cxo: [one-hot → 256-dim]
```

**Iteration 1**: First hop
```
Alice receives messages from:
  - uni_top10 (she studied there)
  - role_cxo (she had that role)
  - industry_Technology (her industry)

Alice's embedding ← Alice's features + messages from neighbors
```

**Iteration 2-6**: Multi-hop propagation
```
Alice now receives indirect information:
  - From other founders who went to uni_top10
  - From other founders who were role_cxo
  - From shared industries and company sizes

After 6 layers: Alice's embedding captures 6-hop neighborhood context
```

### Training Configuration

**Loss Function**: Precision-Weighted Binary Cross-Entropy
```python
# False positives cost 10x more than false negatives
weight_pos = 1.0
weight_neg = precision_weight = 10.0

loss = -[weight_pos * y * log(p) + weight_neg * (1-y) * log(1-p)]
```

**Rationale**: In VC, saying "yes" to a bad founder (FP) costs more than missing a good founder (FN).

**Optimizer**: AdamW
- Learning rate: 0.0005
- Weight decay: 0.001
- Gradient clipping: max_norm = 1.0

**Learning Rate Schedule**:
1. **Warmup** (epochs 0-100): Linear increase from 0 → 0.0005
2. **Cosine decay** (epochs 100-2000): Cosine annealing to ~1e-7

```python
lr(epoch) = {
  epoch < 100: 0.0005 * (epoch / 100)
  epoch ≥ 100: 1e-7 + 0.5 * (0.0005 - 1e-7) * (1 + cos(π * progress))
}
```

**Data Split**:
- Train: 80% (labels visible, used for gradients)
- Validation: 10% (labels visible, used for model selection)
- Test: 10% (labels hidden until final evaluation)

**Early Stopping**:
- Tracks validation P@100 every 50 epochs
- Saves best model checkpoint
- Loads best model at end

### Multi-Seed Ensemble

**Process** (`train_multiseed_ensemble`):
1. Train 10 models with different random seeds (0, 42, 84, ..., 378)
2. Each seed produces:
   - Model weights
   - Founder embeddings (256-dim)
   - Predictions (probabilities)

3. **Ensemble averaging**:
   ```python
   ensemble_predictions = mean([model_0(data), ..., model_9(data)])
   ensemble_embeddings = mean([emb_0, ..., emb_9])
   ```

**Why ensemble?**
- Reduces variance from random initialization
- Typical improvement: +1-2% P@100 vs single model

**Best model selection**:
- Evaluate each of 10 models on validation P@100
- Select highest-performing model as "best_model.pt"

---

## Stage 4: Clustering

**Location**: `src/clustering/kmeans.py`

### Purpose
Identify **founder archetypes** by combining:
- GNN embeddings (network context)
- Baseline features (resume attributes)

### Process

**Input data** (train + val only, 90% of data):
- GNN embeddings: 256-dim vectors
- Baseline features: 20-25 features

**Step 1: Normalization**
```python
# Standardize each feature to mean=0, std=1
embeddings_scaled = (embeddings - mean) / std
baseline_scaled = (baseline - mean) / std
combined = concat(embeddings_scaled, baseline_scaled)
```

**Step 2: K-Means Clustering**
```python
kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
clusters = kmeans.fit_predict(combined)  # [0, 1, 2, 3]
```

**Step 3: Cluster Analysis**
For each cluster:
1. **Success rate**: % of founders who succeeded
2. **Size**: Number of founders in cluster
3. **GNN predictions**: Mean predicted probability
4. **Top features**: Highest prevalence features

**Example output**:
```
Cluster 0: "Elite Technical Founders" (n=120, success=45%)
  - Top features: edu_is_top10_school, job_is_technical_senior, job_has_cxo_experience
  - Mean GNN score: 0.52

Cluster 1: "Serial Entrepreneurs" (n=150, success=38%)
  - Top features: job_is_repeat_founder, job_has_startup_exp, edu_has_mba
  - Mean GNN score: 0.48

Cluster 2: "Corporate Executives" (n=180, success=25%)
  - Top features: job_has_big_company_exp, job_num_senior_roles, edu_is_business
  - Mean GNN score: 0.35

Cluster 3: "Emerging Founders" (n=350, success=12%)
  - Top features: job_num_prior_jobs (low), edu_has_advanced_degree
  - Mean GNN score: 0.22
```

**Optimal K determination** (if `--auto-clusters`):
- Try K = 2 to 10
- Compute silhouette score for each K
- Select K with highest silhouette score
- Typical optimal: K = 4-6

---

## Stage 5: Evaluation

**Location**: `src/evaluation/metrics.py`, `src/evaluation/inference.py`

### Ranking-Based Evaluation (No Thresholds!)

**Key insight**: Instead of "is this founder good?" → "rank all founders by success probability"

**Primary metric: Precision @ K (P@K)**

Definition:
```
P@K = (# successful founders in top K predictions) / K
```

**Example**:
```
Top 100 predictions:
  - 29 were successful founders
  - P@100 = 29/100 = 29%

Baseline (random):
  - Base rate = 8% successful founders overall
  - P@100 = 8% (expected)

Lift = 29% / 8% = 3.6x better than random
```

**Why P@K?**
- Matches real VC workflow: "Show me your top 100 picks"
- No arbitrary threshold tuning
- Directly optimizes for business value

### Test Set Evaluation

**Step 1: Assign test founders to clusters**
```python
# Use trained scalers and kmeans from training
test_embeddings_scaled = scaler_emb.transform(test_embeddings)
test_baseline_scaled = scaler_base.transform(test_baseline)
test_combined = concat(test_embeddings_scaled, test_baseline_scaled)
test_clusters = kmeans.predict(test_combined)
```

**Step 2: Cluster-aware ranking**
Two strategies:
1. **GNN-only**: Rank by GNN predictions
2. **Cluster-boosted**: Boost predictions for founders in high-success clusters

**Step 3: Compute metrics**
- P@10, P@20, P@50, P@100, P@200
- Lift vs baseline
- Cluster distribution of successful founders

### Benchmark Comparisons

| Method | P@100 | Description |
|--------|-------|-------------|
| Random | 8% | Base rate |
| Tier-1 VCs | 5.6% | Human investors (published) |
| GPTree | 7.8% | LLM decision tree |
| RRF | 13.1% | Random Rule Forest |
| **This GNN** | **~29%** | Heterogeneous GNN + Clustering |

**Improvement**: 3.6x better than random, 2.2x better than RRF

---

## Key Technical Details

### 1. Transductive Learning

**IMPORTANT**: This is a **transductive** system, not inductive.

**What this means**:
- During training, the model sees the **entire graph** including test nodes
- Test node labels are hidden, but their features and connections are visible
- The GNN learns embeddings for ALL nodes simultaneously

**Implications**:
- ✅ **Pro**: Model uses network context (test founders connected to train founders)
- ❌ **Con**: Cannot directly predict on completely new founders not in the graph

**For new founders**: See "Using the Model" section in updated README

### 2. Heterogeneous Graph Benefits

**Why heterogeneous vs homogeneous?**

Homogeneous graph (baseline):
```
All nodes are founders, edges = "similar" relationships
→ Loses type information
```

Heterogeneous graph (this system):
```
Different node types with typed relationships
→ "Alice studied_at Oxford" vs "Alice worked_at Google"
→ Different message functions per edge type
→ Learns that university connections ≠ job connections
```

**Impact**: ~5% improvement in P@100 vs homogeneous baseline

### 3. Why 6 Layers?

- 1-2 layers: Only direct connections (1-2 hops)
- 3-4 layers: 3-4 hops, captures some indirect patterns
- **6 layers**: Optimal depth for this graph
  - Captures "founders who went to same school as successful founders"
  - Captures "founders with similar career paths"
- 8+ layers: Over-smoothing (all embeddings become similar)

**Tested depths**: 2, 4, 6, 8 → 6 layers performed best

### 4. Residual Connections

```python
for layer in layers:
    h_new = GNN_layer(h_old)
    h_old = h_new + h_old  # Residual connection
```

**Why?**
- Prevents vanishing gradients in deep networks
- Allows model to learn "corrections" rather than full transformations
- Enables training 6+ layer networks

---

## Hyperparameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden dim | 256 | Sweet spot for graph size (2000 nodes) |
| Num layers | 6 | Captures 6-hop neighborhood |
| Dropout | 0.4 | Prevents overfitting on small dataset |
| Learning rate | 0.0005 | Conservative for stable training |
| Epochs | 2000 | Ensures convergence |
| Precision weight | 10.0 | FP costs 10x more than FN |
| N seeds | 10 | Diminishing returns after 10 |
| N clusters | 4 | Data-driven optimal K |

---

## Performance Characteristics

**Training time** (on GPU):
- Single model: ~15-20 minutes (2000 epochs)
- 10-seed ensemble: ~3 hours total

**Inference time**:
- Batch prediction on 200 founders: <1 second
- Single founder: ~10ms

**Memory usage**:
- Model size: ~5 MB (256-dim embeddings)
- Graph in memory: ~50 MB (2000 nodes, 20k edges)

---

## Limitations and Future Work

### Current Limitations

1. **Transductive only**: Cannot directly predict on new founders without retraining
2. **Fixed graph structure**: Adding new nodes requires graph reconstruction
3. **No temporal modeling**: Doesn't capture career trajectory timing
4. **Binary labels only**: Doesn't model degree of success (exit size, growth rate)

### Potential Improvements

1. **Inductive GNN**: Use GraphSAINT or subgraph sampling for inductive learning
2. **Temporal GNN**: Model career progression over time
3. **Attention mechanisms**: Use GAT instead of SAGE to learn edge importance
4. **Multi-task learning**: Predict success type (acquisition, IPO, unicorn)
5. **Feature engineering**: Add network centrality, PageRank, community detection
6. **Active learning**: Query most uncertain predictions for labeling

---

## File Reference Map

| Component | Files |
|-----------|-------|
| Data loading | `src/data/loader.py` |
| Feature extraction | `src/data/preprocessing.py` |
| Train/val/test split | `src/data/splitting.py` |
| Graph construction | `src/graph/builder.py`, `src/graph/heterodata.py` |
| GNN model | `src/models/gnn.py` |
| Training loop | `src/models/trainer.py` |
| Loss functions | `src/models/losses.py` |
| Clustering | `src/clustering/kmeans.py` |
| Evaluation | `src/evaluation/metrics.py`, `src/evaluation/inference.py` |
| Visualization | `src/visualization/plots.py` |
| Main pipeline | `main.py` |
| Configuration | `config.py` |
