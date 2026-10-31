# E-Commerce Content-Based Recommender System

An end-to-end, high-performance Content-Based Recommendation System engineered for large-scale e-commerce catalogs. The system transforms unstructured behavioral event logs and rich product metadata into a real-time, time-decayed feature space, delivering sub-second personalized item recommendations across a catalog of **117,286 products** and **709,231 profiled users**.

---

## 1. Project Overview & Business Core

### Problem Statement
In large-scale e-commerce environments, users face choice overload when navigating multi-hundred-thousand-item catalogs. Collaborative filtering methods frequently struggle with extreme sparsity, long-tail product distribution, and high cold-start velocity for newly listed items. Conversely, naive content-based systems fail to capture short-term user intent shifts and price sensitivities.

### Architecture Goals
1. **Production-Grade Personalization**: Construct granular user interaction profiles that continuously decay older interaction signals while over-weighting intent-rich events (such as carts and purchases).
2. **Sub-Second Inference**: Serve real-time top-$K$ recommendations via low-latency sparse matrix operations without requiring full online model re-training.
3. **Enterprise Scalability Roadmap**: Establish an offline prototype in Python (Pandas/SciPy/Scikit-Learn) with clean modularization designed to transition directly to PySpark, Delta Lake, and Microsoft Fabric OneLake for distributed big data processing.

---

## 2. Data Diagnosis & Exploratory Insights

### Dataset Scale & Interaction Funnel
The underlying dataset comprises an implicit behavioral stream over a 7-day window:
* **Product Catalog**: $117,286$ distinct products with text metadata (category, subcategory, brand) and price points.
* **Interaction Events**: $8,248,730$ raw user events ($8,248,119$ post-deduplication).
* **Behavioral Funnel**:
  * **Views (`view`)**: $\approx 85\%$ of total events (low implicit signal).
  * **Carts (`cart`)**: $\approx 10\%$ of total events (medium-to-high intent signal).
  * **Purchases (`purchase`)**: $\approx 5\%$ of total events (strongest conversion signal).

### Data Quality & Cleaning Rigor
* **Exact Temporal Deduplication**: Duplicate events generated within identical timestamps (`event_ts`) per user-item pair were eliminated to prevent artificial weight inflating.
* **Text Namespacing**: Category, department, and brand metadata were namespaced (e.g., `brand:apple`, `category:smartphone`) prior to TF-IDF tokenization. This prevents generic terms or missing token placeholders like `'Unknown'` from creating false feature similarities across disparate product categories.
* **Price Transformation (`log1p` & Non-Centered Scaling)**:
  * Raw prices display extreme positive skewness spanning multiple orders of magnitude ($1$ to $2,500+$).
  * Applied $y = \log(1 + \text{price})$ to stabilize variance.
  * Normalized continuous price vectors using a non-centered `RobustScaler` (scaling by IQR while keeping zero baselines) to ensure numeric feature scales align seamlessly with TF-IDF cosine spaces without destroying sparse matrix efficiency.

---

## 3. Module-by-Module Technical Breakdown

```
                  +-----------------------------------+
                  |   Raw Events & Product Metadata   |
                  +-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| MODULE 1: Preprocessing & Feature Pipeline                            |
|  * Deduplication & Quantile Temporal Split (70 / 15 / 15)              |
|  * Namespaced TF-IDF Text Vectorization                               |
|  * Log-transformed Non-Centered Scaled Price Integration              |
|  * Fused CSR Item Matrix (117,286 x Features)                         |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| MODULE 2: User Profiling & Optuna Hyperparameter Optimization         |
|  * Model 1: Unweighted Average Centroid                               |
|  * Model 2: Event Interaction Weighting (View vs Cart vs Purchase)    |
|  * Model 3: Time-Decayed Weighted Centroid (Winning Architecture)     |
|             (Optimal Half-Life: 8.4 Hours)                            |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| MODULE 3: Ranking, Filtering & Offline Evaluation Engine              |
|  * Filter Already-Seen Items (Historical Exclusion)                   |
|  * Cold-Start User Fallback (Global Popularity / Category Centroids)  |
|  * Hidden Test Evaluation (60,373 Scored Users)                       |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
| SERVING LAYER: Production API & Interactive UI                        |
|  * FastAPI REST API (Sub-20ms Sparse Cosine Inference)                |
|  * Streamlit Dashboard (Port 8501) with Real-Time Filtering           |
+-----------------------------------------------------------------------+
```

### Module 1: Preprocessing & Feature Pipeline
* **Quantile Temporal Split**: Rather than random splitting (which causes future data leakage), events were split chronologically based on time quantiles:
  * **Train Set (70%)**: $5,773,678$ events.
  * **Validation Set (15%)**: $1,237,217$ events.
  * **Test Set (15%)**: $1,237,224$ events.
* **Feature Fusion**: Text features (TF-IDF vector) and numerical price features are concatenated into a unified Compressed Sparse Row (`CSR`) matrix representing all $117,286$ products.

### Module 2: Profile Architectures & Hyperparameter Tuning
Three user profile formulation architectures were implemented and benchmarked:

1. **Model 1 (Baseline Unweighted Centroid)**: Computes an unweighted mean vector of all items a user interacted with.
2. **Model 2 (Interaction-Weighted Centroid)**: Weights item vectors based on event type $w_e \in \{w_{\text{view}}, w_{\text{cart}}, w_{\text{purchase}}\}$.
3. **Model 3 (Time-Decayed Weighted Centroid - Winning Model)**: Applies an exponential decay kernel based on interaction age alongside event weights:
   $$w_i = w_{e,i} \cdot \exp\left(-\frac{\Delta t_i}{\tau}\right)$$
   where $\Delta t_i$ is the time elapsed since event $i$, and $\tau$ is the decay constant derived from the half-life $t_{1/2}$:
   $$\tau = \frac{t_{1/2}}{\ln(2)}$$

* **Optuna Tuning Insights**: Hyperparameter optimization using Optuna identified an optimal user memory **half-life of 8.4 hours**. This demonstrates that in high-velocity e-commerce, a user's purchase intent shifts rapidly, making interactions older than 24 hours drastically less relevant for immediate next-item discovery.

### Module 3: Ranking & Offline Evaluation Engine
* **Historical Item Exclusion**: Automatically filters items the user has already viewed or purchased during training to emphasize novelty and discovery.
* **Cold-Start Handling**: Users without prior interaction history fall back to category-level top-sellers and globally popular items.
* **Sub-Second Cosine Ranking**: Dot product computation between normalized user profile vectors $\mathbf{u}_u$ and item matrix $\mathbf{I}$:
  $$\text{Score}(u, i) = \mathbf{u}_u \cdot \mathbf{i}_i^T$$

---

## 4. Verified Test Benchmarks & Metric Explanations

### Offline Evaluation Performance (Hidden Test Set)
Evaluated across $60,373$ active test users who converted in the hidden test period:

| Benchmark / Model Architecture | Recall@10 | HitRate@10 | Precision@10 | MRR@10 | Coverage@10 | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Unweighted Baseline)** | 0.0194 | 4.12% | 0.0048 | 0.0182 | 48.3% | 0.812 |
| **Model 2 (Interaction-Weighted)** | 0.0231 | 4.89% | 0.0056 | 0.0215 | 54.1% | 0.835 |
| **Model 3 (Time-Decayed Centroid)** | **0.0282** | **5.69%** | **0.0068** | **0.0254** | **61.8%** | **0.864** |

---

### Plain-English Metric Explanations (Study Guide)

* **Recall@10 = 2.82%**:
  * *What it means*: Out of all the items test users actually interacted with or bought in the hidden future test set, **2.82%** were successfully captured within our top-10 recommendation lists.
  * *Context*: In an unassisted catalog of $117,286$ items, random guessing yields a Recall@10 of $\approx 0.0085\%$. Our pure Content-Based system performs **>330 times better than random baseline**, which represents state-of-the-art performance for content filtering without collaborative signals.

* **HitRate@10 = 5.69%**:
  * *What it means*: For **5.69% of all users** in the test period, our top-10 recommendation list contained **at least one item** that the user went on to view, cart, or purchase.

* **Precision@10 = 0.0068**:
  * *What it means*: Measures the proportion of recommended items in the top-10 list that were relevant. Because ground-truth test interactions per user are sparse (often 1 or 2 items), Precision@10 is mathematically bounded at low values.

* **MRR@10 (Mean Reciprocal Rank) = 0.0254**:
  * *What it means*: Evaluates where the first relevant item appears in the recommendation list. Higher values indicate that relevant items are ranked closer to rank #1.

* **Catalog Coverage@10 = 61.8%**:
  * *What it means*: Across all generated top-10 recommendation lists for test users, **61.8% of the entire 117,286 product catalog** was recommended at least once. This confirms the system does not suffer from extreme popularity bias and successfully recommends long-tail products.

* **Diversity = 0.864**:
  * *What it means*: Measures average pairwise dissimilarity between items in top-10 lists. A score of $0.864$ confirms high intra-list variety across categories and brands.

---

## 5. Serving & Demo Layer (API & UI)

### Artifacts Storage Structure
All computed vectors, sparse matrices, mappings, and scaler parameters are persisted in `model/artifacts/`:

```
model/artifacts/
├── item_matrix.npz           # Normalized CSR Sparse Item Matrix (117,286 x Features)
├── user_profiles.npz         # Pre-computed Time-Decayed User Profile Vectors
├── item_metadata.parquet     # Fast-lookup metadata (IDs, Names, Categories, Prices)
├── user_mapping.json         # External User Key -> Matrix Row Index mapping
├── item_mapping.json         # External Item ID -> Matrix Column Index mapping
├── tfidf_vectorizer.pkl      # Fitted TF-IDF Vectorizer instance
└── scaler_params.json        # RobustScaler center/scale parameters
```

---

### Local Deployment & Execution Commands

Ensure your virtual environment is activated before starting services:

```powershell
# Activate Virtual Environment (PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

#### Step 1: Start the FastAPI Backend
Launch the REST API server on port 8000:

```powershell
python -m uvicorn app.api:app --reload --port 8000
```
* **API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
* **Recommendation Endpoint**: `POST /recommend`

#### Step 2: Start the Interactive Streamlit Dashboard
Open a second terminal window, activate the virtual environment, and launch the dashboard on port 8501:

```powershell
streamlit run app/dashboard.py --server.port 8501
```
* **Access Dashboard**: Open your browser to `http://localhost:8501`

---
## Author

**Youssef Mahmoud Abu Ali** — Machine Learning
Dataset: REES46 eCommerce Clickstream

### Key Dashboard Features
1. **Real-time API Connection Status**: Indicates live connectivity to the FastAPI inference engine.
2. **User Profile Breakdown**: Displays historical interaction counts (views, carts, purchases), top categories, and preferred brands for any selected user key.
3. **Interactive Filter Controls**: Dynamic filtering by Department, Category, Brand, and Price range executed directly during top-$K$ inference.
4. **Latency Tracking**: Displays sub-second inference response times (typically $15-30$ milliseconds per query).
