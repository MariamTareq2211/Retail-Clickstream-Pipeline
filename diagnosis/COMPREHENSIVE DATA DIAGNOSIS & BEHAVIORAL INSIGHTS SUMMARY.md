================================================================================
            COMPREHENSIVE DATA DIAGNOSIS & BEHAVIORAL INSIGHTS SUMMARY         
================================================================================

1. CATALOG & PRODUCTS DIAGNOSIS (dim_product):
   • Catalog Size        : 117,286 valid unique products (Placeholder row `product_id = -1` removed).
   • Data Completeness   : Zero missing values (Nulls); missing details are represented as categorical 'Unknown'.
   • Category Hierarchy  : 14 Level-1 categories, 57 Level-2 categories, and 82 Level-3 categories.
   • Brand Diversity     : 2,641 unique brands providing rich metadata diversity for TF-IDF.
   • Critical Text Insight (Unknown Trap & Namespacing):
     - 'Unknown' is heavily prevalent: L1/L2 missing in 53.4%, L3 missing in 74.2%, and Brand missing in 25.5%.
     - 17.1% of products have all fields as 'Unknown' (zero text signal).
     - Engineering Action: Implemented strict Namespacing and explicit filtering in `metadata_text` construction to prevent generic 'Unknown' tokens from acting as false similarity bridges between unrelated products. Zero-text products safely fallback to Zero Vectors relying on price and user interactions.

2. INTERACTIONS & BEHAVIORAL DIAGNOSIS (fact_events):
   • Interaction Volume  : 8,248,730 total raw interaction records across 940,426 unique users.
   • User Engagement     : Median of 4 interactions per user.
     ↳ Behavioral Insight: Shows high user engagement; having 4+ interactions for over half the user base 
                           allows us to construct rich, dense User Preference Vectors without heavy Cold-Start.
   • Event Funnel        : 94.97% Page Views (Clicks), 2.84% Cart Additions, and 2.19% Purchases.
     ↳ Behavioral Insight: Users demonstrate implicit interest hierarchy; Carting and Purchasing signify 
                           much stronger intent than casual clicks, requiring weighted interaction scoring.
   • Catalog Coverage    : 98.51% (115,543 out of 117,286 products have active user interactions).
     ↳ Behavioral Insight: Almost the entire catalog is active, meaning cold products are extremely rare (<1.5%), 
                           making Content-Based Filtering highly effective across the entire system.
   • Temporal Tracking   : Verified `event_ts` column in Gold schema, successfully unblocking Temporal Splitting and Time-Decayed model engineering.

3. DATA ANOMALIES & DEDUPLICATION (Pre-Split Aggregation Strategy):
   • Duplicates & Timestamps : With `event_ts` active, exact duplicate records drop to negligible levels (611 rows), proving that previous raw duplicate volume was largely un-timestamped click/session noise.
   • Action Plan (Applied): 
     - GroupBy aggregation retained with an exposed `count_transform` parameter (raw / log1p / capped) optimized dynamically via Optuna in Module 2.
     - Raw Events Count     : 8,248,730
     - Cleaned Events Count : 5,056,471 (with `interaction_count` retained for flexibility).

4. PRICE FEATURE ANALYSIS & DOMAIN INSIGHTS (base_price):
   • Distribution Skew   : Highly right-skewed distribution (Skewness = 3.64).
   • Outliers (IQR)      : 12,080 products (10.30%) exceed the upper IQR bound ($442.43).
     ↳ Behavioral Insight: High price variance is completely natural in E-Commerce domains (e.g., cheap accessories 
                           vs. high-end electronics) and represents true business catalog structure, not data errors.
   • Engineering Strategy (Post-Train/Test Split):
     - Apply Log Transformation `np.log1p(base_price)` to compress price scale and neutralize skewness.
     - Apply `RobustScaler` (with `with_centering=False`) on transformed prices using Train Set parameters to preserve high-value items without leakage and prevent zero-vector collapse for median-priced items.

================================================================================