# Project Title: Unsupervised Learning — Clustering & Dimensionality Reduction

## Objective
To meticulously diagnose, correct, and extend a K-Means clustering pipeline, applying robust statistical and machine learning methodologies to economic and synthetic behavioral datasets, thereby demonstrating proficiency in unsupervised learning and dimensionality reduction techniques.

## Methodology
*   **Pipeline Diagnosis & Correction:** Identified and rectified critical errors within an initial K-Means pipeline, including the omission of feature standardization, incorrect API parameter usage (`k` instead of `n_clusters`), improper ordering of PCA application (before scaling), and lack of reproducibility due to missing `random_state`.
*   **Corrected Pipeline Construction:** Built an end-to-end K-Means clustering pipeline incorporating `StandardScaler` for feature normalization, `KMeans` with appropriate parameters, and `PCA` for visualization of clustered data.
*   **Customer Segmentation Application:** Applied the refined clustering methodology to synthetic behavioral data to segment customers, simulating real-world fintech applications.
*   **Dimensionality Reduction Comparison:** Evaluated and contrasted the effectiveness of PCA and UMAP in reducing data dimensionality, particularly in preserving cluster separation for visualization.
*   **Reusable Module Development:** Developed a modular Python script (`clustering_utils.py`) encapsulating key clustering functions: `run_kmeans_pipeline` (for full pipeline execution), `evaluate_k_range` (for K-value selection via WCSS and silhouette scores), and `plot_pca_clusters` (for standardized visualization).

## Key Findings
*   For the module's internal self-test, `K=3` was identified as optimal based on the silhouette score, indicating strong cluster separation for the synthetic `make_blobs` data.
*   In the customer segmentation task, where the true number of segments was `K=4`, UMAP demonstrated superior visual separation and clearer cluster boundaries compared to PCA, especially for non-linear data structures, effectively highlighting its advantage in preserving local data structure.
