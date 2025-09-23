import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import random
import io
import requests

np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="InCURA", page_icon="data/Logo_incura.svg", layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("data/Logo_incura.svg", width=120)
with col2:
    st.title("Integrative Gene Clustering based on Transcription Factor Binding Sites")
    
    st.markdown(
    "InCURA enables clustering of differentially expressed genes (DEGs) based on shared transcription factor (TF) binding patterns. \n\n"
    "Paste a list of DEGs into the first text field. In the second text field you can either paste all expressed genes in your dataset to automatically "
    "filter for expressed TFs or paste a list of TFs of interest to explore regulatory modules, "
    "visualize gene clusters, and identify enriched TF binding sites.\n\n"
    "Best performance is reached in a range of 150 - 1500 DEGs. \n\n"
    "If you are unsure about the number of clusters please run the k-means performance metrics at the bottom of the page. \n\n"
    "**Note:** This implementation of InCURA uses a pre-computed TF binding site matrix "
    "with a fixed background model based on all protein coding genes in the respective organism. For more versatile functionality use the [GitHub version of InCURA](https://github.com/SinghalLab/incura)."
    )

# -------------------------------
# Dataset selection
# -------------------------------
dataset_choice = st.radio(
    "Select organism:",
    options=["Mouse", "Human"],
    index=0,
    horizontal=True
)

# -------------------------------
# Load Preprocessed TFBS Matrix
# -------------------------------
@st.cache_data
def load_preprocessed_matrix(species: str) -> pd.DataFrame:
    urls = {
        "Mouse": "https://zenodo.org/records/15866266/files/full_count_matrix_mouse.tsv?download=1",
        "Human": "https://zenodo.org/records/15866266/files/full_count_matrix_human.tsv?download=1"
    }
    url = urls.get(species)
    if not url:
        raise ValueError("Invalid species selection")

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download matrix. Status: {response.status_code}")

    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")), sep="\t", index_col=0)
    df.index = df.index.str.lower()  # lowercase gene names
    df.columns = df.columns.str.lower()  # lowercase TF names
    return df

with st.spinner("Loading TFBS matrix..."):
    matrix = load_preprocessed_matrix(dataset_choice)
    
# -------------------------------
# Example run button
# -------------------------------
# --- Keep example choice in session state ---
if "use_example" not in st.session_state:
    st.session_state.use_example = False

# Button sets the flag
if st.button("▶️ Run Example"):
    st.session_state.use_example = True

# Optional reset button
if st.session_state.use_example:
    if st.button("🔄 Reset to custom input"):
        st.session_state.use_example = False

# --- Use example or custom input ---
if st.session_state.use_example:
    with open("data/DEGs_ko.txt") as f:
        rows_text = f.read()
    with open("data/genes.txt") as f:
        cols_text = f.read()
else:
    # --- Gene Input ---
    st.subheader("Filtering for Differentially Expressed Genes")
    rows_text = st.text_area(
        "Paste gene names here (one gene per line):", 
        placeholder="DEG1\nDEG2\nDEG3"
    )

    # --- TF Input ---
    st.subheader("Filtering for Transcription Factors")
    cols_text = st.text_area(
        "Paste list of **all expressed genes** or **TFs of interest** here (one per line):", 
        placeholder="Gene1\nGene2\nGene3"
    )



# Standardize parsing of input
row_list_raw = [x.strip() for x in rows_text.replace(',', '\n').splitlines() if x.strip()]
row_list = [x.lower() for x in row_list_raw]
valid_rows = [r for r in row_list if r in matrix.index]

col_list_raw = [x.strip() for x in cols_text.replace(',', '\n').splitlines() if x.strip()]
col_list = [x.lower() for x in col_list_raw]
valid_cols = [c for c in col_list if c in matrix.columns]

st.markdown(f"**Genes pasted:** {len(row_list_raw)} &nbsp;&nbsp;|&nbsp;&nbsp; **Valid genes:** {len(valid_rows)}")
st.markdown(f"**TFs pasted:** {len(col_list_raw)} &nbsp;&nbsp;|&nbsp;&nbsp; **Valid TFs:** {len(valid_cols)}")


# ---------------------------------
# TF enrichment 
# ---------------------------------
@st.cache_data
def tfbs_cluster_enrichment(binary_matrix, cluster_labels, pval_threshold=0.05):
    unique_clusters = np.unique(cluster_labels)
    enrichment_results = []

    for cluster_label in unique_clusters:
        cluster_genes = binary_matrix[cluster_labels == cluster_label]
        background_genes = binary_matrix[cluster_labels != cluster_label]

        cluster_size = len(cluster_genes)
        background_size = len(background_genes)

        for tfbs in binary_matrix.columns:
            cluster_tfbs_count = cluster_genes[tfbs].sum()
            background_tfbs_count = background_genes[tfbs].sum()

            contingency_table = [
                [cluster_tfbs_count, cluster_size - cluster_tfbs_count],
                [background_tfbs_count, background_size - background_tfbs_count],
            ]
            _, p_value = fisher_exact(contingency_table)

            enrichment_results.append({
                'TFBS': tfbs,
                'Cluster': cluster_label,
                'p_value': p_value
            })

    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df['corrected_pval'] = multipletests(enrichment_df['p_value'], method='fdr_bh')[1]
    significant_results = enrichment_df[enrichment_df['corrected_pval'] < pval_threshold]

    return significant_results

# --- Filter + Show ---
if valid_rows and valid_cols:
    count_matrix = matrix.loc[valid_rows, valid_cols]
    count_matrix = count_matrix.astype(np.float32)
    st.success(f"Filtered to matrix with shape: {count_matrix.shape}")


    if count_matrix.empty:
        st.warning("No matching rows found after filtering. Check gene and TF names.")
    else:

        # --- Download filtered count matrix ---
        csv_matrix = count_matrix.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download Matrix of Regulatory Profiles",
            data=csv_matrix,
            file_name="incura_matrix.csv",
            mime="text/csv"
        )

        # --- UMAP ---
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(count_matrix.values)

        # --- KMeans ---
        n_clusters = st.slider("Number of clusters (KMeans)", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(count_matrix.values)

        # --- HeatMAP ---
        binary_matrix = count_matrix.astype(bool).astype(int)
        
        # Create a DataFrame for easier grouping
        df_binary = binary_matrix.copy()
        df_binary['cluster'] = cluster_labels
        
        # Compute cluster centroids (average TFBS presence per cluster)
        centroids = df_binary.groupby('cluster').mean()  # <--- removed .drop(columns='cluster')

        # Compute variance of each TF across clusters
        tf_variance = centroids.var(axis=0)
        
        # Select top 20 most variable TFs
        top_tfs = tf_variance.sort_values(ascending=False).head(20).index
        
        # Subset centroids to top TFs
        centroids_top = centroids[top_tfs]

            # Capitalise gene/TF names if desired
        if dataset_choice == "Mouse":
            centroids_top["TFBS"] = centroids_top["TFBS"].str.capitalize()
        elif dataset_choice == "Human":
            centroids_top["TFBS"] = centroids_top["TFBS"].str.upper()

        
        # --- Plot side by side ---
        # Create two columns: UMAP (left) and Heatmap (right)
        col1, col2 = st.columns([1, 1])  # equal width, you can adjust
        
        with col1:
            st.subheader("UMAP Projection with Clusters")
            fig_umap, ax = plt.subplots(figsize=(5, 4))  # smaller figure
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=cluster_labels, cmap='tab10', s=40  # smaller point size
            )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
            ax.legend(handles, [f"Cluster {i}" for i in range(n_clusters)], title="Clusters")
            st.pyplot(fig_umap)
        
        with col2:
            st.subheader("Cluster TFBS Patterns (Top 20 TFs)")
            fig_heat, ax = plt.subplots(figsize=(5, 4))  # smaller figure
            sns.heatmap(centroids_top.T, cmap="viridis", annot=False, ax=ax)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("TFBS")
            st.pyplot(fig_heat)

        # -------------------------------
        # Volcano plot with dynamic top TF labels
        # -------------------------------
        binary_matrix = count_matrix.astype(bool).astype(int)
        
        cluster_series = pd.Series(cluster_labels, index=binary_matrix.index, name='cluster')

        # -------------------------------
        # Side-by-side Volcano Plot & TFBS Enrichment
        # -------------------------------
        st.subheader("Cluster Comparison & Enrichment")
        
        # Use two columns
        col1, col2 = st.columns([1, 1])
        
        # --- Volcano plot in left column ---
        with col1:
            with st.expander("Volcano Plot: Compare Two Clusters"):
                selected_clusters = st.multiselect(
                    "Select two clusters", 
                    options=sorted(set(cluster_labels)),
                    default=sorted(set(cluster_labels))[:2]
                )
                if len(selected_clusters) == 2:
                    clust1, clust2 = selected_clusters
        
                    # Filter genes
                    genes_clust1 = binary_matrix[cluster_series == clust1]
                    genes_clust2 = binary_matrix[cluster_series == clust2]
        
                    # Frequency difference & p-values
                    freq1 = genes_clust1.mean()
                    freq2 = genes_clust2.mean()
                    diff = freq1 - freq2
        
                    from scipy.stats import fisher_exact
                    pvals = []
                    for tf in binary_matrix.columns:
                        table = [
                            [genes_clust1[tf].sum(), len(genes_clust1) - genes_clust1[tf].sum()],
                            [genes_clust2[tf].sum(), len(genes_clust2) - genes_clust2[tf].sum()]
                        ]
                        _, p = fisher_exact(table)
                        pvals.append(p)
        
                    volcano_df = pd.DataFrame({
                        'TF': binary_matrix.columns,
                        'diff': diff.values,
                        'pval': pvals
                    })
                    volcano_df['-log10(pval)'] = -np.log10(volcano_df['pval'])
                    volcano_df['significant'] = volcano_df['pval'] < 0.05
        
                    # Slider: number of top TFs
                    top_n = st.slider("Number of top TFs to label", min_value=5, max_value=50, value=20)
                    top_tfs = volcano_df.nsmallest(top_n, 'pval').copy()
        
                    # Species formatting
                    if dataset_choice == "Mouse":
                        top_tfs['TF_display'] = top_tfs['TF'].str.capitalize()
                    else:
                        top_tfs['TF_display'] = top_tfs['TF'].str.upper()
        
                    # Plot volcano
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6,5))
                    ax.scatter(
                        volcano_df['diff'], 
                        volcano_df['-log10(pval)'], 
                        c=volcano_df['significant'].map({True: 'red', False: 'gray'}),
                        alpha=0.7
                    )
        
                    for _, row in top_tfs.iterrows():
                        ax.text(row['diff'], row['-log10(pval)'], row['TF_display'], fontsize=9, ha='right', va='bottom')
        
                    ax.set_xlabel(f"TFBS Frequency Difference: Cluster {clust1}-{clust2}")
                    ax.set_ylabel("-log10(p-value)")
                    ax.set_title(f"Volcano Plot: Cluster {clust1} vs Cluster {clust2}")
                    st.pyplot(fig)
                else:
                    st.info("Select exactly two clusters to compare.")
        
        # --- TFBS enrichment heatmap in right column ---
        with col2:
            with st.expander("TFBS Enrichment Heatmap per Cluster"):

                st.subheader("Identify Enriched TFBS Driving the Clustering")
        
                binary_matrix = count_matrix.astype(bool).astype(int)  # ensure binary
                cluster_series = pd.Series(cluster_labels, index=count_matrix.index)
        
                enrichment_df = tfbs_cluster_enrichment(binary_matrix, cluster_series)
        
                if enrichment_df.empty:
                    st.warning("No significantly enriched TFBS found at the specified threshold.")
                else:
                    tfbs_counts = enrichment_df.groupby("TFBS")["Cluster"].nunique()
                    ubiquitous_tfbs = tfbs_counts[tfbs_counts > 1].index
                    enrichment_df = enrichment_df[~enrichment_df["TFBS"].isin(ubiquitous_tfbs)]
        
                    top_tfbs = (
                        enrichment_df.groupby("Cluster")
                        .apply(lambda x: x.nsmallest(10, "p_value"))
                        .reset_index(drop=True)                        )
                    enrichment_df = enrichment_df[enrichment_df["TFBS"].isin(top_tfbs["TFBS"])]
                        
                    # Format TF names in final enrichment_df before plotting
                    if dataset_choice == "Mouse":
                        enrichment_df["TFBS_display"] = enrichment_df["TFBS"].apply(lambda x: x.capitalize())
                    elif dataset_choice == "Human":
                        enrichment_df["TFBS_display"] = enrichment_df["TFBS"].apply(lambda x: x.upper())
                    else:
                        enrichment_df["TFBS_display"] = enrichment_df["TFBS"]



                    # Pivot and plot as before
                    pivot_df = enrichment_df.pivot(index="TFBS_display", columns="Cluster", values="corrected_pval")
                    pivot_df = pivot_df.sort_values(by=pivot_df.columns.tolist())
        
                    fig, ax = plt.subplots(figsize=(6, max(4, 0.3*len(pivot_df))))
                    import seaborn as sns
                    sns.heatmap(-np.log10(pivot_df), cmap="viridis", annot=False, ax=ax)
                    ax.set_title("-log10(corrected p-values) of TFBS enrichment")
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("TFBS")
                    st.pyplot(fig)

                    csv_enrich = enrichment_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Enrichment Results (CSV)",
                        data=csv_enrich,
                        file_name="incura_tfbs_enrichment.csv",
                        mime="text/csv"
                    )

        

        # --- Show cluster assignments ---
        clustered_df = pd.DataFrame({
            "gene": count_matrix.index,
            "cluster": cluster_labels
        })

        # Capitalise gene/TF names if desired
        if dataset_choice == "Mouse":
            clustered_df["gene"] = clustered_df["gene"].str.capitalize()
        elif dataset_choice == "Human":
            clustered_df["gene"] = clustered_df["gene"].str.upper()

        st.subheader("Cluster Assignments")
        st.dataframe(clustered_df)

        # --- Download ---
        csv = clustered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cluster Assignments (CSV)",
            data=csv,
            file_name="incura_gene_clusters.csv",
            mime="text/csv"
        )

        # --- Optional: KMeans performance metrics ---
        st.subheader("Optional: Evaluate  Clustering Performance")
        run_metrics = st.checkbox("Run KMeans metrics (Inertia & Silhouette) to determine optimal k")

        if run_metrics:
            st.write("Calculating performance metrics across k=2 to k=10...")
            X = count_matrix.values
            k_values = range(2, 10)
            inertias = []
            silhouette_scores = []

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                score = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
                silhouette_scores.append(score)

            fig, ax1 = plt.subplots(figsize=(8, 5))
            color = 'tab:blue'
            ax1.set_xlabel('Number of clusters (k)', fontsize=14)
            ax1.set_ylabel('Inertia', color=color, fontsize=14)
            ax1.plot(k_values, inertias, marker='o', color=color, label='Inertia')
            ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
            ax1.tick_params(axis='x', labelsize=12)

            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Silhouette Score', color=color, fontsize=14)
            ax2.plot(k_values, silhouette_scores, marker='s', linestyle='--', color=color, label='Silhouette Score')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

            plt.title('K-Means Performance Metrics', fontsize=16)
            fig.tight_layout()
            st.pyplot(fig)


else:
    st.warning("Please paste at least 50 valid genes and 4 valid TFs.")

st.markdown(
    """
    <hr style="margin-top: 2em; margin-bottom: 1em">
    <div style='text-align: center; font-size: 0.85em; color: gray;'>
        © 2025 InCURA. By Lorna Wessels. Developed for academic research purposes.
    </div>
    """,
    unsafe_allow_html=True
)
