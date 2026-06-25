import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, davies_bouldin_score
from io import StringIO
import gzip

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WIUC Gene Expression Anomaly Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAlert {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TITLE & INTRODUCTION
# ─────────────────────────────────────────
st.title("🔍 WIUC Gene Expression Anomaly Detector")
st.markdown("""
This tool analyses gene expression data to detect **unusual biological samples** 
that may represent disease states, rare subtypes, or clinically significant outliers.

**How it works — 4 stages:**
1. 🔵 **PCA** reduces thousands of genes into fewer patterns
2. 🟢 **Factor Analysis** is applied to PCA output to uncover hidden biological signals
3. 🔴 **LOF** scores each sample by how isolated it is from its neighbours
4. ✅ **Consensus** flags samples that both methods agree are anomalous

### Supported File Formats:
- NCBI GEO files: `.txt`, `.tsv`, `.txt.gz`
- Excel files: `.xlsx`, `.xls`
- Plain tables: `.csv`

> ✅ Files must have **genes as rows** and **samples as columns**.
""")

# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload your gene expression file",
    type=["txt", "tsv", "csv", "gz", "xlsx", "xls"],
    help="Genes as rows, samples as columns"
)

if uploaded_file is None:
    st.info("👆 Please upload a file to begin analysis.")
    st.stop()

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith('.gz'):
            with gzip.open(uploaded_file, 'rt', encoding='utf-8') as f:
                lines = [line for line in f if not line.startswith(('!', '#'))]
            clean = '\n'.join(lines)
            sep = '\t' if '\t' in clean[:500] else ','
            return pd.read_csv(StringIO(clean), sep=sep, index_col=0, low_memory=False)

        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file, index_col=0)

        else:
            content = uploaded_file.read().decode('utf-8')
            lines = [l for l in content.splitlines() if not l.startswith(('!', '#'))]
            clean = '\n'.join(lines)
            sep = '\t' if '\t' in clean[:500] else ','
            return pd.read_csv(StringIO(clean), sep=sep, index_col=0, low_memory=False)

    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

data = load_data(uploaded_file)

if data is None or data.empty:
    st.error("Uploaded file is empty or could not be parsed.")
    st.stop()
if data.shape[0] < 3 or data.shape[1] < 3:
    st.error("Dataset must have at least 3 genes (rows) and 3 samples (columns).")
    st.stop()

st.success(f"✅ Loaded **{data.shape[1]} samples** and **{data.shape[0]} genes** successfully.")

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────
@st.cache_data
def run_pipeline(data, _sample_ids):
    # --- Step 1: Clean ---
    data_clean = data.apply(pd.to_numeric, errors='coerce').dropna()
    if data_clean.empty:
        return None, "No valid numeric data found after cleaning."

    # Transpose: rows = samples, columns = genes
    data_T = data_clean.T

    # Log2 transform (handles zeros safely)
    data_log = np.log2(data_T + 1)
    data_log.columns = [str(c).strip() for c in data_log.columns]

    # Standardise
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_log)

    # --- Step 2: PCA (dimensionality reduction) ---
    n_components_pca = min(10, data_scaled.shape[1], data_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components_pca)
    pca_result = pca.fit_transform(data_scaled)         # shape: (n_samples, n_components_pca)
    pca_2d = pca_result[:, :2]                          # first 2 PCs for plotting

    # --- Step 3: FA applied ON TOP of PCA output (not raw data) ---
    n_fa = min(2, pca_result.shape[1], pca_result.shape[0] - 1)
    fa = FactorAnalysis(n_components=n_fa, random_state=42)
    fa_result = fa.fit_transform(pca_result)            # FA on PCA output

    # --- Step 4: LOF on FA result ---
    n_neighbors = min(10, len(fa_result) - 1)
    if n_neighbors < 2:
        return None, "Not enough samples for LOF detection (need at least 3)."

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(fa_result)
    lof_scores = -lof.negative_outlier_factor_

    # Adaptive threshold: 95th percentile
    threshold = np.percentile(lof_scores, 95)
    labels = np.where(lof_scores > threshold, -1, 1)    # -1 = anomaly, 1 = normal

    # --- Step 5: Metrics ---
    lof_sep_ratio = None
    db_index = None
    sil_score = None

    anomaly_mask = labels == -1
    normal_mask  = labels == 1

    if anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
        mean_anom  = np.mean(lof_scores[anomaly_mask])
        mean_norm  = np.mean(lof_scores[normal_mask])
        lof_sep_ratio = mean_anom / mean_norm if mean_norm > 0 else None

        if len(set(labels)) > 1:
            try:
                db_index  = davies_bouldin_score(fa_result, labels)
                sil_score = silhouette_score(fa_result, labels)
            except Exception:
                pass

    # --- Step 6: Build result DataFrame ---
    plot_df = pd.DataFrame({
        'Sample':          _sample_ids,
        'PC1':             pca_2d[:, 0],
        'PC2':             pca_2d[:, 1],
        'FA1':             fa_result[:, 0],
        'FA2':             fa_result[:, 1] if fa_result.shape[1] > 1 else np.zeros(len(fa_result)),
        'LOF_Score':       lof_scores,
        'Anomaly':         labels
    })

    # --- Step 7: Heatmap data ---
    gene_var  = data_log.var(axis=0)
    top_genes = gene_var.nlargest(min(50, len(gene_var))).index

    anom_samples   = plot_df[plot_df['Anomaly'] == -1]['Sample'].tolist()
    normal_samples = plot_df[plot_df['Anomaly'] ==  1]['Sample'].tolist()
    ordered        = anom_samples + normal_samples[:min(20, len(normal_samples))]
    heatmap_data   = data_log.loc[ordered, top_genes]
    n_anomalous    = len(anom_samples)

    return {
        'plot_df':       plot_df,
        'pca':           pca,
        'pca_2d':        pca_2d,
        'fa_result':     fa_result,
        'lof_scores':    lof_scores,
        'threshold':     threshold,
        'labels':        labels,
        'heatmap_data':  heatmap_data,
        'n_anomalous':   n_anomalous,
        'data_log':      data_log,
        'db_index':      db_index,
        'sil_score':     sil_score,
        'lof_sep_ratio': lof_sep_ratio,
        'anom_samples':  anom_samples,
        'normal_samples':normal_samples,
    }, None

sample_ids = data.columns.tolist()
result, error = run_pipeline(data, sample_ids)

if error:
    st.error(f"Pipeline failed: {error}")
    st.stop()

# Unpack
plot_df        = result['plot_df']
pca            = result['pca']
threshold      = result['threshold']
labels         = result['labels']
lof_scores     = result['lof_scores']
heatmap_data   = result['heatmap_data']
n_anomalous    = result['n_anomalous']
data_log       = result['data_log']
db_index       = result['db_index']
sil_score      = result['sil_score']
lof_sep_ratio  = result['lof_sep_ratio']
anom_samples   = result['anom_samples']
normal_samples = result['normal_samples']

colors_map = plot_df['Anomaly'].map({1: 'steelblue', -1: 'crimson'})

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 1 — PCA
# ═══════════════════════════════════════════════════════
st.markdown("## 🔵 Stage 1 — PCA: Overall Gene Expression Map")
st.markdown("""
**What is PCA doing here?**  
Your dataset may have thousands of genes per sample — far too many to visualise directly.  
PCA compresses all those genes into a small number of **"summary directions"** called Principal Components (PCs), 
each capturing the most important patterns of variation across samples.

- **PC1 (X-axis)** = the single direction where samples differ the most  
- **PC2 (Y-axis)** = the second most important direction, independent of PC1  
- **Each dot = one sample** (e.g. a patient or tissue)  
- **Blue = typical**, **Red = anomalous** (labelled after LOF runs — shown here for reference)

> 💡 Samples sitting far away from the main cluster already hint at unusual gene activity.
""")

ev = pca.explained_variance_ratio_
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.scatter(plot_df['PC1'], plot_df['PC2'],
            c=colors_map, alpha=0.85, s=60, edgecolor='k', linewidth=0.4)
for _, row in plot_df[plot_df['Anomaly'] == -1].iterrows():
    ax1.annotate(str(row['Sample'])[:10], (row['PC1'], row['PC2']),
                 fontsize=7, color='darkred', alpha=0.8,
                 xytext=(4, 4), textcoords='offset points')
ax1.set_xlabel(f"PC1 ({ev[0]:.1%} variance explained)", fontsize=11)
ax1.set_ylabel(f"PC2 ({ev[1]:.1%} variance explained)", fontsize=11)
ax1.set_title("PCA — Sample Distribution in Gene Expression Space", fontsize=12, pad=10)
blue_patch = mpatches.Patch(color='steelblue', label='Normal')
red_patch  = mpatches.Patch(color='crimson',   label='Anomalous')
ax1.legend(handles=[blue_patch, red_patch], loc='best')
ax1.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# Scree plot
st.markdown("""
**Variance Explained per Component:**  
The bar chart below shows how much information each PC captures. 
The first few PCs carry most of the biological signal — the rest is mostly noise.
""")
fig_scree, ax_scree = plt.subplots(figsize=(7, 3))
x_vals = range(1, len(ev) + 1)
ax_scree.bar(x_vals, ev * 100, color='steelblue', edgecolor='k', alpha=0.8)
ax_scree.plot(x_vals, np.cumsum(ev) * 100, color='red', marker='o',
              markersize=4, linewidth=1.5, label='Cumulative %')
ax_scree.set_xlabel("Principal Component")
ax_scree.set_ylabel("Variance Explained (%)")
ax_scree.set_title("Scree Plot — How Much Each PC Contributes")
ax_scree.legend()
ax_scree.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig_scree)
plt.close(fig_scree)

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 2 — FA (on PCA output)
# ═══════════════════════════════════════════════════════
st.markdown("## 🟢 Stage 2 — Factor Analysis: Uncovering Hidden Biological Signals")
st.markdown("""
**What is Factor Analysis doing here — and why after PCA?**  
PCA gave us a compact representation of the data (fewer dimensions, more manageable).  
Factor Analysis (FA) is now applied **to the PCA output** — not the raw data.  
This is intentional: FA models the *hidden latent structure* within the already-reduced space, 
helping separate biological signal from remaining noise.

- **Factor 1 (X-axis)** = primary hidden biological pattern  
- **Factor 2 (Y-axis)** = secondary hidden pattern  
- **Green = typical**, **Red = anomalous**

> 💡 If a sample looks unusual here AND in the PCA map, that is a strong sign it is genuinely anomalous.
""")

fig2, ax2 = plt.subplots(figsize=(7, 5))
colors_fa = plot_df['Anomaly'].map({1: 'seagreen', -1: 'crimson'})
ax2.scatter(plot_df['FA1'], plot_df['FA2'],
            c=colors_fa, alpha=0.85, s=60, edgecolor='k', linewidth=0.4)
for _, row in plot_df[plot_df['Anomaly'] == -1].iterrows():
    ax2.annotate(str(row['Sample'])[:10], (row['FA1'], row['FA2']),
                 fontsize=7, color='darkred', alpha=0.8,
                 xytext=(4, 4), textcoords='offset points')
ax2.set_xlabel("Factor 1", fontsize=11)
ax2.set_ylabel("Factor 2", fontsize=11)
ax2.set_title("Factor Analysis Map — Hidden Biological Factors", fontsize=12, pad=10)
green_patch = mpatches.Patch(color='seagreen', label='Normal')
red_patch2  = mpatches.Patch(color='crimson',  label='Anomalous')
ax2.legend(handles=[green_patch, red_patch2], loc='best')
ax2.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 3 — LOF with Adaptive Threshold
# ═══════════════════════════════════════════════════════
st.markdown("## 🔴 Stage 3 — LOF Anomaly Scoring with Adaptive Threshold")
st.markdown("""
**What is LOF doing?**  
Local Outlier Factor (LOF) assigns every sample a **"weirdness score"** based on how isolated it is 
compared to its nearest neighbours in the Factor Analysis space.

- A score **close to 1.0** = the sample fits in well with its neighbours → **normal**
- A score **much higher than 1.0** = the sample is far more isolated than its neighbours → **anomalous**

**What is the Adaptive Threshold?**  
Rather than using a fixed cutoff, the threshold is set at the **95th percentile** of all LOF scores.  
This means the top 5% most unusual samples are flagged — adapting automatically to your dataset size and distribution.

> 💡 Samples to the **right of the red dashed line** are flagged as anomalies.
""")

# LOF histogram
fig3, ax3 = plt.subplots(figsize=(8, 4))
n_bins = min(30, max(10, len(lof_scores) // 2))
ax3.hist(lof_scores[labels == 1],  bins=n_bins, color='steelblue',
         edgecolor='k', alpha=0.75, label='Normal samples')
ax3.hist(lof_scores[labels == -1], bins=n_bins, color='crimson',
         edgecolor='k', alpha=0.85, label='Anomalous samples')
ax3.axvline(threshold, color='red', linestyle='--', linewidth=2,
            label=f'Threshold (95th pct) = {threshold:.2f}')
ax3.set_xlabel("LOF Score (higher = more unusual)", fontsize=11)
ax3.set_ylabel("Number of Samples", fontsize=11)
ax3.set_title("LOF Score Distribution — FA-Based Anomaly Detection", fontsize=12)
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# LOF per-sample scatter
st.markdown("""
**Per-Sample LOF Score:**  
The plot below shows each sample's individual LOF score so you can identify exactly 
which samples were flagged and how extreme their scores are.
""")
fig4, ax4 = plt.subplots(figsize=(10, 4))
sample_indices = range(len(plot_df))
ax4.scatter(sample_indices, lof_scores,
            c=colors_map, s=50, edgecolor='k', linewidth=0.4, alpha=0.85)
ax4.axhline(threshold, color='red', linestyle='--', linewidth=1.8,
            label=f'Threshold = {threshold:.2f}')
for i, row in plot_df[plot_df['Anomaly'] == -1].iterrows():
    ax4.annotate(str(row['Sample'])[:10], (i, lof_scores[i]),
                 fontsize=7, color='darkred',
                 xytext=(0, 6), textcoords='offset points', ha='center')
ax4.set_xlabel("Sample Index", fontsize=11)
ax4.set_ylabel("LOF Score", fontsize=11)
ax4.set_title("LOF Score Per Sample — Anomalous Samples Labelled", fontsize=12)
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 4 — QUANTITATIVE METRICS
# ═══════════════════════════════════════════════════════
st.markdown("## 📐 Stage 4 — Quantitative Performance Metrics")
st.markdown("""
These three metrics tell us **how well the anomaly detection is working** — 
whether the anomalous and normal samples are truly distinct from each other.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔵 Silhouette Score")
    if sil_score is not None:
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        if sil_score > 0.5:
            st.success("Strong cluster separation")
        elif sil_score > 0.25:
            st.warning("Moderate separation")
        else:
            st.error("Weak separation")
    else:
        st.info("Not computable")
    st.caption("**Range: -1 to +1**. Closer to +1 means anomalous and normal samples are clearly separated.")

with col2:
    st.markdown("### 🟢 Davies-Bouldin Index")
    if db_index is not None:
        st.metric("Davies-Bouldin Index", f"{db_index:.3f}")
        if db_index < 0.5:
            st.success("Excellent separation")
        elif db_index < 1.0:
            st.warning("Good separation")
        else:
            st.error("Overlapping groups")
    else:
        st.info("Not computable")
    st.caption("**Lower is better.** Measures how compact and well-separated the two groups are.")

with col3:
    st.markdown("### 🔴 LOF Separation Ratio")
    if lof_sep_ratio is not None:
        st.metric("LOF Separation Ratio", f"{lof_sep_ratio:.2f}x")
        if lof_sep_ratio > 2.0:
            st.success("Strong anomaly signal")
        elif lof_sep_ratio > 1.5:
            st.warning("Moderate signal")
        else:
            st.error("Weak signal")
    else:
        st.info("Not computable")
    st.caption("**Higher is better.** Ratio of average anomaly LOF score to average normal LOF score.")

# Metric bar chart
st.markdown("""
**Visual Comparison of Metrics:**  
The bar chart below compares the three metrics side by side on a normalised scale for easy reading.
""")
metrics_available = {}
if sil_score  is not None: metrics_available['Silhouette\n(higher=better)']      = max(0, sil_score)
if db_index   is not None: metrics_available['Davies-Bouldin\n(lower=better)']   = db_index
if lof_sep_ratio is not None: metrics_available['LOF Sep. Ratio\n(higher=better)'] = lof_sep_ratio

if metrics_available:
    fig_m, ax_m = plt.subplots(figsize=(7, 4))
    bar_colors = ['steelblue', 'seagreen', 'tomato']
    bars = ax_m.bar(metrics_available.keys(), metrics_available.values(),
                    color=bar_colors[:len(metrics_available)], edgecolor='k', alpha=0.85)
    for bar, val in zip(bars, metrics_available.values()):
        ax_m.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_m.set_ylabel("Metric Value", fontsize=11)
    ax_m.set_title("Performance Metrics Summary", fontsize=12)
    ax_m.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_m)
    plt.close(fig_m)

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 5 — BOX PLOT COMPARISON
# ═══════════════════════════════════════════════════════
st.markdown("## 📦 Stage 5 — Gene Expression: Anomalous vs Normal")
st.markdown("""
**What does this box plot show?**  
Each box represents the distribution of **average gene expression** across all genes for that group.

- The **line in the middle** of the box = median expression  
- The **box edges** = 25th and 75th percentile (where most samples sit)  
- The **whiskers** = overall spread  
- **Dots** outside whiskers = extreme samples  

> 💡 If the two boxes are clearly separated, anomalous samples genuinely express genes differently from normal ones — 
confirming biological relevance, not just statistical noise.
""")

if anom_samples and normal_samples:
    mean_anom_expr   = data_log.loc[anom_samples].mean(axis=1).values
    mean_normal_expr = data_log.loc[normal_samples[:min(50, len(normal_samples))]].mean(axis=1).values

    fig5, ax5 = plt.subplots(figsize=(6, 5))
    bp = ax5.boxplot(
        [mean_normal_expr, mean_anom_expr],
        tick_labels=['Normal Samples', 'Anomalous Samples'],
        patch_artist=True,
        notch=False,
        boxprops=dict(facecolor='steelblue', alpha=0.7),
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='crimson', markersize=6, alpha=0.6)
    )
    bp['boxes'][1].set_facecolor('crimson')
    bp['boxes'][1].set_alpha(0.7)
    ax5.set_ylabel("Mean Log₂ Gene Expression", fontsize=11)
    ax5.set_title("Average Gene Expression Distribution by Group", fontsize=12)
    ax5.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
else:
    st.info("Box plot requires both anomalous and normal samples to be present.")

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 6 — HEATMAP WITH BOUNDARY LINE
# ═══════════════════════════════════════════════════════
st.markdown("## 🧬 Stage 6 — Gene Activity Heatmap")
st.markdown("""
**How to read this heatmap:**  
Think of it as a **thermal camera for gene activity** across all your samples.

- **Each column** = one sample  
- **Each row** = one gene (top 50 most variable genes shown)  
- 🟡 **Yellow** = high gene activity (gene is turned ON)  
- 🟣 **Purple** = low gene activity (gene is turned OFF)  
- **The white dashed vertical line** separates anomalous samples (left) from normal samples (right)

> 💡 Look for **consistent blocks of yellow or purple on the left side** that don't appear on the right. 
Those patterns indicate the anomalous samples share a common biological signature.
""")

if not heatmap_data.empty:
    fig6, ax6 = plt.subplots(figsize=(13, 8))
    sns.heatmap(
        heatmap_data.T,
        cmap='viridis',
        xticklabels=False,
        yticklabels=True if heatmap_data.shape[1] <= 50 else False,
        cbar_kws={'label': 'Log₂ Gene Expression'},
        ax=ax6
    )
    # Boundary line between anomalous and normal
    if n_anomalous > 0 and n_anomalous < heatmap_data.shape[0]:
        ax6.axvline(x=n_anomalous, color='white', linewidth=2.5,
                    linestyle='--', label='Anomalous | Normal boundary')
        ax6.text(n_anomalous / 2, -1.2, '← Anomalous',
                 ha='center', va='top', color='crimson',
                 fontsize=10, fontweight='bold',
                 transform=ax6.get_xaxis_transform())
        ax6.text(n_anomalous + (heatmap_data.shape[0] - n_anomalous) / 2, -1.2,
                 'Normal →', ha='center', va='top', color='steelblue',
                 fontsize=10, fontweight='bold',
                 transform=ax6.get_xaxis_transform())
    ax6.set_title(f"Gene Activity Heatmap — {n_anomalous} Anomalous | "
                  f"{heatmap_data.shape[0] - n_anomalous} Normal Samples",
                  fontsize=12, pad=15)
    ax6.set_xlabel("Samples", fontsize=11)
    ax6.set_ylabel("Genes (Top 50 Most Variable)", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)
    st.info("💡 The **white dashed line** marks the boundary between anomalous (left) and normal (right) samples.")

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 7 — SUMMARY METRICS TABLE
# ═══════════════════════════════════════════════════════
st.markdown("## 📋 Stage 7 — Pipeline Summary Table")
st.markdown("""
A full summary of all key numbers from the analysis in one place.
""")

total_variance = sum(pca.explained_variance_ratio_[:2]) * 100

summary_data = {
    'Metric': [
        'Total Samples Analysed',
        'Total Genes (Features)',
        'Anomalous Samples Detected',
        'Normal Samples',
        'Anomaly Rate (%)',
        'PCA — Variance Explained (PC1)',
        'PCA — Variance Explained (PC2)',
        'PCA — Total Variance (PC1+PC2)',
        'LOF Adaptive Threshold (95th pct)',
        'Min LOF Score',
        'Max LOF Score',
        'Mean LOF Score (Normal)',
        'Mean LOF Score (Anomalous)',
        'LOF Separation Ratio',
        'Silhouette Score',
        'Davies-Bouldin Index',
    ],
    'Value': [
        len(plot_df),
        data.shape[0],
        n_anomalous,
        len(plot_df) - n_anomalous,
        f"{100 * n_anomalous / len(plot_df):.1f}%",
        f"{pca.explained_variance_ratio_[0]:.1%}",
        f"{pca.explained_variance_ratio_[1]:.1%}",
        f"{total_variance:.1f}%",
        f"{threshold:.3f}",
        f"{lof_scores.min():.3f}",
        f"{lof_scores.max():.3f}",
        f"{lof_scores[labels==1].mean():.3f}"  if (labels==1).any()  else "N/A",
        f"{lof_scores[labels==-1].mean():.3f}" if (labels==-1).any() else "N/A",
        f"{lof_sep_ratio:.3f}" if lof_sep_ratio else "N/A",
        f"{sil_score:.3f}"    if sil_score     else "N/A",
        f"{db_index:.3f}"     if db_index      else "N/A",
    ],
    'Interpretation': [
        '—', '—',
        'Samples flagged by LOF as anomalous',
        'Samples within normal range',
        'Proportion of dataset flagged',
        'Primary gene expression pattern',
        'Secondary gene expression pattern',
        'Combined explanatory power of 2D plot',
        'Top 5% most unusual score cutoff',
        'Most typical sample score (≈1.0 is normal)',
        'Most extreme anomaly detected',
        'Average score for normal group',
        'Average score for anomalous group',
        '>2.0 = strong separation; >1.5 = moderate',
        '>0.5 = strong; >0.25 = moderate; <0.25 = weak',
        '<0.5 = excellent; <1.0 = good; >1.0 = overlap',
    ]
}

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════
# STAGE 8 — RESULTS & DOWNLOAD
# ═══════════════════════════════════════════════════════
st.markdown("## ✅ Detected Anomalies")

if n_anomalous > 0:
    st.markdown(f"Found **{n_anomalous} anomalous sample(s)** out of **{len(plot_df)} total samples** "
                f"({100*n_anomalous/len(plot_df):.1f}% of dataset).")

    anomaly_report = plot_df[plot_df['Anomaly'] == -1][
        ['Sample', 'PC1', 'PC2', 'FA1', 'FA2', 'LOF_Score']
    ].copy()
    anomaly_report['LOF_Score'] = anomaly_report['LOF_Score'].round(4)
    anomaly_report = anomaly_report.reset_index(drop=True)

    st.dataframe(anomaly_report, use_container_width=True, hide_index=True)

    # Full report CSV
    full_report = plot_df.copy()
    full_report['Status'] = full_report['Anomaly'].map({-1: 'Anomalous', 1: 'Normal'})
    full_report['LOF_Score'] = full_report['LOF_Score'].round(4)
    csv_data = full_report[['Sample', 'Status', 'LOF_Score', 'PC1', 'PC2', 'FA1', 'FA2']].to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Full Anomaly Report (CSV)",
        data=csv_data,
        file_name="gene_anomaly_report.csv",
        mime="text/csv",
        help="Downloads all samples with their anomaly status and LOF scores"
    )
else:
    st.success("✅ No anomalies detected. All samples appear to have typical gene expression patterns.")

st.markdown("---")
st.caption("💡 WIUC Gene Expression Anomaly Detector | PCA → FA → LOF Pipeline | Accepts GEO .txt.gz, Excel, CSV")
