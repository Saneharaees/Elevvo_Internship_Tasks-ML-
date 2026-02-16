import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Page config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Segmentation Analysis")
st.markdown("---")

# --- STEP 1: Data Loading ---
st.header(" Step 1: Data Loading")

# Directly load the dataset (no file upload needed)
try:
    df = pd.read_csv('Mall_Customers.csv')
    st.success("✅ Dataset loaded successfully!")
    
    # Show data preview
    st.subheader("Dataset Preview:")
    st.dataframe(df.head())
    
    st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"**Columns:** {', '.join(df.columns)}")
    
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.info("Please make sure 'Mall_Customers.csv' is in the same directory as this app")
    st.stop()

# --- STEP 2: Feature Selection & Scaling ---
st.header("Step 2: Feature Selection & Scaling")

# Select features (as per original code)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
st.write("**Selected Features:** Annual Income (k$) and Spending Score (1-100)")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.success("✅ Features scaled successfully!")

# Show scaled data preview
st.subheader("Scaled Data Preview (first 5 rows):")
scaled_df = pd.DataFrame(
    X_scaled, 
    columns=['Annual Income (scaled)', 'Spending Score (scaled)']
)
st.dataframe(scaled_df.head())

st.markdown("---")

# --- STEP 3: Elbow Method ---
st.header("Step 3: Elbow Method for Optimal K")

# Calculate WCSS
wcss = []
K_range = range(1, 11)
for i in K_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Create matplotlib figure for elbow method
fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
ax_elbow.plot(range(1, 11), wcss, marker='o', linestyle='--', color='teal', linewidth=2, markersize=8)
ax_elbow.set_title('Elbow Method to Find Optimal K', fontsize=14, fontweight='bold')
ax_elbow.set_xlabel('Number of Clusters', fontsize=12)
ax_elbow.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
ax_elbow.grid(True, alpha=0.3)

# Highlight the elbow point (k=5 as per original code)
ax_elbow.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Optimal K=5')
ax_elbow.legend()

st.pyplot(fig_elbow)

# Show WCSS values
st.subheader("WCSS Values:")
wcss_df = pd.DataFrame({
    'K': list(K_range),
    'WCSS': [round(x, 2) for x in wcss]
})
st.dataframe(wcss_df)

st.info(" **Based on the elbow graph, K=5 appears to be optimal** (as per original code)")

st.markdown("---")

# --- STEP 4: K-Means Clustering ---
st.header(" Step 4: K-Means Clustering (K=5)")

# Perform K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Show cluster distribution
st.subheader("K-Means Cluster Distribution:")
cluster_dist = df['KMeans_Cluster'].value_counts().sort_index()
st.bar_chart(cluster_dist)

# --- STEP 5: DBSCAN Clustering ---
st.header(" Step 5: DBSCAN Clustering (Bonus)")

# Perform DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Show DBSCAN results
st.subheader("DBSCAN Results:")
n_clusters_dbscan = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise = list(df['DBSCAN_Cluster']).count(-1)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Clusters", n_clusters_dbscan)
with col2:
    st.metric("Noise Points", n_noise)
with col3:
    st.metric("Clustered Points", len(df) - n_noise)

st.markdown("---")

# --- STEP 6: Visualization (2D Plots) with Custom Labels ---
st.header(" Step 6: Customer Segments Visualization")

# Create the two subplots with adjusted spacing for horizontal color bar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- K-Means Plot with Custom Labels ---
# Color mapping for K-Means clusters
kmeans_colors = ['red', 'blue', 'green', 'purple', 'orange']
kmeans_labels = ['Red Cross', 'Blue Dots', 'Green Dots', 'Purple Dots', 'Orange Dots']

# Plot K-Means clusters with custom colors
for cluster_id in range(5):
    cluster_data = df[df['KMeans_Cluster'] == cluster_id]
    ax1.scatter(
        cluster_data['Annual Income (k$)'], 
        cluster_data['Spending Score (1-100)'],
        c=kmeans_colors[cluster_id], 
        label=kmeans_labels[cluster_id],
        s=100,
        alpha=0.7,
        marker='o'
    )

# Add centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax1.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=200, 
    c='black', 
    marker='X', 
    edgecolors='white',
    linewidth=2,
    label='Centroids'
)

ax1.set_title('K-Means Customer Segments', fontsize=14, fontweight='bold')
ax1.set_xlabel('Annual Income (k$)', fontsize=12)
ax1.set_ylabel('Spending Score (1-100)', fontsize=12)
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# --- DBSCAN Plot with Custom Labels and Horizontal Color Bar ---
# Color mapping for DBSCAN clusters (excluding noise)
dbscan_colors = ['red', 'yellow', 'pink', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta']
dbscan_labels = ['Red Cluster', 'Yellow Cluster', 'Pink Cluster', 'Blue Cluster', 
                 'Green Cluster', 'Purple Cluster', 'Orange Cluster', 'Brown Cluster', 
                 'Cyan Cluster', 'Magenta Cluster']

# Get unique clusters (excluding noise for now)
unique_clusters = sorted([c for c in df['DBSCAN_Cluster'].unique() if c != -1])
noise_points = df[df['DBSCAN_Cluster'] == -1]

# Plot DBSCAN clusters
for i, cluster_id in enumerate(unique_clusters):
    cluster_data = df[df['DBSCAN_Cluster'] == cluster_id]
    color_idx = i % len(dbscan_colors)
    ax2.scatter(
        cluster_data['Annual Income (k$)'], 
        cluster_data['Spending Score (1-100)'],
        c=dbscan_colors[color_idx], 
        label=f"Cluster {cluster_id}",
        s=100,
        alpha=0.7,
        marker='o'
    )

# Plot noise points in BLACK
if len(noise_points) > 0:
    ax2.scatter(
        noise_points['Annual Income (k$)'], 
        noise_points['Spending Score (1-100)'],
        c='black', 
        label='Noise',
        s=100,
        alpha=0.7,
        marker='x',
        linewidths=2
    )

ax2.set_title('DBSCAN Customer Segments', fontsize=14, fontweight='bold')
ax2.set_xlabel('Annual Income (k$)', fontsize=12)
ax2.set_ylabel('Spending Score (1-100)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Create horizontal color bar/legend at the bottom of DBSCAN plot
# Create custom legend handles
legend_elements = []

# Add cluster legend items
for i, cluster_id in enumerate(unique_clusters[:7]):  # Limit to first 7 clusters to avoid overcrowding
    color_idx = i % len(dbscan_colors)
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dbscan_colors[color_idx],
               markersize=10, label=f'Cluster {cluster_id}')
    )

# Add noise to legend
if len(noise_points) > 0:
    legend_elements.append(
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black',
               markersize=10, markeredgecolor='black', label='Noise (Black X)')
    )

# Add legend at the bottom of the plot (horizontal)
ax2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25), 
           ncol=4, fontsize=9, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
st.pyplot(fig)

# Add interpretation with cluster details
st.info("""
**Visualization Interpretation:**
- **K-Means** creates spherical clusters based on distance
  - Red Cross, Blue Dots, Green Dots, Purple Dots, Orange Dots
- **DBSCAN** identifies clusters based on density
  - Colored dots represent different density-based clusters
  - **Black X marks represent noise points (-1)**
  - Horizontal color bar at the bottom shows cluster colors
""")

st.markdown("---")

# --- STEP 7: Average Spending Analysis ---
st.header(" Step 7: Average Spending Analysis ")

# Calculate average spending per cluster
avg_spending = df.groupby('KMeans_Cluster')['Spending Score (1-100)'].agg(['mean', 'count']).round(2)
avg_spending.columns = ['Average Spending', 'Number of Customers']

st.subheader("Average Spending Score per K-Means Cluster:")
st.dataframe(avg_spending)

# Create a bar plot for average spending
fig_spending, ax_spending = plt.subplots(figsize=(10, 5))
clusters = avg_spending.index
avg_values = avg_spending['Average Spending']
colors = ['red', 'blue', 'green', 'purple', 'orange']

bars = ax_spending.bar(clusters, avg_values, color=colors, edgecolor='black')
ax_spending.set_title('Average Spending Score by Cluster', fontsize=14, fontweight='bold')
ax_spending.set_xlabel('Cluster', fontsize=12)
ax_spending.set_ylabel('Average Spending Score', fontsize=12)
ax_spending.set_xticks(clusters)
ax_spending.set_xticklabels(['Red Cross', 'Blue Dots', 'Green Dots', 'Purple Dots', 'Orange Dots'])

# Add value labels on bars
for bar, val in zip(bars, avg_values):
    height = bar.get_height()
    ax_spending.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
st.pyplot(fig_spending)

# --- Additional Summary Statistics ---
st.markdown("---")
st.header("📊 Summary Statistics")

col_sum1, col_sum2 = st.columns(2)

with col_sum1:
    st.subheader("K-Means Cluster Summary")
    kmeans_summary = df.groupby('KMeans_Cluster').agg({
        'Annual Income (k$)': ['mean', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'min', 'max'],
        'KMeans_Cluster': 'count'
    }).round(2)
    kmeans_summary.columns = ['Income_Mean', 'Income_Min', 'Income_Max', 
                              'Spending_Mean', 'Spending_Min', 'Spending_Max', 'Count']
    # Add cluster names
    kmeans_summary.index = ['Red Cross', 'Blue Dots', 'Green Dots', 'Purple Dots', 'Orange Dots']
    st.dataframe(kmeans_summary)

with col_sum2:
    st.subheader("DBSCAN Cluster Summary")
    dbscan_summary = df[df['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster').agg({
        'Annual Income (k$)': ['mean', 'min', 'max'],
        'Spending Score (1-100)': ['mean', 'min', 'max'],
        'DBSCAN_Cluster': 'count'
    }).round(2)
    dbscan_summary.columns = ['Income_Mean', 'Income_Min', 'Income_Max', 
                              'Spending_Mean', 'Spending_Min', 'Spending_Max', 'Count']
    
    # Add color names to DBSCAN clusters
    color_names = ['Red', 'Yellow', 'Pink', 'Blue', 'Green', 'Purple', 'Orange', 'Brown', 'Cyan', 'Magenta']
    new_index = []
    for i, idx in enumerate(dbscan_summary.index):
        if i < len(color_names):
            new_index.append(f"{color_names[i]} (Cluster {idx})")
        else:
            new_index.append(f"Cluster {idx}")
    dbscan_summary.index = new_index
    st.dataframe(dbscan_summary)
    
    # Show noise summary
    if n_noise > 0:
        st.write(f"**Noise Points (-1):** {n_noise} customers (shown as **Black X** in the plot)")

# --- Original Console Output ---
st.markdown("---")
st.header(" Original Console Output")

console_output = f"""
Dataset Preview:
{df.head().to_string()}

--- Bonus: Average Spending Per Cluster ---
Red Cross: {avg_spending.loc[0, 'Average Spending'] if 0 in avg_spending.index else 'N/A'}
Blue Dots: {avg_spending.loc[1, 'Average Spending'] if 1 in avg_spending.index else 'N/A'}
Green Dots: {avg_spending.loc[2, 'Average Spending'] if 2 in avg_spending.index else 'N/A'}
Purple Dots: {avg_spending.loc[3, 'Average Spending'] if 3 in avg_spending.index else 'N/A'}
Orange Dots: {avg_spending.loc[4, 'Average Spending'] if 4 in avg_spending.index else 'N/A'}
"""

st.code(console_output, language='python')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 Customer Segmentation Dashboard | Based on Mall_Customers.csv</p>
    <p style='font-size: 12px; color: gray;'>Using K-Means (K=5) and DBSCAN (eps=0.5, min_samples=5)</p>
    <p style='font-size: 12px; color: gray;'>K-Means: Red Cross, Blue Dots, Green Dots, Purple Dots, Orange Dots | DBSCAN: Colored clusters with <span style='color: black; font-weight: bold;'>Black X</span> for noise</p>
    <p style='font-size: 12px; color: gray;'>DBSCAN horizontal color legend at bottom shows cluster colors</p>
</div>
""", unsafe_allow_html=True)