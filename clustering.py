import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


df = pd.read_csv("supernova_data.csv")
print(f"\n✓ data loaded")

# Scaling
feature_cols = ['peak_mag', 'decline_rate', 'B_V_color', 'duration_days', 'host_mass']
X = df[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "-" * 70)
print("features")
print(df[feature_cols].describe())



def k_means(X, k, max_iter=300, random_state=42):
    np.random.seed(random_state)

    # Initialize centroids randomly
    centroid_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[centroid_indices]
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        new_centroids = np.copy(centroids)
        # Update centroids as mean of points in each cluster
        for i in range(k):
            cluster_points = X[new_labels == i]
            if len(cluster_points) == 0:
                continue
            new_centroids[i] = cluster_points.mean(axis=0)

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids
        labels = new_labels

    return labels, centroids




# K-Means
labels_kmeans, kmeans = k_means(X_scaled, k=3)
df['kmeans_labels'] = labels_kmeans

print("✓ clustering algorithm completed")


# VISUALIZATION

# Figure 1: Comparison of clustering results in 2D
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Supernova Clustering Results Comparison', fontsize=16, fontweight='bold')

viz_features = [
    ('peak_mag', 'decline_rate', 0, 1),
    ('B_V_color', 'duration_days', 2, 3),
]

for row_idx, (feat_x, feat_y, x_idx, y_idx) in enumerate(viz_features):
    for col_idx, algo in enumerate(['true_type', 'kmeans_labels']):
        ax = axes[row_idx, col_idx]

        if algo == 'true_type':
            for sn_type, color in zip(['Ia', 'II', 'Ib/c'], ['#f2cc8f', '#3d405b', '#e07a5f']):
                mask = df[algo] == sn_type
                ax.scatter(df.loc[mask, feat_x], df.loc[mask, feat_y],
                           c=color, label=sn_type, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            title_prefix = "True Types"
        else:
            scatter = ax.scatter(df[feat_x], df[feat_y], c=df[algo],
                                 cmap='viridis', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            title_prefix = algo.replace('_labels', '').upper()

        ax.set_xlabel(feat_x.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel(feat_y.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{title_prefix}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if algo == 'true_type':
            ax.legend()

plt.tight_layout()
plt.show()

# Figure 2: 3D visualization
fig = plt.figure(figsize=(12, 5))

for idx, algo in enumerate(['true_type', 'kmeans_labels']):
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

    if algo == 'true_type':
        colors = df[algo].map({'Ia': '#f2cc8f', 'II': '#3d405b', 'Ib/c': '#e07a5f'})
        title = "True Types"
    else:
        colors = df[algo].map({0: '#e07a5f', 1: '#f2cc8f', 2: '#3d405b'})
        title = algo.replace('_labels', '').upper()

    scatter = ax.scatter(df['peak_mag'], df['decline_rate'], df['B_V_color'],
                         c=colors, alpha=0.7, s=30, edgecolors='k', linewidth=0.3)

    ax.set_xlabel('Peak Magnitude', fontsize=9)
    ax.set_ylabel('Decline Rate', fontsize=9)
    ax.set_zlabel('B-V Color', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')

plt.suptitle('3D Visualization of Supernova Clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()