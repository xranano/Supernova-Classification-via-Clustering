# Supernova Classification via Clustering

## Project Overview
This project demonstrates the use of unsupervised clustering algorithms to classify synthetic supernovae into types Ia, II, and Ib/c based on photometric and host galaxy features. The main goal is to explore the ability of clustering methods to recover inherent patterns in astrophysical data.

---

## Dataset
- **Source:** Synthetic supernova dataset (`supernova_data.csv`)
- **Number of samples:** 300  
  - Type Ia: 100  
  - Type II: 120  
  - Type Ib/c: 80  
- **Features:**
  - `peak_mag`: Peak magnitude
  - `decline_rate`: Decline rate
  - `B_V_color`: B-V color index
  - `duration_days`: Duration of the supernova event
  - `host_mass`: Host galaxy mass

---

## Methodology
1. **Data Preprocessing**
   - Standardization using Z-score normalization to ensure features contribute equally.
2. **Clustering Algorithms**
   - **K-Means:** Partition data into 3 clusters by minimizing Euclidean distances.
   - **Optional Extensions:** K-Medoids, DBSCAN (with automatic `eps` tuning), or hierarchical clustering.
3. **Evaluation**
   - Silhouette Score, Davies-Bouldin Score, and Calinski-Harabasz Score to assess clustering quality.

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/xranano/Supernova-Clustering.git
cd Supernova-Clustering
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
---
## Usage
1. Place supernova_data.csv in the project directory.
2. Run the clustering script:
```bash
python clustering.py
```
3. The script will:
   -   Standardize the data
   - Perform K-Means clustering (and optionally other algorithms)
   - Generate 2D and 3D visualizations of clustering results
   - Print clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz scores)
---
## Visualization
- 2D Scatter Plots: Compare true supernova types vs. clustering results.
- 3D Scatter Plots: Visualize clusters in feature space (peak_mag, decline_rate, B_V_color).

---
## Results
- K-Means clustering recovers the main structure of the supernova types:
- Type Ia forms a distinct cluster with high peak magnitudes and short durations.
- Type II separates due to slower decline rates and longer durations.
- Type Ib/c partially overlaps with other clusters.
- Silhouette Score (K-Means): ~0.30 (moderate separation).