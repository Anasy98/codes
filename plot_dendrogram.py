import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Your data (Euclidean distances table)
data = np.array([
    [0.0, 2.6, 2.8, 2.1, 0.8, 1.7, 3.9, 3.8, 4.1, 4.2],
    [2.6, 0.0, 1.8, 2.3, 2.7, 1.4, 1.9, 1.7, 2.0, 2.2],
    [2.8, 1.8, 0.0, 1.2, 2.9, 2.1, 2.2, 1.6, 2.3, 2.0],
    [2.1, 2.3, 1.2, 0.0, 2.4, 1.8, 2.6, 2.5, 3.0, 2.9],
    [0.8, 2.7, 2.9, 2.4, 0.0, 2.0, 4.2, 4.0, 4.4, 4.5],
    [1.7, 1.4, 2.1, 1.8, 2.0, 0.0, 2.4, 2.6, 2.7, 3.0],
    [3.9, 1.9, 2.2, 2.6, 4.2, 2.4, 0.0, 1.2, 0.6, 1.3],
    [3.8, 1.7, 1.6, 2.5, 4.0, 2.6, 1.2, 0.0, 1.0, 0.6],
    [4.1, 2.0, 2.3, 3.0, 4.4, 2.7, 0.6, 1.0, 0.0, 0.9],
    [4.2, 2.2, 2.0, 2.9, 4.5, 3.0, 1.3, 0.6, 0.9, 0.0]
])

# Convert to condensed distance matrix
condensed_data = squareform(data)

# Perform hierarchical clustering with the 'complete' method
Z = linkage(condensed_data, method='complete')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendro = dendrogram(Z, labels=[f'o{i+1}' for i in range(data.shape[0])])

# Annotate the merge distances
icoord = np.array(dendro['icoord'])
dcoord = np.array(dendro['dcoord'])
for i, d in zip(icoord, dcoord):
    x = 0.5 * sum(i[1:3])
    y = d[1]
    plt.plot(x, y, 'ro')
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Dendrogram (Complete Linkage)')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()

# Print linkage matrix
print("Linkage matrix:")
print(Z)
