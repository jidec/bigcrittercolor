import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Sample data: 3D normally distributed
n = 1000
x = np.random.normal(size=n)
y = np.random.normal(size=n)
z = np.random.normal(size=n)
data = np.vstack([x, y, z])
# this data can also be RGB or HSV valeus

# Init the KDE
kde = gaussian_kde(data)

# Evaluate the KDE on the data
density = kde(data)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normalize the colors based on the density
norm = plt.Normalize(vmin=density.min(), vmax=density.max())
colors = plt.cm.viridis(norm(density))

# Display the scatter plot
mesh = ax.scatter(x, y, z, c=colors, marker='o', edgecolors='w', s=50)

# Add a colorbar
mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
mappable.set_array(density)
cbar = plt.colorbar(mappable, ax=ax, shrink=0.75)
cbar.set_label('Density')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()