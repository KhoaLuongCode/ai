import numpy as np
import matplotlib.pyplot as plt
from k_means import load_iris_data, k_means

def plot_decision_boundaries(data, centers, cluster_indices, K):
    # Create a meshgrid over the data range
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Flatten the grid to pass into the distance computation
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute distances from each point in the grid to each cluster center
    distances = np.linalg.norm(grid[:, np.newaxis] - centers, axis=2)
    
    # Assign each grid point to the nearest cluster center
    grid_cluster_indices = np.argmin(distances, axis=1)
    
    # Reshape the cluster assignments back into the meshgrid shape
    grid_cluster_indices = grid_cluster_indices.reshape(xx.shape)
    
    # Plot the decision boundaries
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, grid_cluster_indices, alpha=0.3, cmap='viridis')
    
    # Plot the data points
    plt.scatter(data[:, 0], data[:, 1], c=cluster_indices, cmap='viridis', edgecolor='k', label='Data Points')
    
    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
    
    plt.title(f'K-Means Decision Boundaries (K={K})')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.show()

def main():
    np.random.seed(10)
    # Load the Iris data (dimensions 3 and 4, petal length and width)
    data = load_iris_data('irisdata.csv', use_dimensions=[2, 3])
    
    # Values of K to test
    K_values = [2, 3]
    
    for K in K_values:
        # Run k-means clustering
        centers, cluster_indices, D_values = k_means(data, K)
        
        # Plot the decision boundaries
        plot_decision_boundaries(data, centers, cluster_indices, K)
    
if __name__ == '__main__':
    main()
