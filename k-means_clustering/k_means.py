import numpy as np
import matplotlib.pyplot as plt
import sys

# Load the Iris dataset with optional feature selection
def load_iris_data(filepath='irisdata.csv', use_dimensions=None):
    try:
        if use_dimensions is None:
            use_columns = [0, 1, 2, 3]
        else:
            use_columns = use_dimensions
        # Load data, skip header, and use specified columns (features)
        data = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=use_columns)
        return data
    except IOError:
        print(f"Error: File '{filepath}' not found.")
        exit(1)
    except ValueError:
        print(f"Error: File '{filepath}' has incorrect format.")
        exit(1)


# Initialize cluster centers randomly
def initialize_centers(data, K):
    indices = np.random.choice(data.shape[0], K, replace=False)
    centers = data[indices]
    return centers

# Assign data points to the nearest cluster center
def assign_clusters(data, centers):
    distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    cluster_indices = np.argmin(distances, axis=1)
    return cluster_indices

# Update cluster centers based on current assignments
def update_centers(data, cluster_indices, K):
    D = data.shape[1]
    centers = np.zeros((K, D))
    for k in range(K):
        cluster_data = data[cluster_indices == k]
        if len(cluster_data) > 0:
            centers[k] = cluster_data.mean(axis=0)
        else:
            # Reinitialize empty cluster center randomly
            centers[k] = data[np.random.choice(data.shape[0])]
    return centers

# Compute the objective function D
def compute_objective(data, centers, cluster_indices):
    squared_distances = np.linalg.norm(data - centers[cluster_indices], axis=1) ** 2
    D = np.sum(squared_distances)
    return D

# K-Means algorithm
def k_means(data, K, max_iterations=100, tol=1e-6):
    centers = initialize_centers(data, K)
    D_values = []
    for iteration in range(max_iterations):
        cluster_indices = assign_clusters(data, centers)
        centers = update_centers(data, cluster_indices, K)
        D = compute_objective(data, centers, cluster_indices)
        D_values.append(D)
        # Check for convergence
        if iteration > 0 and abs(D_values[-2] - D_values[-1]) < tol:
            print(f"Converged at iteration {iteration}")
            break
    return centers, cluster_indices, D_values

def k_means_track_centers(data, K, max_iterations=100, tol=1e-6):
    centers = initialize_centers(data, K)
    D_values = []
    centers_history = [centers.copy()]  # Store initial centers
    for iteration in range(max_iterations):
        cluster_indices = assign_clusters(data, centers)
        centers = update_centers(data, cluster_indices, K)
        D = compute_objective(data, centers, cluster_indices)
        D_values.append(D)
        centers_history.append(centers.copy())  # Store centers after update
        # Check for convergence
        if iteration > 0 and abs(D_values[-2] - D_values[-1]) < tol:
            print(f"Converged at iteration {iteration}")
            break
    return centers, cluster_indices, D_values, centers_history


"""------------------------------------------------------------------------------------------------------------------------------ PART A"""

def main():
    np.random.seed(10)
    # Load the Iris data for Parts A and B (all four dimensions)
    data_full = load_iris_data('irisdata.csv')
    
    # Values of K to test
    K_values = [2, 3]
    
    # Store results for plotting
    results = {}
    
    # Part A and B: Apply K-Means for each K using all dimensions
    for K in K_values:
        print(f"\nRunning K-Means with K={K} on all 4 dimensions")
        centers, cluster_indices, D_values = k_means(data_full, K)
        results[K] = {
            'centers': centers,
            'cluster_indices': cluster_indices,
            'D_values': D_values
        }
        print(f"Final Objective Function D for K={K}: {D_values[-1]:.4f}")
    
    # Plot the objective function D over iterations for each K
    plt.figure(figsize=(10, 6))
    for K in K_values:
        plt.plot(results[K]['D_values'], label=f'K={K}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function D')
    plt.title('K-Means Learning Curves (All 4 dimensions)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Part C: Use only dimensions 3 and 4 (petal length and width)
    data_2d = load_iris_data('irisdata.csv', use_dimensions=[2, 3])
    
    # Apply K-Means for each K
    for K in K_values:
        print(f"\nRunning K-Means with K={K} on dimensions 3 and 4...")
        centers_history = []  # To store centers at each iteration

        # Modify k_means to store centers at each iteration
        centers, cluster_indices, D_values, centers_history = k_means_track_centers(data_2d, K)
        
        # Plot the data points color-coded by cluster
        plt.figure(figsize=(10, 6))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_indices, cmap='viridis', label='Data Points')
        
        # Plot initial centers
        initial_centers = centers_history[0]
        plt.scatter(initial_centers[:, 0], initial_centers[:, 1], 
                    marker='x', s=200, c='red', label='Initial Centers')
        
        # Plot intermediate centers (every 2 iterations)
        for idx in range(1, len(centers_history) - 1, 2):
            plt.scatter(centers_history[idx][:, 0], centers_history[idx][:, 1], 
                        marker='.', s=50, c='blue', alpha=0.5)
        
        # Plot final centers
        final_centers = centers_history[-1]
        plt.scatter(final_centers[:, 0], final_centers[:, 1], 
                    marker='o', s=100, c='green', label='Final Centers')
        
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.title(f'K-Means Cluster Centers Progression (K={K})')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Terminate the program after computation
    sys.exit("Program terminated after computation.")

"""------------------------------------------------------------------------------------------------------------------------------ PART B, C"""


if __name__ == "__main__":
    main() 