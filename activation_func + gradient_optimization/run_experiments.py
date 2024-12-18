# run_specific_experiment.py

import numpy as np
import matplotlib.pyplot as plt
from sigmoid import single_layer_nn  # Ensure this import matches your file structure
from linear_boundaries import load_iris_data
from neural_networks import gradient_descent, calculate_mse, compute_gradient
from matplotlib.lines import Line2D  # For custom legend handles

def plot_decision_boundary(ax, data, w, b, title):
    # Define the species to plot
    target_species = ['versicolor', 'virginica']

    # Filter data for the target species
    filtered_data = data[data['species'].isin(target_species)]

    # Extract features and labels
    petal_length = filtered_data['petal_length'].values
    petal_width = filtered_data['petal_width'].values
    species = filtered_data['species'].values

    # Define colors for each species
    colors = {'versicolor': 'blue', 'virginica': 'green'}

    # Plot data points
    for sp in target_species:
        sp_data = filtered_data[filtered_data['species'] == sp]
        ax.scatter(
            sp_data['petal_length'],
            sp_data['petal_width'],
            label=sp.capitalize(),
            c=colors[sp],
            edgecolor='k',
            s=60,
            marker='o'
        )

    # Generate a grid of points for plotting the decision boundary
    x_min, x_max = petal_length.min() - 0.5, petal_length.max() + 0.5
    y_min, y_max = petal_width.min() - 0.5, petal_width.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute the output of the neural network for each grid point
    Z = single_layer_nn(grid_points, w, b)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    contour = ax.contour(
        xx,
        yy,
        Z,
        levels=[0.5],
        colors='k',
        linewidths=2
    )

    # Adding labels and title
    ax.set_xlabel('Petal Length (cm)', fontsize=10)
    ax.set_ylabel('Petal Width (cm)', fontsize=10)
    ax.set_title(title, fontsize=12)

    # Create legend
    species_handles = [Line2D([0], [0], marker='o', color='w', label=sp.capitalize(),
                             markerfacecolor=colors[sp], markersize=10, markeredgecolor='k') 
                       for sp in target_species]
    ax.legend(handles=species_handles, title='Classes', fontsize=8)
    ax.grid(True)

def plot_learning_curve(ax, mse_list, title):
    ax.plot(range(1, len(mse_list) + 1), mse_list, marker='o', linestyle='-')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True)

def run_experiment():
    # Load Iris data
    filepath = 'irisdata.csv'  # Ensure this path is correct
    data = load_iris_data(filepath)

    # Filter data for Versicolor and Virginica
    target_species = ['versicolor', 'virginica']
    filtered_data = data[data['species'].isin(target_species)]
    X = filtered_data[['petal_length', 'petal_width']].values
    y_true = np.where(filtered_data['species'] == 'versicolor', 0, 1)

    # Set random seed for reproducibility
    np.random.seed(10)

    # Initialize weights and bias randomly
    initial_w = np.random.uniform(-0.5, 0.5, size=2)
    initial_b = np.random.uniform(-0.5, 0.5)
    initial_params = (initial_w, initial_b)
    print(f"Initial Weights: {initial_w}, Initial Bias: {initial_b}")

    # Define hyperparameters
    learning_rate = 0.1  # Step size
    tolerance = 1e-6     # Stopping criterion
    max_steps = 1000     # Maximum iterations

    # Perform gradient descent
    params_list, mse_list = gradient_descent(
        data_vectors=X,
        pattern_classes=y_true,
        initial_params=initial_params,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tolerance=tolerance
    )

    # Determine the step where MSE is reduced by half
    initial_mse = mse_list[0]
    half_mse = initial_mse / 2
    try:
        half_step = next(i for i, mse in enumerate(mse_list) if mse <= half_mse)
    except StopIteration:
        half_step = len(mse_list) - 1  # If half MSE not reached, take the last step

    print(f"Initial MSE: {initial_mse:.6f}")
    print(f"Half MSE: {half_mse:.6f} reached at step {half_step}")
    print(f"Final MSE: {mse_list[-1]:.6f} at step {len(mse_list)-1}")

    # Select parameters at initial, half, and final steps
    selected_steps = [0, half_step, len(mse_list)-1]
    selected_params = [params_list[s] for s in selected_steps]
    selected_labels = ['Initial Weights', 'Half Error', 'Converged']

    # Plot Decision Boundaries
    fig_db, axes_db = plt.subplots(1, 3, figsize=(18, 5))
    for ax, params, label in zip(axes_db, selected_params, selected_labels):
        w, b = params
        plot_decision_boundary(ax, filtered_data, w, b, title=label)

    fig_db.suptitle('Decision Boundaries at Key Stages', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot Learning Curve
    fig_lc, ax_lc = plt.subplots(figsize=(10, 6))
    plot_learning_curve(ax_lc, mse_list, title='Learning Curve')
    ax_lc.axhline(half_mse, color='r', linestyle='--', label='Half MSE')
    ax_lc.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()
