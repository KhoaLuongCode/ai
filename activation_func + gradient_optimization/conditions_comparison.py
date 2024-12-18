import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sigmoid import single_layer_nn  # Ensure this import matches your file structure
from linear_boundaries import load_iris_data
from neural_networks import gradient_descent
from matplotlib.lines import Line2D  # For custom legend handles
import itertools

def plot_decision_boundary(ax, data, w, b, title):
    target_species = ['versicolor', 'virginica']

    filtered_data = data[data['species'].isin(target_species)]

    petal_length = filtered_data['petal_length'].values
    petal_width = filtered_data['petal_width'].values
    species = filtered_data['species'].values

    colors = {'versicolor': 'blue', 'virginica': 'green'}

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

    x_min, x_max = petal_length.min() - 0.5, petal_length.max() + 0.5
    y_min, y_max = petal_width.min() - 0.5, petal_width.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

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

def plot_learning_curves(ax, mse_dict, title):
    for label, mse_list in mse_dict.items():
        ax.plot(range(1, len(mse_list) + 1), mse_list, marker='o', linestyle='-', label=label)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True)

def compare_combinations():
    filepath = 'irisdata.csv'  # Ensure this path is correct
    data = load_iris_data(filepath)

    # Filter data for Versicolor and Virginica
    target_species = ['versicolor', 'virginica']
    filtered_data = data[data['species'].isin(target_species)]
    X = filtered_data[['petal_length', 'petal_width']].values
    y_true = np.where(filtered_data['species'] == 'versicolor', 0, 1)

    # Define initial conditions
    np.random.seed(42)  # For reproducibility
    initial_conditions = {
        'Good Initial Guess': (np.array([0.163, 0.97]), -2.418),
        'Random Initial Guess': (np.random.uniform(-0.5, 0.5, size=2), np.random.uniform(-0.5, 0.5)),
        'Poor Initial Guess': (np.array([-1.0, -1.0]), 5.0)
    }

    # Define learning rates
    learning_rates = [0.01, 0.1, 0.5]

    # Define stopping criteria (tolerances)
    tolerances = [1e-6, 1e-4, 1e-2]

    # Run experiments for each initial condition
    for init_name, initial_params in initial_conditions.items():
        print(f"\nAnalyzing {init_name}...\n")
        results = {}
        mse_results = {}

        for lr in learning_rates:
            for tol in tolerances:
                label = f"LR = {lr}, Tol = {tol}"
                print(f"Experiment with {label}")

                # Perform gradient descent
                params_list, mse_list = gradient_descent(
                    data_vectors=X,
                    pattern_classes=y_true,
                    initial_params=initial_params,
                    learning_rate=lr,
                    max_steps=1000,
                    tolerance=tol
                )

                # Store results
                results[label] = {
                    'params': params_list[-1],
                    'mse_list': mse_list
                }
                mse_results[label] = mse_list

                # Print final MSE and steps
                print(f"Final MSE: {mse_list[-1]:.6f}")
                print(f"Total Steps: {len(mse_list)}")

        # Plotting Decision Boundaries for this initial condition
        num_plots = len(results)
        cols_db = 3  # Number of columns for decision boundary plots
        rows_db = (num_plots + cols_db - 1) // cols_db  # Ceiling division

        fig_db, axes_db = plt.subplots(rows_db, cols_db, figsize=(5 * cols_db, 4 * rows_db))
        axes_db = axes_db.flatten()

        for idx, (label, result) in enumerate(results.items()):
            ax = axes_db[idx]
            params = result['params']
            w, b = params

            # Plot decision boundary
            plot_decision_boundary(ax, filtered_data, w, b, title=label)

        # Remove any unused subplots
        for idx in range(len(results), len(axes_db)):
            fig_db.delaxes(axes_db[idx])

        fig_db.suptitle(f'Decision Boundaries: {init_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Plotting Learning Curves for this initial condition
        fig_lc, ax_lc = plt.subplots(figsize=(10, 8))
        plot_learning_curves(ax_lc, mse_results, title=f'Learning Curves: {init_name}')
        plt.show()


if __name__ == "__main__":
    np.random.seed(10)
    compare_combinations()

