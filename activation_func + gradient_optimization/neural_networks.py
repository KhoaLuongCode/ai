import numpy as np
import pandas as pd
from sigmoid import single_layer_nn  # Ensure this import matches your file structure
from linear_boundaries import load_iris_data
import matplotlib.pyplot as plt

def calculate_mse(data_vectors, nn_params, pattern_classes):
    # Unpack neural network parameters
    w, b = nn_params

    # Compute neural network outputs
    y_pred = single_layer_nn(data_vectors, w, b)

    # Compute MSE
    mse = np.mean((y_pred - pattern_classes) ** 2)

    return mse

def plot_decision_boundary(ax, data, nn_params, label):
    # Define the species to plot
    target_species = ['versicolor', 'virginica']

    # Filter data for the target species
    filtered_data = data[data['species'].isin(target_species)]

    # Extract features and labels
    petal_length = filtered_data['petal_length'].values
    petal_width = filtered_data['petal_width'].values
    species = filtered_data['species'].values

    # Map species to binary labels: 0 for Versicolor, 1 for Virginica
    species_label = np.where(species == 'versicolor', 0, 1)

    # Define colors for each species
    colors = {'versicolor': 'blue', 'virginica': 'green'}

    # Create the scatter plot
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

    # Unpack neural network parameters
    w, b = nn_params

    # Compute the output of the neural network for each grid point
    Z = single_layer_nn(grid_points, w, b)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    contour = ax.contour(
        xx,
        yy,
        Z,
        levels=[0.5],
        colors='red',
        linewidths=2,
        linestyles='-',
        label=label
    )

    # Adding labels and title
    ax.set_xlabel('Petal Length (cm)', fontsize=12)
    ax.set_ylabel('Petal Width (cm)', fontsize=12)
    ax.set_title('Decision Boundary After Gradient Steps ', fontsize=14)
    ax.legend(title='Species', fontsize=10)
    ax.grid(True)

def plot_learning_curve(ax, mse_list):
    ax.plot(range(1, len(mse_list) + 1), mse_list, marker='o', linestyle='-')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_title('Learning Curve (PART 3B)', fontsize=14)
    ax.grid(True)

def plot_multiple_decision_boundaries(data, params_list, labels_list, title):
    # Define the species to plot
    target_species = ['versicolor', 'virginica']

    # Filter data for the target species
    filtered_data = data[data['species'].isin(target_species)]

    # Extract features and labels
    petal_length = filtered_data['petal_length'].values
    petal_width = filtered_data['petal_width'].values
    species = filtered_data['species'].values

    # Map species to binary labels: 0 for Versicolor, 1 for Virginica
    species_label = np.where(species == 'versicolor', 0, 1)

    # Define colors for each species
    colors = {'versicolor': 'blue', 'virginica': 'green'}

    # Create the scatter plot
    plt.figure(figsize=(12, 10))

    for sp in target_species:
        sp_data = filtered_data[filtered_data['species'] == sp]
        plt.scatter(
            sp_data['petal_length'],
            sp_data['petal_width'],
            label=sp.capitalize(),
            c=colors[sp],
            edgecolor='k',
            s=60,
            marker='o'
        )

    # Generate a grid of points for plotting the decision boundaries
    x_min, x_max = petal_length.min() - 0.5, petal_length.max() + 0.5
    y_min, y_max = petal_width.min() - 0.5, petal_width.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Colors and linestyles for different boundaries
    colors_boundary = plt.cm.viridis(np.linspace(0, 1, len(params_list)))
    linestyles = ['-', '--', '-.', ':'] * (len(params_list) // 4 + 1)

    # Plot each decision boundary
    for idx, (params, label) in enumerate(zip(params_list, labels_list)):
        w, b = params
        # Compute the output of the neural network for each grid point
        Z = single_layer_nn(grid_points, w, b)
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        contour = plt.contour(
            xx,
            yy,
            Z,
            levels=[0.5],
            colors=[colors_boundary[idx]],
            linewidths=2,
            linestyles=linestyles[idx % len(linestyles)],
            label=label
        )

    # Adding labels and title
    plt.xlabel('Petal Length (cm)', fontsize=14)
    plt.ylabel('Petal Width (cm)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(title='species', fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()


def compute_gradient(data_vectors, nn_params, pattern_classes):
    w, b = nn_params
    y_pred = single_layer_nn(data_vectors, w, b)
    error = y_pred - pattern_classes  # Shape: (n_samples,)

    # Compute the derivative of the sigmoid function
    sigmoid_derivative = y_pred * (1 - y_pred)  # Shape: (n_samples,)

    # Compute gradients
    grad_loss = 2 * error * sigmoid_derivative  # Shape: (n_samples,)
    grad_w = np.dot(data_vectors.T, grad_loss) / data_vectors.shape[0]
    grad_b = np.sum(grad_loss) / data_vectors.shape[0]

    return grad_w, grad_b

def gradient_descent(data_vectors, pattern_classes, initial_params, learning_rate, max_steps, tolerance):
    w, b = initial_params
    nn_params = (w, b)
    mse_list = []
    params_list = [nn_params]

    for step in range(max_steps):
        # Compute MSE before update
        mse = calculate_mse(data_vectors, nn_params, pattern_classes)
        mse_list.append(mse)

        # Compute gradients
        grad_w, grad_b = compute_gradient(data_vectors, nn_params, pattern_classes)

        # Update weights and bias using learning rate
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        nn_params = (w.copy(), b)

        # Save updated parameters
        params_list.append(nn_params)

        # Check for convergence
        if len(mse_list) > 1 and abs(mse_list[-1] - mse_list[-2]) < tolerance:
            print(f"Converged after {step + 1} steps.")
            break

    return params_list, mse_list

def main_with_gradient_descent():
    filepath = 'irisdata.csv'  # Path to your CSV file
    data = load_iris_data(filepath)

    # Filter data for Versicolor and Virginica
    target_species = ['versicolor', 'virginica']
    filtered_data = data[data['species'].isin(target_species)]
    X = filtered_data[['petal_length', 'petal_width']].values
    y_true = np.where(filtered_data['species'] == 'versicolor', 0, 1)

    # Initialize neural network parameters (Poor Fit)
    w_initial = np.array([0.163, 0.97], dtype=float)
    b_initial = -2.418
    initial_params = (w_initial, b_initial)

    # Define hyperparameters for gradient descent
    learning_rate = 0.01
    max_steps = 1000
    tolerance = 1e-6

    # Perform gradient descent
    params_list, mse_list = gradient_descent(
        data_vectors=X,
        pattern_classes=y_true,
        initial_params=initial_params,
        learning_rate=learning_rate,
        max_steps=max_steps,
        tolerance=tolerance
    )

    # Select steps to plot (e.g., initial, steps 1, 5, 10, final)
    steps_to_plot = [0, 1, 5, 10, len(params_list) - 1]
    steps_to_plot = [s for s in steps_to_plot if s < len(params_list)]  # Ensure steps are within range

    # Prepare parameters and labels for the selected steps
    params_plot = [params_list[s] for s in steps_to_plot]
    labels_plot = [f'Step {s}' if s != 0 else 'Initial' for s in steps_to_plot]

    # Plot Decision Boundary for each selected step
    for params, label in zip(params_plot, labels_plot):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_decision_boundary(ax, data, params, label=label)
        ax.set_title(f'Decision Boundary at {label} (PART 3B)', fontsize=16)
        plt.show()

    # Plot Learning Curve
    plt.figure(figsize=(8, 6))
    plot_learning_curve(plt.gca(), mse_list)
    plt.show()


def main():
    filepath = 'irisdata.csv'  # Path to your CSV file
    data = load_iris_data(filepath)

    # Filter data for Versicolor and Virginica
    target_species = ['versicolor', 'virginica']
    filtered_data = data[data['species'].isin(target_species)]
    X = filtered_data[['petal_length', 'petal_width']].values
    y_true = np.where(filtered_data['species'] == 'versicolor', 0, 1)

    # Define two different sets of neural network parameters
    # First set: Good fit (small error)
    w1 = np.array([0.163, 1])
    b1 = -2.418
    nn_params1 = (w1, b1)

    # Second set: Poor fit (large error)
    w2 = np.array([-1.0, -1.0], dtype=float)
    b2 = -5.0
    nn_params2 = (w2, b2)

    # Compute MSE for both parameter sets
    mse1 = calculate_mse(X, nn_params1, y_true)
    mse2 = calculate_mse(X, nn_params2, y_true)
    print(f"Initial MSE with first parameter set (Good Fit): {mse1:.4f}")
    print(f"Initial MSE with second parameter set (Poor Fit): {mse2:.4f}")

    # Plot both decision boundaries using the existing function
    plot_multiple_decision_boundaries(data, [nn_params1, nn_params2], ['Good Fit', 'Poor Fit'], 'Decision Boundaries for Good and Poor Fit on the Iris Dataset (PART 2B)')

    # Now, perform small gradient steps starting from the poor fit parameters
    # Initialize neural network parameters with poor fit
    w = w1.copy()
    b = b1
    nn_params = (w, b)

    # Define the step size (small change)
    step_size = 0.05  # Adjust as needed for small updates

    # Number of gradient steps
    num_steps = 10

    # Store parameters and labels for plotting
    params_list = [nn_params]
    labels_list = ['Initial Poor Fit']

    print("\nPerforming Gradient Steps with Small Updates:")
    for i in range(num_steps):
        # Compute MSE before updat
        mse = calculate_mse(X, nn_params, y_true)
        print(f"Step {i+1}: MSE = {mse:.4f}")

        # Compute gradients
        grad_w, grad_b = compute_gradient(X, nn_params, y_true)

        # Update weights and bias with small step (without a learning rate)
        w = w - step_size * grad_w
        b = b - step_size * grad_b
        nn_params = (w.copy(), b)

        # Print detailed weight and bias changes
        print(f"Step {i+1}: Weights = {w}, Bias = {b}")
        print(f"Step {i+1}: Gradient W = {grad_w}, Gradient B = {grad_b}\n")

        # Append new parameters and labels for plotting
        params_list.append(nn_params)
        labels_list.append(f'Step {i+1}')

    # Combine all parameter sets: initial poor fit and gradient steps
    all_params = params_list  # Includes initial and all updated parameters
    all_labels = labels_list  # Includes labels for all steps

    # Plot all decision boundaries together
    plot_multiple_decision_boundaries(data, all_params, all_labels, 'Decision Boundaries on Iris Dataset with Gradient Steps (PART 2D)')


if __name__ == "__main__":
    main()
    main_with_gradient_descent()
