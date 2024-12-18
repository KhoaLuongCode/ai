import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sigmoid import single_layer_nn

# Function to load Iris data using pandas
def load_iris_data(filepath='irisdata.csv'):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: File '{filepath}' has incorrect format.")
        sys.exit(1)

def plot_iris(data):
    target_species = ['versicolor', 'virginica']
    filtered_data = data[data['species'].isin(target_species)]

    colors = {'versicolor': 'blue', 'virginica': 'green'}

    plt.figure(figsize=(8, 6))

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

    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.title('Iris Dataset: Versicolor vs Virginica (PART 1A)', fontsize=14)
    plt.legend(title='Species')
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def plot_decision_boundary(data):
    target_species = ['versicolor', 'virginica']

    filtered_data = data[data['species'].isin(target_species)]

    petal_length = filtered_data['petal_length'].values
    petal_width = filtered_data['petal_width'].values
    species = filtered_data['species'].values

    species_label = np.where(species == 'versicolor', 0, 1)

    colors = np.array(['blue', 'green'])
    point_colors = colors[species_label]

    # Manually set weights and bias to roughly separate the two classes
    w = np.array([0.163, 0.97])  
    b = -2.418
                    

    plt.figure(figsize=(8, 6))
    for sp, color in zip(target_species, colors):
        sp_data = filtered_data[filtered_data['species'] == sp]
        plt.scatter(
            sp_data['petal_length'],
            sp_data['petal_width'],
            label=sp.capitalize(),
            c=color,
            edgecolor='k',
            s=60,
            marker='o'
        )

    x_min, x_max = petal_length.min() - 0.5, petal_length.max() + 0.5
    y_min, y_max = petal_width.min() - 0.5, petal_width.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Prepare grid points as input features
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute the output of the neural network for each grid point
    Z = single_layer_nn(grid_points, w, b)
    Z = Z.reshape(xx.shape)

    # Determine the decision boundary (where sigmoid output is 0.5)
    boundary = Z >= 0.5

    # Plot the decision boundary
    plt.contour(xx, yy, boundary, levels=[0.5], colors='red', linewidths=2, linestyles='--')

    # Adding labels and title
    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.title('Decision Boundary on Iris Dataset (PART 1C)', fontsize=14)
    plt.legend(title='Species')
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()
    
# Function to plot the neural network output surface
def plot_nn_output_surface(data):
    # Define the species to plot
    target_species = ['versicolor', 'virginica']

    # Filter data for the target species
    filtered_data = data[data['species'].isin(target_species)]

    # Extract features
    petal_length = filtered_data['petal_length'].values
    print(petal_length)
    petal_width = filtered_data['petal_width'].values
    print(petal_width)

    # Manually set weights and bias (same as in plot_decision_boundary)
    w = np.array([0.164, 0.986])  # Weights for petal length and petal width
    b = -2.414                     # Bias term

    # Generate a grid of points for plotting the surface
    x_min, x_max = petal_length.min() - 0.5, petal_length.max() + 0.5
    y_min, y_max = petal_width.min() - 0.5, petal_width.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    # Prepare grid points as input features
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute the output of the neural network for each grid point
    Z = single_layer_nn(grid_points, w, b)
    Z = Z.reshape(xx.shape)

    # Create a 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)

    # Plot the data points
    # Map species to binary labels: 0 for Versicolor, 1 for Virginica
    species = filtered_data['species'].values
    species_label = np.where(species == 'versicolor', 0, 1)
    colors = np.array(['blue', 'green'])
    point_colors = colors[species_label]

    ax.scatter(petal_length, petal_width, species_label, c=point_colors, edgecolor='k', s=60, depthshade=False)

    # Labels and title
    ax.set_xlabel('Petal Length (cm)', fontsize=12)
    ax.set_ylabel('Petal Width (cm)', fontsize=12)
    ax.set_zlabel('Neural Network Output', fontsize=12)
    ax.set_title('Neural Network Output Surface (PART 1D)', fontsize=14)

    plt.show()




def plot_classifier_output(data):
    target_species = ['versicolor', 'virginica']

    # Filter data for the target species
    filtered_data = data[data['species'].isin(target_species)]

    # Extract features
    petal_length = filtered_data['petal_length'].values
    petal_width = filtered_data['petal_width'].values
    species = filtered_data['species'].values

    # Map species to binary labels: 0 for Versicolor, 1 for Virginica
    species_label = np.where(species == 'versicolor', 0, 1)

    # Define the weights and bias (same as in plot_decision_boundary)
    w = np.array([0.163, 0.97])
    b = -2.418

    # Compute the neural network output for the data points
    X = np.c_[petal_length, petal_width]
    nn_output = single_layer_nn(X, w, b)

    # Determine unambiguous examples (very low or very high nn_output)
    unambiguous_indices = (nn_output < 0.2) | (nn_output > 0.8)
    boundary_indices = (nn_output >= 0.4) & (nn_output <= 0.6)

    # Plot setup
    plt.figure(figsize=(8, 6))

    # Plot unambiguous examples
    plt.scatter(
        petal_length[unambiguous_indices],
        petal_width[unambiguous_indices],
        c='orange',
        label='Unambiguous Examples',
        edgecolor='k',
        s=80,
        marker='o'
    )

    # Plot boundary examples
    plt.scatter(
        petal_length[boundary_indices],
        petal_width[boundary_indices],
        c='purple',
        label='Boundary Examples',
        edgecolor='k',
        s=80,
        marker='x'
    )

    # Plot all data points with species labels
    colors = np.array(['blue', 'green'])
    plt.scatter(
        petal_length,
        petal_width,
        c=colors[species_label],
        label='Data Points (Versicolor & Virginica)',
        edgecolor='k',
        s=60,
        alpha=0.5
    )

    # Labels and title
    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.title('Classifier Output for Selected Examples (PART 1E)', fontsize=14)
    plt.legend(title='Legend')
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()

# Add this call in the main function to execute the new plot
def main():
    filepath = 'irisdata.csv'  # Path to your CSV file
    data = load_iris_data(filepath)
    plot_iris(data)
    plot_decision_boundary(data)
    plot_nn_output_surface(data)
    plot_classifier_output(data)

if __name__ == "__main__":
    main()
