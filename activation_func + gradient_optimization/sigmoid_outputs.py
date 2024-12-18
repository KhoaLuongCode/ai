import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sigmoid import single_layer_nn
from Homework9.linear_boundaries import load_iris_data

# Define classifier parameters
w = np.array([0.163, 0.97])
b = -2.418

def classify_examples(data, w, b):
    examples = []
    for _, row in data.iterrows():
        features = [row['petal_length'], row['petal_width']]
        prob = single_layer_nn(features, w, b)
        classification = 'Virginica' if prob >= 0.5 else 'Versicolor'
        confidence = 'Near Decision Boundary'
        if prob >= 0.7:
            confidence = 'Unambiguous Virginica'
        elif prob <= 0.3:
            confidence = 'Unambiguous Versicolor'
        
        examples.append({
            'petal_length': row['petal_length'],
            'petal_width': row['petal_width'],
            'species': row['species'],
            'classifier_output': prob,
            'classification': classification,
            'confidence': confidence
        })
    return examples

def main():
    filepath = 'irisdata.csv'  # Path to your CSV file
    data = load_iris_data(filepath)

    # Classify all examples in the dataset
    examples = classify_examples(data, w, b)

    # Filter examples based on criteria
    unambiguous_virginica = [ex for ex in examples if ex['confidence'] == 'Unambiguous Virginica']
    unambiguous_versicolor = [ex for ex in examples if ex['confidence'] == 'Unambiguous Versicolor']
    near_boundary = [ex for ex in examples if ex['confidence'] == 'Near Decision Boundary']

    # Let user choose examples to display
    print("Examples classified:")
    print(f"  Unambiguous Virginica: {len(unambiguous_virginica)}")
    print(f"  Unambiguous Versicolor: {len(unambiguous_versicolor)}")
    print(f"  Near Decision Boundary: {len(near_boundary)}")

    # Display selected examples
    selected_category = input("Choose category to display (virginica/versicolor/boundary): ").strip().lower()
    if selected_category == 'virginica':
        selected_examples = unambiguous_virginica
    elif selected_category == 'versicolor':
        selected_examples = unambiguous_versicolor
    elif selected_category == 'boundary':
        selected_examples = near_boundary
    else:
        print("Invalid category. Showing all examples.")
        selected_examples = examples

    # Print the selected examples
    for i, ex in enumerate(selected_examples, 1):
        print(f"Example {i}:")
        print(f"  Petal Length: {ex['petal_length']} cm")
        print(f"  Petal Width: {ex['petal_width']} cm")
        print(f"  Species: {ex['species'].capitalize()}")
        print(f"  Classifier Output: {ex['classifier_output']:.3f}")
        print(f"  Classification: {ex['classification']} ({ex['confidence']})\n")

if __name__ == "__main__":
    main()