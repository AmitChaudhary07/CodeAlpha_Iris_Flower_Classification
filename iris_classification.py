import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load the Iris dataset
def load_data():
    """Load the Iris dataset from CSV file"""
    try:
        # Load data from CSV file
        data = pd.read_csv('DS/Iris.csv')
        
        print("\nOriginal columns in CSV:", data.columns.tolist())
        
        # Drop the Id column as it's not a feature
        data = data.drop('Id', axis=1)
        
        # Rename columns to match the expected format
        column_mapping = {
            'SepalLengthCm': 'sepal length (cm)',
            'SepalWidthCm': 'sepal width (cm)',
            'PetalLengthCm': 'petal length (cm)',
            'PetalWidthCm': 'petal width (cm)',
            'Species': 'target'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Convert species to numeric targets (0: setosa, 1: versicolor, 2: virginica)
        species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        data['target'] = data['target'].map(species_map)
        
        return data
    except FileNotFoundError:
        print("Error: Iris.csv file not found in the DS directory!")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

# Preprocess the data
def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    # Get species names for better readability
    species_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                              target_names=species_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_names,
                yticklabels=species_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Visualize the data
def visualize_data(data):
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=data, x='sepal length (cm)', y='sepal width (cm)', 
                   hue='target', palette='deep')
    plt.title('Sepal Length vs Width')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=data, x='petal length (cm)', y='petal width (cm)', 
                   hue='target', palette='deep')
    plt.title('Petal Length vs Width')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    print("Loading data...")
    data = load_data()
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(f"Total samples: {len(data)}")
    print("\nSample distribution:")
    species_counts = data.groupby('target').size()
    for target, count in species_counts.items():
        species_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'][int(target)]
        print(f"{species_name}: {count} samples")
    
    # Visualize the data
    print("Visualizing data distributions...")
    visualize_data(data)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Print feature importance with actual feature names
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']
    importances = model.feature_importances_
    
    # Create DataFrame with feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    main() 