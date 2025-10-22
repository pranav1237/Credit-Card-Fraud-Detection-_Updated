import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'data', 'notebooks', 'src')
sys.path.insert(0, src_path)

# Now import the modules
from train_model import train
from predict import predict
import pandas as pd

if __name__ == "__main__":
    print("Starting Credit Card Fraud Detection...")
    
    # Train the model
    print("Training the model...")
    train()
    
    # Example prediction
    print("Making a prediction...")
    sample = pd.read_csv('data/creditcard.csv').drop(['Class', 'Time'], axis=1).iloc[[0]]
    result = predict(sample)
    print("Prediction:", "Fraud" if result[0] == 1 else "Legit")
    