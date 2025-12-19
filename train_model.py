"""
Simple Gradient Boosting Model for Allstate Claims Severity
This is a starting point - add feature engineering as needed.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load training and test data."""
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    return train, test


def preprocess_data(train, test):
    """
    Basic preprocessing - label encoding for categoricals.
    This is where you'll add feature engineering later.
    """
    print("\nPreprocessing data...")
    
    # Separate features and target
    X_train = train.drop(['id', 'loss'], axis=1)
    y_train = train['loss']
    X_test = test.drop('id', axis=1)
    test_ids = test['id']
    
    # Identify categorical and continuous features
    cat_features = [col for col in X_train.columns if col.startswith('cat')]
    cont_features = [col for col in X_train.columns if col.startswith('cont')]
    
    print(f"Categorical features: {len(cat_features)}")
    print(f"Continuous features: {len(cont_features)}")
    
    # Label encode categorical features
    label_encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        # Fit on combined train + test to handle unseen categories
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    print("Label encoding complete.")
    return X_train, y_train, X_test, test_ids


def train_model(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting model with simple hyperparameters."""
    print("\nTraining model...")
    
    # Simple Gradient Boosting parameters - start basic, tune later
    model = GradientBoostingRegressor(
        n_estimators=100,  # Start small, increase later
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        verbose=1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set."""
    print("\nEvaluating model...")
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.4f}")
    return mae


def make_predictions(model, X_test):
    """Make predictions on test set."""
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test)
    return predictions


def save_submission(test_ids, predictions, filename='submission.csv'):
    """Save predictions in submission format."""
    submission = pd.DataFrame({
        'id': test_ids,
        'loss': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"\nSubmission saved to {filename}")
    return submission


def main():
    """Main training pipeline."""
    # Load data
    train, test = load_data()
    
    # Preprocess
    X_train, y_train, X_test, test_ids = preprocess_data(train, test)
    
    # Train/validation split
    print("\nSplitting data...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"Train set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train model
    model = train_model(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate
    val_mae = evaluate_model(model, X_val, y_val)
    
    # Make predictions
    predictions = make_predictions(model, X_test)
    
    # Save submission
    submission = save_submission(test_ids, predictions)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Validation MAE: {val_mae:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()

