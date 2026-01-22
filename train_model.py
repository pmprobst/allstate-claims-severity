"""
Improved Gradient Boosting Model for Allstate Claims Severity
Includes early stopping, better hyperparameters, and basic feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def preprocess_data(train, test, use_log_transform=True):
    """
    Enhanced preprocessing with comprehensive feature engineering.
    - Label encoding for categoricals
    - Frequency encoding for categoricals
    - Target encoding (mean/median/std per category) with smoothing
    - Continuous feature interactions (ratios, products, sums, differences)
    - Statistical aggregations across continuous features
    - Categorical-continuous interactions
    - Option for log transformation of target
    """
    print("\nPreprocessing data...")
    
    # Separate features and target
    X_train = train.drop(['id', 'loss'], axis=1).copy()
    y_train = train['loss'].copy()
    X_test = test.drop('id', axis=1).copy()
    test_ids = test['id']
    
    # Identify categorical and continuous features
    cat_features = [col for col in X_train.columns if col.startswith('cat')]
    cont_features = [col for col in X_train.columns if col.startswith('cont')]
    
    print(f"Categorical features: {len(cat_features)}")
    print(f"Continuous features: {len(cont_features)}")
    
    # Store original categorical values for target encoding (before label encoding)
    cat_train_orig = X_train[cat_features].copy()
    cat_test_orig = X_test[cat_features].copy()
    
    # Label encode categorical features
    print("Label encoding categorical features...")
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
    
    # Feature Engineering: Frequency encoding for categoricals
    print("Adding frequency encoding features...")
    for col in cat_features:
        # Calculate frequency on training set only to avoid data leakage
        freq_map = X_train[col].value_counts().to_dict()
        X_train[f'{col}_freq'] = X_train[col].map(freq_map)
        X_test[f'{col}_freq'] = X_test[col].map(freq_map).fillna(0)  # Fill unseen with 0
    
    # 1. TARGET ENCODING (Mean/Median/Std per category)
    print("\nAdding target encoding features (mean/median/std)...")
    global_mean = y_train.mean()
    global_std = y_train.std()
    smoothing_factor = 10  # Additive smoothing to reduce overfitting
    
    for col in cat_features:
        # Compute statistics on training set only
        target_stats = y_train.groupby(cat_train_orig[col]).agg(['mean', 'median', 'std']).to_dict()
        target_counts = cat_train_orig[col].value_counts().to_dict()
        
        # Mean encoding with smoothing
        mean_map = {}
        median_map = {}
        std_map = {}
        
        for cat_val in target_stats['mean'].keys():
            count = target_counts.get(cat_val, 0)
            # Bayesian averaging: weighted average between category mean and global mean
            weight = count / (count + smoothing_factor)
            mean_map[cat_val] = weight * target_stats['mean'][cat_val] + (1 - weight) * global_mean
            median_map[cat_val] = target_stats['median'][cat_val]
            std_val = target_stats['std'][cat_val]
            std_map[cat_val] = std_val if not np.isnan(std_val) else global_std
        
        # Apply to train and test
        X_train[f'{col}_target_mean'] = cat_train_orig[col].map(mean_map).fillna(global_mean)
        X_test[f'{col}_target_mean'] = cat_test_orig[col].map(mean_map).fillna(global_mean)
        
        X_train[f'{col}_target_median'] = cat_train_orig[col].map(median_map).fillna(y_train.median())
        X_test[f'{col}_target_median'] = cat_test_orig[col].map(median_map).fillna(y_train.median())
        
        X_train[f'{col}_target_std'] = cat_train_orig[col].map(std_map).fillna(global_std)
        X_test[f'{col}_target_std'] = cat_test_orig[col].map(std_map).fillna(global_std)
    
    print(f"  Added {len(cat_features) * 3} target encoding features")
    
    # 2. CONTINUOUS FEATURE INTERACTIONS
    print("\nAdding continuous feature interactions...")
    n_cont = len(cont_features)
    interaction_count = 0
    
    # Ratios (avoid division by zero)
    for i in range(n_cont):
        for j in range(i + 1, n_cont):
            col1, col2 = cont_features[i], cont_features[j]
            # Ratio 1/2
            X_train[f'{col1}_div_{col2}'] = X_train[col1] / (X_train[col2] + 1e-8)
            X_test[f'{col1}_div_{col2}'] = X_test[col1] / (X_test[col2] + 1e-8)
            # Ratio 2/1
            X_train[f'{col2}_div_{col1}'] = X_train[col2] / (X_train[col1] + 1e-8)
            X_test[f'{col2}_div_{col1}'] = X_test[col2] / (X_test[col1] + 1e-8)
            interaction_count += 2
    
    # Products
    for i in range(n_cont):
        for j in range(i + 1, n_cont):
            col1, col2 = cont_features[i], cont_features[j]
            X_train[f'{col1}_mul_{col2}'] = X_train[col1] * X_train[col2]
            X_test[f'{col1}_mul_{col2}'] = X_test[col1] * X_test[col2]
            interaction_count += 1
    
    # Differences
    for i in range(n_cont):
        for j in range(i + 1, n_cont):
            col1, col2 = cont_features[i], cont_features[j]
            X_train[f'{col1}_sub_{col2}'] = X_train[col1] - X_train[col2]
            X_test[f'{col1}_sub_{col2}'] = X_test[col1] - X_test[col2]
            interaction_count += 1
    
    # Sum of all continuous features
    X_train['cont_sum'] = X_train[cont_features].sum(axis=1)
    X_test['cont_sum'] = X_test[cont_features].sum(axis=1)
    interaction_count += 1
    
    print(f"  Added {interaction_count} continuous interaction features")
    
    # 3. STATISTICAL AGGREGATIONS ACROSS CONTINUOUS FEATURES
    print("\nAdding statistical aggregations across continuous features...")
    
    # Row-wise statistics
    X_train['cont_mean'] = X_train[cont_features].mean(axis=1)
    X_test['cont_mean'] = X_test[cont_features].mean(axis=1)
    
    X_train['cont_median'] = X_train[cont_features].median(axis=1)
    X_test['cont_median'] = X_test[cont_features].median(axis=1)
    
    X_train['cont_std'] = X_train[cont_features].std(axis=1)
    X_test['cont_std'] = X_test[cont_features].std(axis=1)
    
    X_train['cont_min'] = X_train[cont_features].min(axis=1)
    X_test['cont_min'] = X_test[cont_features].min(axis=1)
    
    X_train['cont_max'] = X_train[cont_features].max(axis=1)
    X_test['cont_max'] = X_test[cont_features].max(axis=1)
    
    X_train['cont_range'] = X_train['cont_max'] - X_train['cont_min']
    X_test['cont_range'] = X_test['cont_max'] - X_test['cont_min']
    
    # Count features above/below thresholds (using training set statistics)
    cont_mean_threshold = X_train[cont_features].mean().mean()
    cont_median_threshold = X_train[cont_features].median().median()
    
    X_train['cont_above_mean_count'] = (X_train[cont_features] > cont_mean_threshold).sum(axis=1)
    X_test['cont_above_mean_count'] = (X_test[cont_features] > cont_mean_threshold).sum(axis=1)
    
    X_train['cont_above_median_count'] = (X_train[cont_features] > cont_median_threshold).sum(axis=1)
    X_test['cont_above_median_count'] = (X_test[cont_features] > cont_median_threshold).sum(axis=1)
    
    print(f"  Added 8 statistical aggregation features")
    
    # 4. CATEGORICAL-CONTINUOUS INTERACTIONS
    print("\nAdding categorical-continuous interactions...")
    # Limit to top 20 categorical features by variance to avoid feature explosion
    # Use frequency variance as proxy for importance
    cat_variances = {}
    for col in cat_features:
        cat_variances[col] = X_train[f'{col}_freq'].var()
    
    top_cats = sorted(cat_variances.items(), key=lambda x: x[1], reverse=True)[:20]
    top_cat_features = [col for col, _ in top_cats]
    
    interaction_count = 0
    for cat_col in top_cat_features:
        for cont_col in cont_features:
            # Group by categorical value and compute stats of continuous feature
            grouped_stats = X_train.groupby(cat_train_orig[cat_col])[cont_col].agg(['mean', 'median', 'std']).to_dict()
            
            # Mean
            mean_map = grouped_stats.get('mean', {})
            X_train[f'{cat_col}_{cont_col}_mean'] = cat_train_orig[cat_col].map(mean_map).fillna(X_train[cont_col].mean())
            X_test[f'{cat_col}_{cont_col}_mean'] = cat_test_orig[cat_col].map(mean_map).fillna(X_train[cont_col].mean())
            
            # Median
            median_map = grouped_stats.get('median', {})
            X_train[f'{cat_col}_{cont_col}_median'] = cat_train_orig[cat_col].map(median_map).fillna(X_train[cont_col].median())
            X_test[f'{cat_col}_{cont_col}_median'] = cat_test_orig[cat_col].map(median_map).fillna(X_train[cont_col].median())
            
            # Std
            std_map = grouped_stats.get('std', {})
            X_train[f'{cat_col}_{cont_col}_std'] = cat_train_orig[cat_col].map(std_map).fillna(X_train[cont_col].std())
            X_test[f'{cat_col}_{cont_col}_std'] = cat_test_orig[cat_col].map(std_map).fillna(X_train[cont_col].std())
            
            interaction_count += 3
    
    print(f"  Added {interaction_count} categorical-continuous interaction features (top 20 cats)")
    
    print(f"\nTotal features after engineering: {X_train.shape[1]}")
    
    # Log transform target if requested (helps with skewed distribution)
    if use_log_transform:
        print("Applying log transformation to target...")
        y_train = np.log1p(y_train)  # log(1 + x) to handle zeros
    
    return X_train, y_train, X_test, test_ids, use_log_transform


def train_model(X_train, y_train, X_val, y_val, use_log_transform=True):
    """
    Train Gradient Boosting model with improved hyperparameters and early stopping.
    """
    print("\nTraining model...")
    
    # Improved hyperparameters
    n_estimators = 500  # More trees with early stopping
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=7,  # Slightly deeper
        learning_rate=0.05,  # Lower learning rate for better generalization
        subsample=0.85,  # Row sampling
        max_features='sqrt',  # Column sampling
        min_samples_split=10,  # Prevent overfitting
        min_samples_leaf=5,  # Prevent overfitting
        random_state=42,
        verbose=1
        # Note: We handle validation manually using staged_predict
    )
    
    # Train with early stopping using staged_predict
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Find best number of estimators using staged predictions
    print("Finding optimal number of estimators...")
    best_val_mae = float('inf')
    best_n_estimators = 50
    patience = 30
    no_improve_count = 0
    
    staged_predictions = list(model.staged_predict(X_val))
    
    for i, y_val_pred in enumerate(staged_predictions, start=1):
        if use_log_transform:
            y_val_pred = np.expm1(y_val_pred)  # Transform back from log space
            y_val_actual = np.expm1(y_val)  # y_val is in log space, transform back
        else:
            y_val_actual = y_val
        
        val_mae = mean_absolute_error(y_val_actual, y_val_pred)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_n_estimators = i
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if i % 50 == 0 or i <= 100:
            print(f"  Iteration {i}: Val MAE = {val_mae:.4f} (Best: {best_val_mae:.4f} at {best_n_estimators})")
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at iteration {i}")
            break
    
    print(f"\nBest validation MAE: {best_val_mae:.4f} at {best_n_estimators} estimators")
    
    # Retrain with optimal number of estimators
    if best_n_estimators < n_estimators:
        print(f"Retraining with {best_n_estimators} estimators...")
        model.set_params(n_estimators=best_n_estimators)
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_val, y_val, use_log_transform=True):
    """Evaluate model on validation set with detailed metrics."""
    print("\nEvaluating model...")
    y_pred = model.predict(X_val)
    
    # Transform back from log space if needed
    if use_log_transform:
        y_pred = np.expm1(y_pred)
        y_val_actual = np.expm1(y_val)
    else:
        y_val_actual = y_val
    
    mae = mean_absolute_error(y_val_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred))
    
    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Mean target:     {y_val_actual.mean():.2f}")
    print(f"Std target:      {y_val_actual.std():.2f}")
    
    return mae


def make_predictions(model, X_test, use_log_transform=True):
    """Make predictions on test set."""
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test)
    
    # Transform back from log space if needed
    if use_log_transform:
        predictions = np.expm1(predictions)
    
    # Ensure no negative predictions
    predictions = np.maximum(predictions, 0)
    
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
    # Configuration
    USE_LOG_TRANSFORM = True  # Set to False to disable log transformation
    
    # Load data
    train, test = load_data()
    
    # Preprocess
    X_train, y_train, X_test, test_ids, use_log_transform = preprocess_data(
        train, test, use_log_transform=USE_LOG_TRANSFORM
    )
    
    # Train/validation split
    print("\nSplitting data...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"Train set: {X_train_split.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train model
    model = train_model(
        X_train_split, y_train_split, X_val, y_val,
        use_log_transform=use_log_transform
    )
    
    # Evaluate
    val_mae = evaluate_model(model, X_val, y_val, use_log_transform=use_log_transform)
    
    # Make predictions
    predictions = make_predictions(model, X_test, use_log_transform=use_log_transform)
    
    # Save submission
    submission = save_submission(test_ids, predictions)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Validation MAE: {val_mae:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()

