import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import logging
import os 

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    try:
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Validate loaded data
        if df.shape[0] == 0:
            logger.error("Loaded dataframe is empty (0 rows)")
            raise ValueError("Dataset contains no rows")
        
        # Log success
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found at: {filepath}")
        raise  # Re-raise to stop the pipeline
        
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error loading file: {e}")
        raise

def clean_rate_column(df):
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Count issues before cleaning (for logging)
    new_count = (df['rate'] == 'NEW').sum()
    dash_count = (df['rate'] == '-').sum()
    nan_count = df['rate'].isnull().sum()
    
    logger.info(f"Found: {new_count} 'NEW', {dash_count} '-', {nan_count} NaN values")
    
    # Step 1: Replace 'NEW' and '-' with NaN
    df['rate'] = df['rate'].replace(['NEW', '-'], np.nan)
    
    # Step 2: Remove ALL whitespace (handles both "3.9/5" and "3.9 /5")
    df['rate'] = df['rate'].str.strip()  # Remove leading/trailing spaces
    
    # Step 3: Remove '/5' suffix
    df['rate'] = df['rate'].str.replace('/5', '', regex=False)
    
    # Step 4: Convert to float
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    
    # Log results
    final_nan_count = df['rate'].isnull().sum()
    logger.info(f"Cleaning complete. Total NaN values: {final_nan_count}")
    logger.info(f"Rate range: {df['rate'].min():.1f} to {df['rate'].max():.1f}")
    logger.info(f"Mean rating: {df['rate'].mean():.2f}")
    
    return df


def clean_cost_column(df):
    # Make a copy
    df = df.copy()
    
    # Store original column name
    original_col = 'approx_cost(for two people)'
    new_col = 'cost_for_two'
    
    # Count missing values before cleaning
    missing_before = df[original_col].isnull().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")
    
    df = df.rename(columns={original_col: new_col})

    # Remove commas
    df[new_col] = df[new_col].str.replace(',', '', regex=False)

    # Convert to numeric
    df[new_col] = pd.to_numeric(df[new_col], errors='coerce')

    # Check for negative values
    negative_count = (df[new_col] < 0).sum()
    if negative_count > 0:
        logger.warning(f"Found {negative_count} negative cost values!")

    # Check for extremely high values (potential data errors)
    very_high = (df[new_col] > 10000).sum()
    if very_high > 0:
        logger.warning(f"Found {very_high} costs > 10,000 (potential outliers)")
    
    # Log results
    missing_after = df[new_col].isnull().sum()
    valid_count = df[new_col].notna().sum()
    logger.info(f"After cleaning - Valid costs: {valid_count}, Missing: {missing_after}")
    logger.info(f"Cost range: {df[new_col].min():.0f} to {df[new_col].max():.0f}")
    logger.info(f"Mean cost: {df[new_col].mean():.0f}")

    return df


def handle_missing_values(df):
    # Make a copy
    df = df.copy()
    
    initial_rows = len(df)
    logger.info(f"Starting with {initial_rows} rows")
    
    # Drop unnecessary columns
    columns_to_drop = ['url', 'address', 'name', 'phone', 'dish_liked', 
                       'reviews_list', 'menu_item']
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    logger.info(f"Dropped {len(existing_cols_to_drop)} unnecessary columns")

    # Drop rows where rate is NaN 
    rows_before = len(df)
    df = df.dropna(subset=['rate'])
    rows_dropped = rows_before - len(df)
    logger.info(f"Dropped {rows_dropped} rows with missing rate")

    # Handle missing values in remaining features
    
    # cost_for_two - impute with median
    if df['cost_for_two'].isnull().sum() > 0:
        missing_count = df['cost_for_two'].isnull().sum()
        median_cost = df['cost_for_two'].median()
        df['cost_for_two'] = df['cost_for_two'].fillna(median_cost)
        logger.info(f"Imputed {missing_count} missing cost values with median: {median_cost:.0f}")

    # cuisines - fill with 'Others' (very few missing)
    if df['cuisines'].isnull().sum() > 0:
        missing_count = df['cuisines'].isnull().sum()
        df['cuisines'] = df['cuisines'].fillna('Others')
        logger.info(f"Filled {missing_count} missing cuisines with 'Others'")

    # rest_type - fill with 'Unknown'
    if df['rest_type'].isnull().sum() > 0:
        missing_count = df['rest_type'].isnull().sum()
        df['rest_type'] = df['rest_type'].fillna('Unknown')
        logger.info(f"Filled {missing_count} missing rest_type with 'Unknown'")

    # location - drop rows (very few missing)
    if df['location'].isnull().sum() > 0:
        rows_before = len(df)
        df = df.dropna(subset=['location'])
        rows_dropped = rows_before - len(df)
        logger.info(f"Dropped {rows_dropped} rows with missing location")

    # Check if any missing values remain
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Warning: {remaining_missing} missing values still remain!")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        logger.info("No missing values remain!")

    # Log final shape and summary
    final_rows = len(df)
    rows_dropped = initial_rows - final_rows
    logger.info(f"Dropped {rows_dropped} rows ({rows_dropped/initial_rows*100:.1f}%)")
    logger.info(f"Final dataset: {final_rows} rows, {len(df.columns)} columns")
    
    return df

def create_target_variable(df, threshold=3.75):
    # Make a copy
    df = df.copy()
    
    # Create binary target: 1 if rating >= 3.75, else 0
    df['is_successful'] = (df['rate'] >= threshold).astype(int)
    
    # Count successful and unsuccessful restaurants
    successful_count = (df['is_successful'] == 1).sum()
    unsuccessful_count = (df['is_successful'] == 0).sum()
    total = len(df)   

    # Log the distribution
    logger.info("Target Variable Created")
    logger.info(f"Successful (rating ≥ 3.75): {successful_count} ({successful_count/total*100:.1f}%)")
    logger.info(f"Unsuccessful (rating < 3.75): {unsuccessful_count} ({unsuccessful_count/total*100:.1f}%)")

    # Check for class imbalance (if >70% or <30%, log warning)
    success_ratio = successful_count / total
    if success_ratio > 0.7:
        logger.warning(f"Class imbalance: {success_ratio*100:.1f}% are successful. Consider stratified sampling.")
    elif success_ratio < 0.3:
        logger.warning(f"Class imbalance: Only {success_ratio*100:.1f}% are successful. Consider oversampling.")
    else:
        logger.info(f"Classes are balanced ({success_ratio*100:.1f}% successful)")

    return df


def encode_binary_features(df):
    df = df.copy()
    
    binary_cols = ['online_order', 'book_table']
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
            if df[col].isnull().sum() > 0:
                logger.warning(f"{col} has unexpected values!")
                print(df[col].value_counts())
    
    return df

def encode_onehot_features(df, max_categories=20, top_n=15):
    df = df.copy()
    initial_columns = len(df.columns)
    
    columns_to_encode = []
    
    # Handle listed_in(type) - dining type
    if 'listed_in(type)' in df.columns:
        unique_count = df['listed_in(type)'].nunique()
        if unique_count <= max_categories:
            columns_to_encode.append('listed_in(type)')
            logger.info(f"Will OHE 'listed_in(type)' ({unique_count} categories)")
        else:
            logger.warning(f"Skipping 'listed_in(type)' - too many categories ({unique_count})")
    
    # Handle rest_type - restaurant type (group rare categories first)
    if 'rest_type' in df.columns:
        unique_count = df['rest_type'].nunique()
        
        if unique_count > max_categories:
            # Group rare categories
            logger.info(f"'rest_type' has {unique_count} categories - grouping rare ones:")
            
            # Get top N categories
            top_categories = df['rest_type'].value_counts().head(top_n).index.tolist()
            
            # Count how many will be grouped
            other_count = (~df['rest_type'].isin(top_categories)).sum()
            
            # Replace rare categories with 'Other'
            df['rest_type'] = df['rest_type'].apply(lambda x: x if x in top_categories else 'Other')
            
            unique_after = df['rest_type'].nunique()
            logger.info(f"Grouped {other_count} restaurants into 'Other' category")
            logger.info(f"Reduced to {unique_after} categories: {top_n} common types + Other")
            
            columns_to_encode.append('rest_type')
        else:
            columns_to_encode.append('rest_type')
            logger.info(f"Will OHE 'rest_type' ({unique_count} categories)")
    
    # Perform one-hot encoding
    if columns_to_encode:
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True, dtype=int)
        
        final_columns = len(df.columns)
        new_columns = final_columns - initial_columns + len(columns_to_encode)
        
        logger.info(f"One-hot encoding complete: created {new_columns} new binary columns")
        logger.info(f"Total columns now: {final_columns}")
    else:
        logger.info("No columns to one-hot encode")
    
    return df

def encode_location_features(df):
    df = df.copy()
    
    # Frequency encode location
    if 'location' in df.columns:
        df['location_freq'] = df['location'].map(df['location'].value_counts())
        df = df.drop(columns=['location'])
        logger.info(f"Frequency encoded 'location' to 'location_freq'")
    
    # Frequency encode listed_in(city)
    if 'listed_in(city)' in df.columns:
        df['city_freq'] = df['listed_in(city)'].map(df['listed_in(city)'].value_counts())
        df = df.drop(columns=['listed_in(city)'])
        logger.info(f"Frequency encoded 'listed_in(city)' → 'city_freq'")
    
    return df

def encode_cuisine_features(df, top_n=20):
    logger.info(f"Encoding cuisines (top {top_n} cuisines).")
    df = df.copy()
    
    if 'cuisines' not in df.columns:
        return df
    
    # 1: Count number of cuisines
    df['cuisine_count'] = df['cuisines'].str.split(',').str.len()
    
    # 2: Multi-label binarizer for top N cuisines
    # Split cuisines and get top N
    all_cuisines = df['cuisines'].str.split(',').explode().str.strip()
    top_cuisines = all_cuisines.value_counts().head(top_n).index.tolist()
    
    # Create binary columns for top cuisines
    for cuisine in top_cuisines:
        col_name = f'cuisine_{cuisine.replace(" ", "_").replace(",", "").lower()}'
        df[col_name] = df['cuisines'].str.contains(cuisine, case=False, na=False).astype(int)
    
    # Drop original cuisines column
    df = df.drop(columns=['cuisines'])
    
    logger.info(f"Created {len(top_cuisines)} cuisine binary features + cuisine_count")
    return df


def preprocess_pipeline(filepath, save_path='data/preprocessed/'):

    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    # Create save directory if doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Step 1: Load data
    df = load_data(filepath)
    
    # Step 2: Clean numeric columns
    df = clean_rate_column(df)
    df = clean_cost_column(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Create target variable
    df = create_target_variable(df)
    
    # Step 5: Encode categorical features
    df = encode_binary_features(df)
    df = encode_onehot_features(df)
    df = encode_location_features(df)
    df = encode_cuisine_features(df)
    
    # Step 6: Separate features and target
    X = df.drop(columns=['rate', 'is_successful'])  
    y = df['is_successful']
    
    # Step 7: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    # Step 8: Save preprocessed data
    with open(os.path.join(save_path, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(save_path, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(save_path, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(save_path, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    with open(os.path.join(save_path, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    logger.info(f"Saved preprocessed data to {save_path}")
    logger.info("PREPROCESSING COMPLETE!")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_pipeline(
        filepath=r"../data/raw/zomato.csv",
        save_path=r"../data/preprocessed"
    )
