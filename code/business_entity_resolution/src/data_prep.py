import pandas as pd
import os

def load_and_preprocess(file_path):
    """
    Loads TSV data and performs basic text normalization.
    """
    # Strict requirement: Read with tab separator as addresses contain commas
    df = pd.read_csv(file_path, sep="\t")
    
    # Fill missing values and convert text to lowercase for uniform matching
    cols_to_normalize = ['business_name', 'business_address', 'country']
    for col in cols_to_normalize:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
            
    return df

def main():
    # Path updated to point to the cleaned root 'dataset' folder
    DATA_DIR = "../../../dataset/train/"
    
    print("Loading datasets...")
    
    # Load all three sources
    s1_train = load_and_preprocess(os.path.join(DATA_DIR, "train_source1.tsv"))
    s2_train = load_and_preprocess(os.path.join(DATA_DIR, "train_source2.tsv"))
    s3_train = load_and_preprocess(os.path.join(DATA_DIR, "train_source3.tsv"))
    
    # Load ground truth (no need for text normalization here)
    ground_truth = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\t")
    
    print("Data loaded successfully!\n")
    print(f"Source 1 Shape: {s1_train.shape}")
    print(f"Source 2 Shape: {s2_train.shape}")
    print(f"Source 3 Shape: {s3_train.shape}")
    print(f"Ground Truth Shape: {ground_truth.shape}")
    
    print("\nSample Data from Source 1:")
    print(s1_train.head(3))

if __name__ == "__main__":
    main()