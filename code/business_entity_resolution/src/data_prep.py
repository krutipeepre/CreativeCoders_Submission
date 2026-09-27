import pandas as pd
import os

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep="\t")
    cols_to_normalize = ['business_name', 'business_address', 'country']
    for col in cols_to_normalize:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
    return df

def main():
    # 1. Exact script ki location nikalo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Project ke root folder tak jao
    root_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
    
    # 3. Clean structure wala path try karo
    DATA_DIR = os.path.join(root_dir, "dataset", "train")
    
    # 4. Agar folders theek se move nahi hue the, toh purana path try karo
    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.join(root_dir, "DataSet", "student_resource", "dataset", "train")
        
    print(f"Loading data from: {DATA_DIR}")
    
    s1_train = load_and_preprocess(os.path.join(DATA_DIR, "train_source1.tsv"))
    s2_train = load_and_preprocess(os.path.join(DATA_DIR, "train_source2.tsv"))
    s3_train = load_and_preprocess(os.path.join(DATA_DIR, "train_source3.tsv"))
    
    print("\nData loaded successfully!")
    print(f"Source 1 Shape: {s1_train.shape}")
    print("\nSample Data from Source 1:")
    print(s1_train.head(2))

if __name__ == "__main__":
    main()