import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep="\t")
    cols = ['business_name', 'business_address', 'country']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.lower().str.strip()
    return df

def get_top_k_candidates(s1_text, cand_text, cand_ids, k=5):
    """
    Uses TF-IDF on character n-grams and Cosine Similarity to find top K candidates.
    """
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    cand_tfidf = vectorizer.fit_transform(cand_text)
    s1_tfidf = vectorizer.transform(s1_text)
    
    nn = NearestNeighbors(n_neighbors=min(k, len(cand_ids)), metric='cosine', n_jobs=-1)
    nn.fit(cand_tfidf)
    distances, indices = nn.kneighbors(s1_tfidf)
    
    candidate_lists = []
    for row_indices in indices:
        candidate_lists.append([cand_ids.iloc[i] for i in row_indices])
    return candidate_lists

def main():
    # Robust path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
    
    DATA_DIR = os.path.join(root_dir, "dataset", "train")
    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.join(root_dir, "DataSet", "student_resource", "dataset", "train")
        
    OUTPUT_DIR = os.path.join(root_dir, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data for blocking...")
    s1 = load_and_preprocess(os.path.join(DATA_DIR, "train_source1.tsv"))
    s2 = load_and_preprocess(os.path.join(DATA_DIR, "train_source2.tsv"))
    s3 = load_and_preprocess(os.path.join(DATA_DIR, "train_source3.tsv"))
    
    # Combine name and address for richer context
    s1_text = s1['business_name'] + " " + s1['business_address']
    s2_text = s2['business_name'] + " " + s2['business_address']
    s3_text = s3['business_name'] + " " + s3['business_address']
    
    print("Generating candidates from Source 2...")
    s1_s2_cands = get_top_k_candidates(s1_text, s2_text, s2['entity_id'], k=5)
    
    print("Generating candidates from Source 3...")
    s1_s3_cands = get_top_k_candidates(s1_text, s3_text, s3['entity_id'], k=5)
    
    print("Formatting candidate pairs...")
    candidate_results = []
    for i, s1_id in enumerate(s1['entity_id']):
        all_cands = s1_s2_cands[i] + s1_s3_cands[i]
        # Remove duplicates and join with comma
        cand_str = ",".join(list(set(all_cands)))
        candidate_results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})
        
    cand_df = pd.DataFrame(candidate_results)
    
    # Save the output exactly as required
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    cand_df.to_csv(cand_path, sep="\t", index=False)
    
    print(f"\nSuccess! Candidate pairs saved to: {cand_path}")
    print(cand_df.head())

if __name__ == "__main__":
    main()