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
    Uses TF-IDF and Cosine Similarity to find top K candidates.
    """
    # Fit TF-IDF on candidate text to build vocabulary
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    cand_tfidf = vectorizer.fit_transform(cand_text)
    s1_tfidf = vectorizer.transform(s1_text)
    
    # Use Nearest Neighbors with cosine distance
    nn = NearestNeighbors(n_neighbors=min(k, len(cand_ids)), metric='cosine', n_jobs=-1)
    nn.fit(cand_tfidf)
    
    distances, indices = nn.kneighbors(s1_tfidf)
    
    # Map indices back to candidate entity_ids
    candidate_lists = []
    for row_indices in indices:
        candidate_lists.append([cand_ids.iloc[i] for i in row_indices])
        
    return candidate_lists

def main():
    DATA_DIR = "../../../dataset/train/"
    OUTPUT_DIR = "../../../output/"
    
    print("Loading data...")
    s1 = load_and_preprocess(os.path.join(DATA_DIR, "train_source1.tsv"))
    s2 = load_and_preprocess(os.path.join(DATA_DIR, "train_source2.tsv"))
    s3 = load_and_preprocess(os.path.join(DATA_DIR, "train_source3.tsv"))
    
    # Combine name and address for a richer text representation
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
        # Combine S2 and S3 candidates
        all_cands = s1_s2_cands[i] + s1_s3_cands[i]
        # Remove duplicates just in case and join with commas
        cand_str = ",".join(list(set(all_cands)))
        candidate_results.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})
        
    cand_df = pd.DataFrame(candidate_results)
    
    # Save exact format required for submission
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"Candidate pairs saved to {cand_path}")
    print(cand_df.head())

if __name__ == "__main__":
    main()