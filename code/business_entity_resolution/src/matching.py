import pandas as pd
import os

def get_jaccard_similarity(str1, str2):
    """Calculates word-level Jaccard similarity between two strings."""
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
    
    DATA_DIR = os.path.join(root_dir, "dataset", "train")
    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.join(root_dir, "DataSet", "student_resource", "dataset", "train")
        
    OUTPUT_DIR = os.path.join(root_dir, "output")
    
    print("Loading data for matching...")
    # Read the preprocessed datasets to build a lookup dictionary
    def load_clean(path):
        df = pd.read_csv(path, sep="\t")
        df['text'] = df['business_name'].fillna('').str.lower() + " " + df['business_address'].fillna('').str.lower()
        return df.set_index('entity_id')['text'].to_dict()

    s1_dict = load_clean(os.path.join(DATA_DIR, "train_source1.tsv"))
    s2_dict = load_clean(os.path.join(DATA_DIR, "train_source2.tsv"))
    s3_dict = load_clean(os.path.join(DATA_DIR, "train_source3.tsv"))
    
    # Combine S2 and S3 dicts for easy lookup
    cand_dict = {**s2_dict, **s3_dict}
    
    print("Loading candidate pairs...")
    candidates_df = pd.read_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t")
    
    # We set a strict threshold to favor precision (important for F_0.5 score)
    SIMILARITY_THRESHOLD = 0.55
    
    matching_results = []
    
    print("Filtering candidates based on similarity...")
    for _, row in candidates_df.iterrows():
        s1_id = row['source1_entity_id']
        s1_text = s1_dict.get(s1_id, "")
        
        cands_str = str(row['candidate_entity_ids'])
        if pd.isna(row['candidate_entity_ids']) or cands_str.strip() == "":
            cands = []
        else:
            cands = cands_str.split(',')
            
        final_matches = []
        for cand_id in cands:
            cand_text = cand_dict.get(cand_id, "")
            score = get_jaccard_similarity(s1_text, cand_text)
            
            if score >= SIMILARITY_THRESHOLD:
                final_matches.append(cand_id)
                
        # Empty list is handled correctly (singleton)
        matched_str = ",".join(final_matches)
        matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': matched_str})
        
    final_df = pd.DataFrame(matching_results)
    
    output_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    final_df.to_csv(output_path, sep="\t", index=False)
    
    print(f"\nSuccess! Final matching results saved to: {output_path}")
    print(final_df.head())

if __name__ == "__main__":
    main()