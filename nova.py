import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances



"""PHASE 1"""

print("Opening dataset")
df = pd.read_csv("napoleon_bonaparte.csv")

print(f"Number of rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

model_dir = "models/paraphrase-multilingual-mpnet-base-v2"
if not os.path.exists(model_dir):
    print("\nModel downloading...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    model.save(model_dir)
    print("Model downloaded and saved")
else:
    print("\nModel loading...")
    model = SentenceTransformer(model_dir)
    print("Model loaded")

# Check if 'coordinates_question' and 'coordinates_answer' columns exist and are filled
if ('coordinates_question' not in df.columns or df['coordinates_question'].isnull().any() or
    'coordinates_answer' not in df.columns or df['coordinates_answer'].isnull().any()):

    if 'coordinates_question' not in df.columns:
        df['coordinates_question'] = None
    if 'coordinates_answer' not in df.columns:
        df['coordinates_answer'] = None

    print("\nEncoding missing embeddings...")

    for i, row in df.iterrows():
        if pd.isna(row['coordinates_question']) or row['coordinates_question'] == 'None':
            vec = model.encode(str(row['QUESTION']), convert_to_numpy=True, show_progress_bar=False)
            df.at[i, 'coordinates_question'] = json.dumps(vec.tolist())
        if pd.isna(row['coordinates_answer']) or row['coordinates_answer'] == 'None':
            vec = model.encode(str(row['ANSWER']), convert_to_numpy=True, show_progress_bar=False)
            df.at[i, 'coordinates_answer'] = json.dumps(vec.tolist())

    print("Saving updated CSV...")
    df.to_csv("napoleon_bonaparte.csv", index=False)
    print("CSV updated with embeddings.")
else:
    print("Embeddings already present, skipping encoding.")











""" PHASE 2"""

print("\nLoading embeddings into numpy arrays...")
question_vectors = np.vstack(df['coordinates_question'].apply(lambda x: np.array(json.loads(x))).values)
answer_vectors = np.vstack(df['coordinates_answer'].apply(lambda x: np.array(json.loads(x))).values)

while True:
    user_input = str(input("\nYou: "))
    new_point = model.encode(user_input, convert_to_numpy=True, show_progress_bar=False)

    distances = cosine_distances([new_point], question_vectors)[0]
    nearest_idx = np.argsort(distances)[:4]

    selected_answers = answer_vectors[nearest_idx]

    total_internal_distances = []
    for vec in selected_answers:
        dist_sum = np.sum(cosine_distances([vec], selected_answers))
        total_internal_distances.append(dist_sum)

    best_idx_in_4 = np.argmin(total_internal_distances)
    best_idx = nearest_idx[best_idx_in_4]

    print(f"\nBot: {df.iloc[best_idx]['ANSWER']}")
