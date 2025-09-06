"""PHASE 0 : IMPORT THE LIBRARY, GET THE DATASET, INITIALIZE THE MODELS, SETUP THE PARAMETERS"""

print("PHASE 0 : INITIALIZATION")
# Import the library
print("Import the library")
import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


# Get the dataset
print("Opening dataset")
df = pd.read_csv("napoleon_bonaparte.csv")

print(f"Number of rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Initialize the models
model_dir = "models/paraphrase-multilingual-mpnet-base-v2"
if not os.path.exists(model_dir):
    print("\nModel downloading...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    model.save(model_dir)
    print("Model downloaded and saved")
else:
    print("\nModel loading...")
    model = SentenceTransformer(model_dir)
    print("Model loaded")

# Set the parameters
ANSWER_COL = "A"
QUESTION_COL = "Q"

# PROMPT WHICH WILL BE UTILIZED, ADAPT IT WITH THE CONTEXT
PROMPT_TEMPLATE = """
YOU ARE NAPOLEON BONAPARTE.

The user says: "{user_input}"

Below are the 5 most relevant Q&A pairs from your past writings:

{qa_pairs}

TASK:
- Analyze these 5 Q&A pairs.
- Ignore any that are irrelevant or off-topic.
- Write a single, coherent answer to the user's request.
- The answer must logically address the user's question.
- Your style must imitate the tone and manner of speaking found in the relevant Q&A pairs.
- It's a conversation you have to remember what the user told you and what you responded
"""

"""PHASE 1 : MAPPING"""
print("\nPHASE 1 : MAPPING")
# Check if 'coordinates_question' and 'coordinates_answer' columns exist
if (
    "coordinates_question" not in df.columns
    or df["coordinates_question"].isnull().any()
    or "coordinates_answer" not in df.columns
    or df["coordinates_answer"].isnull().any()
):

    if "coordinates_question" not in df.columns:
        df["coordinates_question"] = None
    if "coordinates_answer" not in df.columns:
        df["coordinates_answer"] = None

    print("Encoding missing embeddings...")

    for i, row in df.iterrows():
        if (
            pd.isna(row["coordinates_question"])
            or row["coordinates_question"] == "None"
        ):
            vec = model.encode(
                str(row[QUESTION_COL]),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            df.at[i, "coordinates_question"] = json.dumps(vec.tolist())
        if (
            pd.isna(row["coordinates_answer"])
            or row["coordinates_answer"] == "None"
        ):
            vec = model.encode(
                str(row[ANSWER_COL]),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            df.at[i, "coordinates_answer"] = json.dumps(vec.tolist())

    print("Saving updated CSV...")
    df.to_csv("napoleon_bonaparte.csv", index=False)
    print("CSV updated with embeddings.")
else:
    print("Embeddings already present, skipping encoding.")


"""PHASE 2 : THE USER ASK A QUESTION, GET THE BEST QUESTION ANSWER PAIR, AND PUT IN INTO THE PROMPT"""
print(
    "\nPHASE 2 : THE USER ASK A QUESTION, GET THE BEST QUESTION ANSWER PAIR, AND PUT IN INTO THE PROMPT"
)
print("Loading embeddings into numpy arrays...")
question_vectors = np.vstack(
    df["coordinates_question"].apply(lambda x: np.array(json.loads(x))).values
)
answer_vectors = np.vstack(
    df["coordinates_answer"].apply(lambda x: np.array(json.loads(x))).values
)

while True:
    user_input = str(input("\nYou: "))
    new_point = model.encode(
        user_input, convert_to_numpy=True, show_progress_bar=False
    )

    distances = cosine_distances([new_point], question_vectors)[0]
    nearest_idx = np.argsort(distances)[:5]  # Get top 5

    # Collect the best 5 Question Answer pairs
    qa_list = []
    for idx in nearest_idx:
        q = df.iloc[idx][QUESTION_COL]
        a = df.iloc[idx][ANSWER_COL]
        qa_list.append(f"Q: {q}\nA: {a}")

    qa_text = "\n\n".join(qa_list)

    # Format the prompt
    final_prompt = PROMPT_TEMPLATE.format(
        user_input=user_input, qa_pairs=qa_text
    )

    print("\n \n \n \n \n \n \n \n \n \n")
    print(final_prompt)
    # You can send the prompt to an LLM like ChatGPT, GEMINI for generate the answer
