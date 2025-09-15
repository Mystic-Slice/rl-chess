from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import os

df = pd.read_csv("puzzles_modified.csv")
print(df.head())


pipe = pipeline('text-generation', model='Qwen/Qwen3-4B-Instruct-2507', device='cuda')
print("Loaded model: ", pipe, pipe.device)

def process_example(example):
    fen_modified = example['FEN']
    best_move = example['Best Move UCI']

    prompt_messages = [
        {
            'role': 'user',
            'content': "You are give a chess board in the FEN format and the next best move in UCI format (<start cell><end cell>). Assume the move given to you is always valid. You should output a short valid reasoning (~ 200 words) for why that move is the next best move, as if you are playing the game. Do not indicate that this move was give to you in your reasoning.\n"\
                f"Chess Board: {fen_modified}\n"\
                    f"Next Move: {best_move}\n"
        }
    ]

    # print(prompt_messages)

    output = pipe(prompt_messages, max_new_tokens=512)[0]['generated_text'][-1]['content']

    return output

# df = df[:15]

reasonings = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    filename = f"reasonings/{i}.txt"
    if os.path.exists(filename):
        print(f"Skipping {i} -> Already Exists")
        continue

    output = process_example(row)
    reasonings.append(output)
    # print(output)

    with open(filename, 'w') as f:
        print(output, file=f, end="")