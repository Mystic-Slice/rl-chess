from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_data

from stockfish import Stockfish
import chess
import chess.engine

import re

def extract_moves(completion):
    match = re.search(r"<best_move>([a-h][1-8][a-h][1-8][qrbn]?)</best_move>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

# def get_prompt(fen_string):
#     PROMPT_MESSAGES = [
#         {
#             'role': 'user',
#             'content': "You are given a chess board. You should suggest the next best move from that position. The best move should be in the Universal Chess Interface notation `<start cell><end cell>`. Output your reasoning between <reasoning> and </reasoning> symbols. Keep it short. Output the final move between <best_move> and </best_move> symbols."\
#                 f"Chess Board: {fen_string}" \
#         }
#     ]

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print("Base model loaded")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print("Tokenizer loaded")

peft_model = PeftModelForCausalLM.from_pretrained(base_model, "rl_chess_out/final_model")
print("Peft model Loaded")

pipe = pipeline('text-generation', model=peft_model, tokenizer=tokenizer)

print(pipe.device)

ds = get_data()

print(ds['test'])

test_ds = ds['test']

stockfish = Stockfish(
    limit=chess.engine.Limit(time=0.01)
)

for sample in test_ds.select(range(10)):
    print("fen:", sample['fen'])
    board = chess.Board(sample['fen'])

    print(board)
    print("Best Move: ", sample['best_move'])
    print("Best Move UCI: ", sample['best_move_uci'])
    
    score = stockfish.get_score(sample['fen'], sample['best_move_uci'])
    print("Stockfish evaluation: ", score)

    response = pipe(sample['prompt'], max_new_tokens=512)[0]['generated_text'][-1]['content']
    move = extract_moves(response)
    print("Model response: ", response)
    print("Model move: ", move)

    score = stockfish.get_score(sample['fen'], move)
    print("Stockfish evaluation: ", score)

    print("="*100)

del stockfish
    
