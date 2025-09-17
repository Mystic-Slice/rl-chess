import pandas as pd
import datasets
import fen_tokenizer
import chess

def process_example(sample):
    fen_string = sample['fen']

    board = chess.Board(fen_string)
    board_string = str(board)

    best_move = sample['best_move_uci']
    
    # Only UCI
    # PROMPT_MESSAGES = [
    #     {
    #         'role': 'user',
    #         'content': "You are given a chess board. You should suggest the next best move from that position. The best move should be in the Universal Chess Interface notation `<start cell><end cell>`. Output your reasoning between <reasoning> and </reasoning> symbols. Keep it short. Output the final move between <best_move> and </best_move> symbols."\
    #             f"Chess Board: \n{board_string}" \
    #     }
    # ]

    # Allow SAN
    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': "You are given a chess board. You should suggest the next best move from that position. Output your reasoning between <reasoning> and </reasoning> symbols. Output the final move between <best_move> and </best_move> symbols."\
                f"Chess Board: \n{board_string}" \
        }
    ]

    return {
        'prompt': PROMPT_MESSAGES,
    }

def get_data():
    df = pd.read_csv("puzzles_with_reasoning.csv")
    ds = datasets.Dataset.from_pandas(df)

    ds = ds.map(process_example, num_proc=8)

    ds = ds.train_test_split(test_size=0.01, seed=42)

    return ds

    
