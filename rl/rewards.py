import re
from stockfish import Stockfish
import chess
import chess.engine

def extract_moves_strict(completion):
    match = re.search(r"<best_move>([a-h][1-8][a-h][1-8][qrbn]?)</best_move>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

def extract_moves(completion):
    match = re.search(r"<best_move>(.*)</best_move>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<best_move>[a-h][1-8][a-h][1-8]</best_move>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else -1.0 for match in matches]
    return rewards_list

def accuracy_reward(completions, fen, best_move, **kwargs):
    stockfish = Stockfish(
        limit=chess.engine.Limit(time=0.01)
    )
    completion_contents = [completion[0]["content"] for completion in completions]
    moves = [extract_moves(comp) for comp in completion_contents]

    scores = []
    for comp, fen, move, best_move in zip(completion_contents, fen, moves, best_move):
        print("Board: ", str(chess.Board(fen)))
        print("Model Move: ", move)
        print("Best Move: ", best_move)
        print("Completion: ", comp)
        print("="*50)
        if move is None:
            scores.append(-1000)
        else:
            scores.append(stockfish.get_score(fen, move))

    rewards = [score/1000.0 for score in scores]

    print("Final scores: ", scores)
    print("Final rewards: ", rewards)
    del stockfish
    return rewards
