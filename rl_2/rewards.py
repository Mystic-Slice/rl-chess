import re
from stockfish import Stockfish
import chess
import chess.engine

def extract_move_strict(completion):
    match = re.search(r"<best_move>([a-h][1-8][a-h][1-8][qrbn]?)</best_move>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

def extract_move(completion):
    match = re.search(r"<best_move>(.*)</best_move>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

def format_reward_sample(completion):
    """Reward function that checks if the completion has a specific format."""
    # print(completion)
    pattern = r"^<reasoning>.*?</reasoning>\s*<best_move>(.*)</best_move><|im_end|>$"
    return 1.0 if re.match(pattern, completion, re.DOTALL) else 0.0

def accuracy_reward_sample(completion, sample):
    stockfish = Stockfish(
        limit=chess.engine.Limit(time=0.01)
    )
    move = extract_move(completion)
    # print("Board: ", str(chess.Board(sample['fen'])))
    # print("Model Move: ", move)
    # print("Best Move: ", sample['best_move'])
    # print("Completion: ", completion)
    if move is None:
        reward = -2000
    else:
        reward = stockfish.get_score(sample['fen'], move)

    reward /= 500.0
    # print("Final reward: ", reward)
    del stockfish
    return reward

def compute_reward(completion, sample):
    format_reward = format_reward_sample(completion)
    accuracy_reward = accuracy_reward_sample(completion, sample)

    reward = format_reward + accuracy_reward

    metrics = {
        'format_reward': format_reward,
        'accuracy_reward': accuracy_reward,
    }

    print(f"Reward: {reward} | Format: {format_reward} | Accuracy: {accuracy_reward}")

    return reward, metrics