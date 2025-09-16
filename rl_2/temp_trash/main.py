
import os
from pathlib import Path


import gc
import re
import time
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from rewards import format_reward_sample, accuracy_reward_sample, compute_reward
from data import get_data
from model import get_model

import wandb
from utils import (
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_free_port,
    find_last_checkpoint,
    prepare_model_inputs,
    load_model_into_vllm
)


# Needed to stop DeepSpeed from complaining
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(find_free_port())
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

os.environ['VLLM_USE_V1']='0'


# Model configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Total number of training iterations
NUM_ITERATIONS = 1000
NUM_SAMPLES = 1
# Number of responses to generate for each input prompt (i.e. group size in GRPO)
GENERATIONS_PER_SAMPLE = 16
# Number of episodes to collect per iteration for training
EPISODES_PER_ITERATION = NUM_SAMPLES * GENERATIONS_PER_SAMPLE
# Controls how much the policy can deviate from the reference model
KL_COEFFICIENT = 0.001

SAVE_STEPS = 20
EVAL_STEPS = 50

# Training hyperparameters
# Batch size for each GPU device during training
PER_DEVICE_BATCH_SIZE = 8
# Learning rate for model updates
LEARNING_RATE = 2e-5

# Sampling parameters
# Maximum number of tokens to generate in each response
MAX_RESPONSE_TOKENS = 512
# Controls randomness in generation (higher = more random)
TEMPERATURE = 1.0
# Nucleus sampling parameter (1.0 = disabled)
TOP_P = 1.0
# Top-k sampling parameter (-1 = disabled)
TOP_K = -1  # no top k

# DeepSpeed configuration
# DeepSpeed config for the policy model
deepspeed_config = {
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 2, "overlap_comm": False},
    "train_batch_size": EPISODES_PER_ITERATION,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    "gradient_clipping": 1.0,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": LEARNING_RATE,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "torch_adam": True,
        },
    },
}
# DeepSpeed config for the reference model
ref_deepspeed_config = {
    "bf16": {"enabled": True},
    # Note that we don't train the reference model
    # These are just for compatibility with DeepSpeed.
    "train_batch_size": EPISODES_PER_ITERATION,
    "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
}

RUN_NAME = "qwen_rl_chess_test"
EXP_DIR = Path("outputs") / RUN_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Logs and Checkpoints will be saved to: {EXP_DIR}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
EOS_TOKEN_ID = AutoTokenizer.from_pretrained(MODEL_NAME).eos_token_id
EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)


ds = get_data(tokenizer)
train_dataset = ds['train']
test_dataset = ds['test']
len(train_dataset), len(test_dataset)


print("Target: ", train_dataset[0]["fen"])
print("Available Numbers: ", train_dataset[0]["best_move"])


print(train_dataset[0]["prompt"])


print(train_dataset[0]["input_ids"])


# format_reward_sample("<think>I think the answer is </think>\n<answer>1+2</answer")


# format_reward_sample("I think the answer is </think>\n<answer>1+2</answer>")


# accuracy_reward_sample("I think the answer is </think>\n<answer>1+2</answer>", train_dataset[0])


# accuracy_reward_sample("<reasoning>asdffsd</reasoning><best_move>Nxa3</best_move>", train_dataset[0])


def create_training_episodes(samples, all_generations, all_finish_reasons):
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE))
        for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        'response_lengths': [],
        'rewards': [],
        'non_stop_rate': [],
    }

    for sample, group_indices in zip(samples, groups):
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        response_token_ids = [all_generations[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)

        rewards_and_metrics = [compute_reward(resp, sample) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)
        rewards = np.array(rewards)
        response_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        advantages = [
            [resp_adv] * len(resp)
            for resp_adv, resp in zip(response_advantages, response_token_ids)
        ]

        all_query_token_ids.extend([sample['input_ids']] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(advantages)

        stats['rewards'].extend(rewards)
        stats['non_stop_rate'].extend([fr != 'stop' for fr in finish_reasons])
        stats['response_lengths'].extend([len(ids) for ids in response_token_ids])

        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        'all_query_token_ids': all_query_token_ids,
        'all_response_token_ids': all_responses_token_ids,
        'all_advantages': all_advantages,
    }
    return episodes, stats



# case_0 = {
#     "sample": {"input_ids": [1,2,3], "nums": [1,2,3], "target": 6},
#     "generations": [[4,5, 22, 33], [6,7], [8,9, 11], [10,11]],
#     "finish_reasons": ["stop", "length", "stop", "stop"]
# }

# case = case_0
# episodes, stats = create_training_episodes([case["sample"]], case["generations"], case["finish_reasons"])
# episodes


# case_1 = {
#     "sample": {"input_ids": [33, 44], "nums": [11, 7, 8], "target": 26},
#     "generations": [[1,2], [3,4], [5,6], [7,8]],
#     "finish_reasons": ["stop", "stop", "length", "stop"]
# }
# case = case_1
# episodes, stats = create_training_episodes([case["sample"]], case["generations"], case["finish_reasons"])
# episodes


# case_2 = {
#     "sample": {"input_ids": [9, 8, 7, 6, 5, 4], "nums": [1,2,3,4], "target": 10},
#     "generations": [[9,10], [11,12], [13,14], [15,16]],
#     "finish_reasons": ["length", "length", "stop", "stop"]
# }
# case = case_2
# episodes, stats = create_training_episodes([case["sample"]], case["generations"], case["finish_reasons"])
# episodes


def compute_pg_loss(policy_model, reference_model, batch, total_response_len):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    advantages = batch['advantages']

    labels_mask = (labels[..., 1:] != -100).float()

    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        # 'labels_mask': labels_mask,
    }

    with torch.no_grad():
        ref_logps = compute_token_log_probs(
            reference_model, model_inputs, TEMPERATURE
        )

    logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)

    kl_penalty = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    kl_penalty = kl_penalty * labels_mask

    entropy = -logps.sum() / labels_mask.sum()

    policy_loss = -logps * advantages[..., 1:]
    policy_loss = policy_loss * labels_mask

    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len

    metrics = {
        'policy_loss': policy_loss.sum().item() / total_response_len,
        'kl_penalty': kl_penalty.sum().item() / total_response_len,
        'entropy': entropy.item() / total_response_len
    }
    return loss, metrics


# Initialize main and reference models
policy_model, _ = get_model(MODEL_NAME)
reference_model, _ = get_model(MODEL_NAME)


policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})


policy_model, *_ = deepspeed.initialize(
    model=policy_model,
    config=deepspeed_config,
    model_parameters=policy_model.parameters(),
)
reference_model, *_ = deepspeed.initialize(
    model=reference_model,
    config=deepspeed_config,
)
reference_model.module.cpu()

# print(reference_model.device)
# reference_model.cpu()
# print(reference_model.device)


inference_engine = LLM(
    model=MODEL_NAME,
    skip_tokenizer_init=False,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
    swap_space=1,
    scheduling_policy="fcfs",
    dtype=torch.bfloat16,
    max_model_len=768,
    enable_sleep_mode=True,
)


# Load checkpoint if it exists
begin_iter = 0
ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
if ckpt_path is not None:
    print(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
    out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
    if out is None:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
    begin_iter = ckpt_iter + 1
    load_model_into_vllm(policy_model, inference_engine)


for iteration in trange(NUM_ITERATIONS):
    print(f"Iteration {iteration}/{NUM_ITERATIONS}")

    metrics = {}

    eval_stats = None
    if iteration % EVAL_STEPS == 0:
        print("Evaluating on eval set...")
        eval_episodes, eval_stats = evaluate_on_test_set(
            inference_engine=inference_engine,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            eos_token=EOS_TOKEN,
            eval_sampling_params=SamplingParams(
                temperature=0.3,
                max_tokens=1024,
                n=1,
                detokenize=False,
                stop_token_ids=[EOS_TOKEN_ID],
            ),
            reward_func=lambda completion, sample: compute_reward(
                completion, sample
            ),
        )

    num_samples = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE
    indices = np.random.choice(
        len(train_dataset), size=num_samples, replace=False
    )
    samples = train_dataset.select(indices)

    # Sample responses
    outputs = inference_engine.generate(
        prompt_token_ids=list(samples["input_ids"]),
        sampling_params=SamplingParams(
            n=GENERATIONS_PER_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_RESPONSE_TOKENS,
            detokenize=False,
            stop_token_ids=[EOS_TOKEN_ID],
        )
    )
    all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
    all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]
    inference_engine.sleep(1)

    print(f"Generated {len(all_generations)} responses")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    # Process responses and calculate rewards
    episodes, episodes_stats = create_training_episodes(
        samples,
        all_generations,
        all_finish_reasons,
    )
    for k, v in episodes_stats.items():
        metrics.setdefault(k, []).extend(v)

    episode_table = dump_episodes(
        episodes=episodes,
        episodes_stats=episodes_stats,
        exp_dir=EXP_DIR,
        tokenizer=tokenizer,
        iteration=iteration,
    )

    #########################################################
    # Training
    #########################################################

    # Prepare training batch
    model_inputs = prepare_model_inputs(
        query_token_ids=episodes["all_query_token_ids"],
        response_token_ids=episodes["all_response_token_ids"],
        advantages=episodes["all_advantages"],
        device="cuda"
    )

    # Calculate losses and update model
    policy_model.train()
    # reference_model.cuda()
    # deepspeed
    reference_model.module.cuda()
    reference_model.eval()

    total_response_len = (model_inputs["labels"] != -100).sum().item()

    for i in trange(0, EPISODES_PER_ITERATION, PER_DEVICE_BATCH_SIZE, desc="Gradient Accumulation"):
        batch = {
            k: v[i : i + PER_DEVICE_BATCH_SIZE]
            for k, v in model_inputs.items()
        }

        # Compute policy gradient loss
        loss, loss_metrics = compute_pg_loss(
            policy_model=policy_model,
            reference_model=reference_model,
            batch=batch,
            total_response_len=total_response_len,
        )

        # Track metrics
        metrics.setdefault("loss", []).append(loss.item())
        # grad_norm = policy_model.get_global_grad_norm()
        # if grad_norm is not None:
        #     grad_norm = grad_norm.item()
        # metrics.setdefault("grad_norm", []).append(grad_norm)
        for k, v in loss_metrics.items():
            metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

        # Backpropagation and optimization step
        policy_model.backward(loss, scale_wrt_gas=False)
        
        # Free memory
        del loss, loss_metrics
        if policy_model.is_gradient_accumulation_boundary():
            # reference_model.cpu()
            # deepspeed
            reference_model.module.cpu()

        policy_model.step()

    #########################################################
    # Update inference engine weights
    #########################################################
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    inference_engine.wake_up()
    load_model_into_vllm(policy_model, inference_engine)

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


    #########################################################
    # Log metrics
    #########################################################

    train_metrics = {
        k: np.mean(v) for k, v in metrics.items() if None not in v
    }
    train_metrics["learning_rate"] = policy_model.get_lr()[0]
    logs = {
        "iteration": iteration,
        f"episodes/iter_{iteration:06d}": episode_table,
        **{f"train/{k}": v for k, v in train_metrics.items()},
    }
    if eval_stats is not None:
        eval_metrics = {k: np.mean(v) for k, v in eval_stats.items() if None not in v}
        logs.update({f"eval/{k}": v for k, v in eval_metrics.items()})
    # wandb.log(logs)

    selected_keys = [
        "train/kl_penalty",
        "train/rewards",
        "train/reward_metrics/format_reward",
        "train/reward_metrics/equation_reward",
        "eval/rewards",
        "eval/reward_metrics/format_reward",
        "eval/reward_metrics/equation_reward",
    ]
    selected_metrics = {k: logs[k] for k in selected_keys if k in logs}
    print(f"KEY METRICS: {selected_metrics}")

    if iteration % SAVE_STEPS == 0 and iteration != 0:
        # policy_model.save_pretrained(
        #     str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model")
        # )
        # deepspeed
        policy_model.module.save_pretrained(
            str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "hf_model")
        )
        policy_model.save_checkpoint(
            str(EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}" / "deepspeed")
        )


