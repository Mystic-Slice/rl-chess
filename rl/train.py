from data import get_data
from model import get_model

from trl import GRPOConfig, GRPOTrainer

from rewards import format_reward, accuracy_reward

MODEL_TYPE = 'lora'
# MODEL_TYPE = 'full'

model, tokenizer = get_model("Qwen/Qwen3-4B-Instruct-2507", MODEL_TYPE)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

print(model)
print(tokenizer)

print("Device: ", model.device)

ds = get_data()
print(ds)

train_dataset = ds['train']
test_dataset = ds['test']

training_args = GRPOConfig(
    output_dir="rl_chess_board_test_1024_ctx",
    learning_rate=2e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=64,
    max_steps=500,
    # num_train_epochs=1,
    gradient_checkpointing=True,

    bf16=True,
    # fp16=True,

    # Parameters that control data preprocessing
    num_generations=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    max_completion_length=1024,
    max_prompt_length=256,
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=5,
    save_strategy="steps",
    save_steps=20,

    eval_strategy='steps',
    eval_steps=50,

    use_vllm=True,
    vllm_mode="colocate",
    vllm_model_impl='transformers',
    # vllm_gpu_memory_utilization=0.2, 

    top_k=40,
    top_p=0.95,
    epsilon=0.2,
    epsilon_high=0.28,
    scale_rewards=False,

    reward_weights=[1, 2],

    torch_empty_cache_steps=1,
    # use_liger_loss=True,
    vllm_enable_sleep_mode=True
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print(training_args.device)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

trainer.save_model(training_args.output_dir + '/final_model')

print("All done!!!")
