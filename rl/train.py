from data import get_data
from model import get_model

from trl import GRPOConfig, GRPOTrainer

from rewards import format_reward, accuracy_reward

model, tokenizer = get_model("Qwen/Qwen3-4B-Instruct-2507")

print(model)
print(tokenizer)

ds = get_data()
print(ds)

train_dataset = ds['train']
test_dataset = ds['test']

training_args = GRPOConfig(
    output_dir="rl_chess_board_str_san",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=64,
    max_steps=500,
    # num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    num_generations=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    max_completion_length=512,
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
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train(resume_from_checkpoint=True)
# trainer.train()

trainer.save_model(training_args.output_dir + '/final_model')

print("All done!!!")
