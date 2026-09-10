import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = "/home/shrutee/assamese_gec/train_dataset_90000_with_correct_sentence.csv"

OUTPUT_DIR = "/home/shrutee/assamese_gec/qwen_assamese_gec_lora_labels"
FINAL_MODEL_DIR = "/home/shrutee/assamese_gec/final_model_qwen"


LABEL_MAP = {
    "0": "Clean",
    "1": "Non-dictionary spelling error",
    "1D": "Dictionary spelling error",
    "2": "Verb error",
    "3": "Tense error",
    "4": "Person agreement error",
    "5": "Number agreement error",
    "6": "Pronoun error",
    "7": "Case marker (Karak) error",
    "8": "Bibhakti/Postposition error",
    "9": "Pratyaya/Word formation error",
    "10": "Syntax/Word order error",
    "11": "Punctuation error",
}


def label_to_text(label):
    labels = [x.strip() for x in str(label).split(",") if x.strip()]
    return ", ".join(LABEL_MAP.get(x, x) for x in labels)


def create_messages(row):
    return [
        {
            "role": "user",
            "content": f"""Correct the Assamese sentence. Use the given error types as guidance. Return only the corrected sentence.

Error Types:
{row['error_text']}

Sentence:
{row['incorrect_sentence']}"""
        },
        {
            "role": "assistant",
            "content": row["correct_sentence"]
        }
    ]


def main():
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("No GPU found. Stop and check server GPU allocation.")
        return

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Using dtype:", dtype)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df = df[
        [
            "incorrect_sentence",
            "correct_sentence",
            "error_label",
        ]
    ]

    df = df.dropna(subset=["incorrect_sentence", "correct_sentence", "error_label"])

    df["incorrect_sentence"] = df["incorrect_sentence"].astype(str).str.strip()
    df["correct_sentence"] = df["correct_sentence"].astype(str).str.strip()
    df["error_label"] = df["error_label"].astype(str).str.strip(",")

    df["error_text"] = df["error_label"].apply(label_to_text)
    df["messages"] = df.apply(create_messages, axis=1)

    df["text"] = df["messages"].apply(
        lambda messages: tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    )

    dataset = Dataset.from_pandas(df[["text"]], preserve_index=False)

    print("Dataset size:", len(dataset))
    print("Sample:")
    print(dataset[0]["text"][:700])

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch_fused",
        warmup_steps=200,
        lr_scheduler_type="cosine",
        max_length=512,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Trainer ready")
    print("Starting training...")

    trainer.train()

    print("Saving final model...")

    trainer.model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("Done. Model saved to:", FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()