import os
import torch
import pandas as pd

from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


HF_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"

MODEL_ID = "google/gemma-3-1b-it"

DATA_PATH = "./train_dataset_90000_with_correct_sentence.csv"

OUTPUT_DIR = "./gemma3_1b_assamese_gec_lora_labels"
FINAL_MODEL_DIR = "./final_model"


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
    labels = [x.strip() for x in str(label).split(",")]
    return ", ".join(LABEL_MAP.get(x, x) for x in labels)


def create_prompt(row):
    return f"""<start_of_turn>user
Correct the Assamese sentence. Use the given error types as guidance. Return only the corrected sentence.

Error Types:
{row['error_text']}

Sentence:
{row['incorrect_sentence']}<end_of_turn>
<start_of_turn>model
{row['correct_sentence']}<end_of_turn>"""


def main():
    login(HF_TOKEN)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU found. Stop and check server GPU allocation.")
        return

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

    df["incorrect_sentence"] = df["incorrect_sentence"].astype(str)
    df["correct_sentence"] = df["correct_sentence"].astype(str)
    df["error_label"] = df["error_label"].astype(str).str.strip(",")

    df["error_text"] = df["error_label"].apply(label_to_text)
    df["text"] = df.apply(create_prompt, axis=1)

    dataset = Dataset.from_pandas(df[["text"]])

    print("Dataset size:", len(dataset))
    print("Sample:")
    print(dataset[0]["text"][:500])

    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        warmup_steps=200,
        lr_scheduler_type="cosine",
        max_length=512,
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