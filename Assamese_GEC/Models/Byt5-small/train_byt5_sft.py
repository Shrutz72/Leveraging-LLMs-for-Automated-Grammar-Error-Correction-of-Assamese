import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model


MODEL_ID = "google/byt5-small"

DATA_PATH = "/home/shrutee/assamese_gec/train_dataset_90000_with_correct_sentence.csv"

OUTPUT_DIR = "/home/shrutee/assamese_gec/byt5_assamese_gec_lora_labels"
FINAL_MODEL_DIR = "/home/shrutee/assamese_gec/final_model_byt5"


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


def create_input(row):
    return f"""Correct the Assamese sentence. Use the given error types as guidance. Return only the corrected sentence.

Error Types:
{row['error_text']}

Sentence:
{row['incorrect_sentence']}"""


def tokenize_function(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=512,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=512,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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

    df["input_text"] = df.apply(create_input, axis=1)
    df["target_text"] = df["correct_sentence"]

    dataset = Dataset.from_pandas(
        df[["input_text", "target_text"]],
        preserve_index=False
    )

    print("Dataset size:", len(dataset))
    print("Sample input:")
    print(dataset[0]["input_text"][:700])
    print("\nSample target:")
    print(dataset[0]["target_text"])

    print("Loading model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
        use_safetensors=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q",
            "k",
            "v",
            "o",
            "wi_0",
            "wi_1",
            "wo",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Tokenizing dataset...")

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
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
        
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
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