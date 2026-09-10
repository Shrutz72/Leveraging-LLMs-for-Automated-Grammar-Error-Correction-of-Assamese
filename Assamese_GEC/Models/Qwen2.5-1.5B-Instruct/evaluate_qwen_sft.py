import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL = "/home/shrutee/assamese_gec/final_model_qwen"
TEST_PATH = "/home/shrutee/assamese_gec/test_dataset_10000_with_correct_sentences.csv"
OUTPUT_PATH = "/home/shrutee/assamese_gec/qwen_test_predictions_with_metrics.csv"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
PRINT_EVERY = 20
SAVE_EVERY = 50


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


def normalize_text(text):
    return " ".join(str(text).strip().split())


def make_user_content(row):
    return f"""Correct the Assamese sentence. Use the given error types as guidance. Return only the corrected sentence.

Error Types:
{row['error_text']}

Sentence:
{row['incorrect_sentence']}"""


def make_prompt(tokenizer, row):
    messages = [
        {
            "role": "user",
            "content": make_user_content(row)
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def make_full_text(tokenizer, row, target):
    messages = [
        {
            "role": "user",
            "content": make_user_content(row)
        },
        {
            "role": "assistant",
            "content": target
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def extract_prediction(text):
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("assistant", "")
    return normalize_text(text)


def calculate_loss(model, tokenizer, batch_rows, targets):
    prompts = [make_prompt(tokenizer, row) for _, row in batch_rows.iterrows()]
    full_texts = [
        make_full_text(tokenizer, row, target)
        for (_, row), target in zip(batch_rows.iterrows(), targets)
    ]

    inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=640,
    ).to(model.device)

    labels = inputs["input_ids"].clone()

    prompt_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    for i in range(len(prompts)):
        prompt_len = prompt_inputs["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100

    labels[inputs["attention_mask"] == 0] = -100

    with torch.inference_mode():
        outputs = model(**inputs, labels=labels)

    return outputs.loss.item()


def simple_tokenize(text):
    return normalize_text(text).split()


def sentence_metrics(pred, ref):
    pred_tokens = simple_tokenize(pred)
    ref_tokens = simple_tokenize(ref)

    if len(pred_tokens) == 0:
        return 0.0, 0.0, 0.0

    smoothie = SmoothingFunction().method1

    bleu = sentence_bleu(
        [ref_tokens],
        pred_tokens,
        smoothing_function=smoothie
    )

    gleu = sentence_gleu([ref_tokens], pred_tokens)

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)

    tp = len(pred_set & ref_set)

    precision = tp / len(pred_set) if len(pred_set) else 0.0
    recall = tp / len(ref_set) if len(ref_set) else 0.0

    beta = 0.5

    if precision + recall == 0:
        f05 = 0.0
    else:
        f05 = (1 + beta * beta) * precision * recall / (
            (beta * beta * precision) + recall
        )

    return gleu, bleu, f05


def main():
    df = pd.read_csv(TEST_PATH)

    df = df[["incorrect_sentence", "correct_sentence", "error_label"]].dropna()

    df["incorrect_sentence"] = df["incorrect_sentence"].astype(str).apply(normalize_text)
    df["correct_sentence"] = df["correct_sentence"].astype(str).apply(normalize_text)
    df["error_label"] = df["error_label"].astype(str).str.strip(",")
    df["error_text"] = df["error_label"].apply(label_to_text)

    print("Test size:", len(df))

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading LoRA model...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL,
        local_files_only=True,
    )

    model.eval()

    predictions = []
    losses = []

    print("\nStarting evaluation...\n")

    for batch_no, i in enumerate(
        tqdm(range(0, len(df), BATCH_SIZE), desc="Evaluating"),
        start=1
    ):
        batch = df.iloc[i:i + BATCH_SIZE]
        targets = batch["correct_sentence"].tolist()

        prompts = [make_prompt(tokenizer, row) for _, row in batch.iterrows()]

        batch_loss = calculate_loss(model, tokenizer, batch, targets)
        losses.append(batch_loss)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]

        batch_predictions = []

        for output in outputs:
            new_tokens = output[prompt_length:]
            decoded = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            )
            pred = extract_prediction(decoded)
            batch_predictions.append(pred)

        predictions.extend(batch_predictions)

        if batch_no % PRINT_EVERY == 0 or batch_no == 1:
            current_refs = df.iloc[:len(predictions)]["correct_sentence"].tolist()

            running_exact = [
                normalize_text(pred) == normalize_text(ref)
                for pred, ref in zip(predictions, current_refs)
            ]

            running_acc = np.mean(running_exact)
            running_loss = np.mean(losses)

            print("\n" + "=" * 80)
            print(f"Batch {batch_no} | Processed: {len(predictions)}/{len(df)}")
            print(f"Running Exact Match: {running_acc:.4f}")
            print(f"Running Loss       : {running_loss:.4f}")
            print("-" * 80)
            print("Sample check:")
            print("Incorrect:", batch.iloc[0]["incorrect_sentence"])
            print("Gold     :", targets[0])
            print("Pred     :", batch_predictions[0])
            print("=" * 80)

        if batch_no % SAVE_EVERY == 0:
            partial_df = df.iloc[:len(predictions)].copy()
            partial_df["prediction"] = predictions
            partial_df.to_csv(OUTPUT_PATH, index=False)

    df["prediction"] = predictions

    df["prediction_norm"] = df["prediction"].apply(normalize_text)
    df["correct_norm"] = df["correct_sentence"].apply(normalize_text)
    df["exact_match"] = df["prediction_norm"] == df["correct_norm"]

    print("\nCalculating GLEU, BLEU, and F0.5...")

    gleu_scores = []
    bleu_scores = []
    f05_scores = []

    for pred, ref in tqdm(
        zip(df["prediction_norm"], df["correct_norm"]),
        total=len(df),
        desc="Metrics"
    ):
        gleu, bleu, f05 = sentence_metrics(pred, ref)
        gleu_scores.append(gleu)
        bleu_scores.append(bleu)
        f05_scores.append(f05)

    df["gleu"] = gleu_scores
    df["bleu"] = bleu_scores
    df["f0_5"] = f05_scores

    exact_accuracy = df["exact_match"].mean()
    avg_loss = np.mean(losses)
    avg_gleu = df["gleu"].mean()
    avg_bleu = df["bleu"].mean()
    avg_f05 = df["f0_5"].mean()

    print("\nFINAL RESULTS")
    print("=" * 80)
    print(f"Loss        : {avg_loss:.4f}")
    print(f"Exact Match : {exact_accuracy:.4f}")
    print(f"GLEU        : {avg_gleu:.4f}")
    print(f"BLEU        : {avg_bleu:.4f}")
    print(f"F0.5        : {avg_f05:.4f}")

    print("\nMetrics by error label:")
    label_metrics = df.groupby("error_label")[[
        "exact_match",
        "gleu",
        "bleu",
        "f0_5"
    ]].mean().sort_index()

    print(label_metrics)

    df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved predictions and metrics to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()