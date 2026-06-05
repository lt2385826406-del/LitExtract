"""
"""
import os
import sys
import json
import argparse
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ──────────────────────────────────────────────────────────
#
# ──────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "match":          ["llama-3", "llama3"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "pad_side":       "right",
        "eos_as_pad":     True,
    },
    {
        "match":          ["qwen2.5", "qwen"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "pad_side":       "right",
        "eos_as_pad":     False,
    },
    {
        "match":          ["mistral"],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "pad_side":       "right",
        "eos_as_pad":     True,
    },
]


def get_model_config(model_path: str) -> dict:
    lower = model_path.lower().replace("\\", "/")
    for cfg in MODEL_CONFIGS:
        if any(k in lower for k in cfg["match"]):
            return cfg
    return {
        "match":          [],
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "pad_side":       "right",
        "eos_as_pad":     True,
    }


# ──────────────────────────────────────────────────────────
#
# ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="QLoRA SFT train（Qwen/Mistral/LLaMA-3）")
    ap.add_argument("--train_jsonl",  type=str, required=True,
                    help="Training set path (messages format JSONL)")
    ap.add_argument("--val_jsonl",    type=str, required=True,
                    help="Validation set path")
    ap.add_argument("--base_model",   type=str,
                    default="Qwen2.5-7B-Instruct",
                    help="Model path (local directory or HF repo id)")
    ap.add_argument("--out_dir",      type=str, default="checkpoints",
                    help="Output directory")
    ap.add_argument("--max_seq_len",  type=int, default=8192,
                    help="Max sequence length")
    ap.add_argument("--batch_size",   type=int, default=1,
                    help="per_device_train_batch_size")
    ap.add_argument("--grad_accum",   type=int, default=8,
                    help="gradient_accumulation_steps")
    ap.add_argument("--epochs",       type=float, default=3.0,
                    help="Training epochs")
    ap.add_argument("--lr",           type=float, default=2e-4,
                    help="learning rate")
    ap.add_argument("--lora_r",       type=int, default=16,
                    help="LoRA rank")
    ap.add_argument("--lora_alpha",   type=int, default=32,
                    help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05,
                    help="LoRA dropout")
    ap.add_argument("--eval_steps",   type=int, default=50,
                    help="Evaluation steps interval")
    ap.add_argument("--save_steps",   type=int, default=50,
                    help="Save steps interval")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("QLoRA Training — Qwen / Mistral / LLaMA-3")
    print(f"  base_model  : {args.base_model}")
    print(f"  train_jsonl : {args.train_jsonl}")
    print(f"  val_jsonl   : {args.val_jsonl}")
    print(f"  out_dir     : {args.out_dir}")
    print(f"  max_seq_len : {args.max_seq_len}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}  grad_accum={args.grad_accum}")
    print(f"  lora_r      : {args.lora_r}  lora_alpha={args.lora_alpha}")
    print("=" * 60)

    mcfg = get_model_config(args.base_model)
    print(f"[Config] matched: {mcfg}")

    # GPU
    print(f"[GPU] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  VRAM={props.total_memory // 1024**3} GB")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[Config] bf16={use_bf16}  fp16={use_fp16}")

    #
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Tokenizer
    print(f"\n[Load] Tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, trust_remote_code=True,
    )
    tokenizer.padding_side = mcfg["pad_side"]

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        if mcfg["eos_as_pad"]:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("No pad_token and eos_as_pad=False")

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"Tokenizer has no chat_template — use Instruct/Chat variant")
    print(f"[Tokenizer] vocab={tokenizer.vocab_size}  pad={tokenizer.pad_token!r}")

    # Model
    print(f"\n[Load] Model: {args.base_model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    print(f"[Load] Done in {time.time()-t0:.1f}s")
    model.config.use_cache = False

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=mcfg["target_modules"],
    )
    print(f"[LoRA] r={args.lora_r}  alpha={args.lora_alpha}")

    # Dataset
    print(f"\n[Data] Loading ...")
    ds = load_dataset("json", data_files={
        "train":      args.train_jsonl,
        "validation": args.val_jsonl,
    })
    print(f"[Data] train={len(ds['train'])}  val={len(ds['validation'])}")

    def to_text(example):
        messages = example.get("messages")
        if messages is None:
            raise ValueError("Each sample must contain 'messages' field")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    print("[Data] Applying chat_template ...")
    keep_cols = ["text"]
    ds = ds.map(to_text, remove_columns=[
        c for c in ds["train"].column_names if c not in keep_cols
    ])
    if set(ds["validation"].column_names) - {"text"}:
        ds["validation"] = ds["validation"].remove_columns([
            c for c in ds["validation"].column_names if c not in keep_cols
        ])

    # SFT Config
    total_steps = (len(ds["train"]) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * 0.03)
    sft_cfg_kwargs = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
    )
    try:
        sft_cfg = SFTConfig(max_length=args.max_seq_len, **sft_cfg_kwargs)
    except TypeError:
        sft_cfg = SFTConfig(max_seq_length=args.max_seq_len, **sft_cfg_kwargs)

    # Trainer
    print("\n[Train] Starting SFTTrainer ...")
    try:
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=ds["train"], eval_dataset=ds["validation"],
            peft_config=lora, args=sft_cfg,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=ds["train"], eval_dataset=ds["validation"],
            peft_config=lora, args=sft_cfg,
        )

    t_start = time.time()
    trainer.train()
    elapsed = time.time() - t_start
    print(f"[Train] Finished in {elapsed/3600:.2f}h")

    # Save
    print(f"[Save] Saving to {args.out_dir}")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    config_path = os.path.join(args.out_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        cfg_dict = vars(args)
        cfg_dict["train_time_hours"] = round(elapsed / 3600, 3)
        cfg_dict["bf16"] = use_bf16
        cfg_dict["fp16"] = use_fp16
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
    print(f"[Save] Config: {config_path}")
    print("\n[Done]")


if __name__ == "__main__":
    main()
