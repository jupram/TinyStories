import argparse
import json
import os
from typing import Dict, List, Optional

import torch

from .eval_harness import _find_run_dirs, generate_completion, load_model_and_tokenizer, pick_checkpoint
from .utils import ensure_dir, select_device, set_seed


def read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt and prompt.strip():
        return prompt.strip()
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    raise ValueError("Provide a non-empty prompt via --prompt or --prompt_file")


def to_full_text(prompt: str, completion: str) -> str:
    if not completion:
        return prompt
    if prompt.endswith((" ", "\n", "\t")):
        return prompt + completion
    return f"{prompt}{completion}"


def main():
    parser = argparse.ArgumentParser(description="Generate story completions across all trained runs")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run folders")
    parser.add_argument("--run_names", type=str, nargs="*", default=None, help="Specific run names to include")
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Optional checkpoint step to load for each run (default: latest)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text to continue")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a file containing the prompt")
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto/cuda/cpu")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff (<=0 disables)")
    parser.add_argument("--num_samples", type=int, default=1, help="Completions per model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to save all completions as JSON")
    args = parser.parse_args()

    prompt = read_prompt(args.prompt, args.prompt_file)
    set_seed(args.seed)

    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    run_dirs = _find_run_dirs(args.runs_dir, args.run_names)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    num_samples = max(1, int(args.num_samples))

    results: List[Dict] = []

    print("=" * 88)
    print("Prompt:")
    print(prompt)
    print("=" * 88)

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        checkpoint_path = pick_checkpoint(run_dir, args.checkpoint_step)
        checkpoint_name = os.path.basename(checkpoint_path)
        model, tokenizer, seq_len, model_name = load_model_and_tokenizer(run_dir, checkpoint_path, device)

        run_payload = {
            "run_name": run_name,
            "model": model_name,
            "checkpoint": checkpoint_name,
            "completions": [],
        }

        print(f"run={run_name} model={model_name} checkpoint={checkpoint_name}")
        for sample_idx in range(num_samples):
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                seq_len=seq_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
            full_text = to_full_text(prompt, completion)
            run_payload["completions"].append(
                {
                    "sample": sample_idx + 1,
                    "completion": completion,
                    "full_text": full_text,
                }
            )

            if num_samples == 1:
                print(full_text)
            else:
                print(f"[sample {sample_idx + 1}] {full_text}")
        print("-" * 88)

        results.append(run_payload)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            ensure_dir(out_dir)
        payload = {
            "prompt": prompt,
            "settings": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": top_k,
                "num_samples": num_samples,
                "seed": args.seed,
                "device": str(device),
            },
            "results": results,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved completions to {args.out_json}")


if __name__ == "__main__":
    main()
