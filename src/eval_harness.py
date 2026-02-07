import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from . import data
from .models import build_model
from .utils import check_flash_attention, ensure_dir, select_device, set_seed


@dataclass
class PromptItem:
    prompt_id: str
    category: str
    difficulty: int
    score: float
    prompt: str
    answers: List[str]
    match: str = "contains"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _to_answers(raw: Dict, index: int) -> List[str]:
    if "answers" in raw:
        answers = raw["answers"]
    elif "answer" in raw:
        answers = [raw["answer"]]
    else:
        raise ValueError(f"Prompt index {index} is missing 'answer' or 'answers'")

    if not isinstance(answers, list):
        raise ValueError(f"Prompt index {index} has invalid answers type {type(answers)}")
    clean = [str(a).strip() for a in answers if str(a).strip()]
    if not clean:
        raise ValueError(f"Prompt index {index} has no non-empty answers")
    return clean


def load_prompts(path: str) -> List[PromptItem]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_prompts = payload.get("prompts")
    else:
        raw_prompts = payload

    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("Prompt file must contain a non-empty list or {'prompts': [...]} object")

    prompts: List[PromptItem] = []
    for idx, raw in enumerate(raw_prompts):
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt index {idx} must be a JSON object")
        prompt_text = str(raw.get("prompt", "")).strip()
        if not prompt_text:
            raise ValueError(f"Prompt index {idx} has empty 'prompt'")

        prompt_id = str(raw.get("id", f"prompt_{idx:03d}")).strip()
        category = str(raw.get("category", "general")).strip()
        difficulty = int(raw["difficulty"])
        score = float(raw["score"])
        match = str(raw.get("match", "contains")).strip().lower()
        if match not in {"contains", "exact", "prefix", "regex"}:
            raise ValueError(f"Prompt index {idx} has unsupported match mode '{match}'")

        answers = _to_answers(raw, idx)
        prompts.append(
            PromptItem(
                prompt_id=prompt_id,
                category=category,
                difficulty=difficulty,
                score=score,
                prompt=prompt_text,
                answers=answers,
                match=match,
            )
        )
    return prompts


def _parse_checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def pick_checkpoint(run_dir: str, checkpoint_step: Optional[int]) -> str:
    checkpoints = glob(os.path.join(run_dir, "checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt files found in {run_dir}")

    if checkpoint_step is not None:
        exact = os.path.join(run_dir, f"checkpoint_{checkpoint_step}.pt")
        if not os.path.isfile(exact):
            raise FileNotFoundError(f"Checkpoint {exact} does not exist")
        return exact

    checkpoints = sorted(checkpoints, key=_parse_checkpoint_step)
    return checkpoints[-1]


def _load_config(run_dir: str) -> Dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if not has_prefix:
        return state_dict
    stripped = {}
    for key, value in state_dict.items():
        new_key = key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key
        stripped[new_key] = value
    return stripped


def load_model_and_tokenizer(run_dir: str, checkpoint_path: str, device: torch.device):
    cfg = _load_config(run_dir)
    tokenizer_name = cfg.get("data", {}).get("tokenizer_name", "gpt2")
    tokenizer = data.load_tokenizer(tokenizer_name)
    cfg["model"]["vocab_size"] = tokenizer.vocab_size
    use_flash = check_flash_attention(bool(cfg.get("use_flash", False)))
    model = build_model(cfg, use_flash=use_flash).to(device)

    # Training checkpoints include optimizer/scheduler states, so load full pickle payload.
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = state.get("model", state) if isinstance(state, dict) else state
    if not isinstance(model_state, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    model_state = _strip_compile_prefix(model_state)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    seq_len = int(cfg.get("model", {}).get("seq_len", 256))
    model_name = str(cfg.get("model", {}).get("name", "unknown"))
    return model, tokenizer, seq_len, model_name


def apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, k=k, dim=-1)
    cutoff = values[:, -1].unsqueeze(-1)
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.numel() == 0:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Prompt produced empty tokenization and tokenizer has no eos_token_id")
        input_ids = torch.tensor([[eos_id]], device=device)

    start_len = input_ids.size(1)

    for _ in range(max_new_tokens):
        context = input_ids[:, -seq_len:]
        logits = model(context)[:, -1, :]
        logits = apply_top_k(logits, top_k)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break

    new_tokens = input_ids[:, start_len:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


def _contains_whole_phrase(output_norm: str, answer_norm: str) -> bool:
    if not answer_norm:
        return False
    return f" {answer_norm} " in f" {output_norm} "


def answer_matches(completion: str, answers: Sequence[str], match: str) -> Tuple[bool, Optional[str]]:
    output_norm = normalize_text(completion)
    for answer in answers:
        answer_norm = normalize_text(answer)
        if match == "exact":
            passed = output_norm == answer_norm
        elif match == "prefix":
            passed = output_norm.startswith(answer_norm)
        elif match == "regex":
            passed = re.search(answer, completion, flags=re.IGNORECASE) is not None
        else:
            passed = _contains_whole_phrase(output_norm, answer_norm)
        if passed:
            return True, answer
    return False, None


def _find_run_dirs(runs_dir: str, run_names: Optional[Iterable[str]]) -> List[str]:
    if run_names:
        run_dirs = [os.path.join(runs_dir, name) for name in run_names]
        missing = [p for p in run_dirs if not os.path.isdir(p)]
        if missing:
            raise FileNotFoundError(f"Run directories not found: {missing}")
        return run_dirs

    run_dirs = [p for p in glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    run_dirs.sort()
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_dir}")
    return run_dirs


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_eval_rows_by_prompt(eval_rows: List[Dict], prompts: Sequence[PromptItem]) -> List[Dict]:
    prompt_order = {p.prompt_id: idx for idx, p in enumerate(prompts)}
    return sorted(
        eval_rows,
        key=lambda row: (
            prompt_order.get(row["prompt_id"], 10**9),
            row["prompt_id"],
            row["run_name"],
            row["checkpoint"],
        ),
    )


def build_summary(eval_rows: List[Dict]) -> List[Dict]:
    summary: Dict[str, Dict] = {}
    for row in eval_rows:
        run_name = row["run_name"]
        bucket = summary.setdefault(
            run_name,
            {
                "run_name": run_name,
                "model": row["model"],
                "checkpoint": row["checkpoint"],
                "prompts": 0,
                "passed": 0,
                "points_earned": 0.0,
                "points_possible": 0.0,
            },
        )
        bucket["prompts"] += 1
        bucket["passed"] += int(row["passed"])
        bucket["points_earned"] += float(row["points_earned"])
        bucket["points_possible"] += float(row["points_possible"])

    out = []
    for run_name in sorted(summary.keys()):
        row = summary[run_name]
        prompts = max(1, int(row["prompts"]))
        points_possible = max(1e-12, float(row["points_possible"]))
        row["accuracy"] = row["passed"] / prompts
        row["weighted_accuracy"] = row["points_earned"] / points_possible
        out.append(row)
    return out


def build_breakdown(eval_rows: List[Dict], key: str) -> List[Dict]:
    grouped: Dict[Tuple[str, str], Dict] = {}
    for row in eval_rows:
        run_name = row["run_name"]
        group = str(row[key])
        bucket = grouped.setdefault(
            (run_name, group),
            {
                "run_name": run_name,
                key: group,
                "prompts": 0,
                "passed": 0,
                "points_earned": 0.0,
                "points_possible": 0.0,
            },
        )
        bucket["prompts"] += 1
        bucket["passed"] += int(row["passed"])
        bucket["points_earned"] += float(row["points_earned"])
        bucket["points_possible"] += float(row["points_possible"])

    out = []
    for group_key in sorted(grouped.keys()):
        row = grouped[group_key]
        prompts = max(1, int(row["prompts"]))
        points_possible = max(1e-12, float(row["points_possible"]))
        row["accuracy"] = row["passed"] / prompts
        row["weighted_accuracy"] = row["points_earned"] / points_possible
        out.append(row)
    return out


def to_markdown_table(rows: List[Dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider] + body)


def main():
    parser = argparse.ArgumentParser(description="Prompt-based evaluation harness for TinyStories runs")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing run folders")
    parser.add_argument("--run_names", type=str, nargs="*", default=None, help="Specific run folder names to score")
    parser.add_argument("--prompts", type=str, default="eval/prompts_tinystories.json", help="Prompt JSON file")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="comparisons/eval_harness",
        help="Directory where eval outputs are written",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Optional checkpoint step (default: latest checkpoint per run)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto/cuda/cpu")
    parser.add_argument("--max_new_tokens", type=int, default=8, help="Max generated tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=None, help="Optional top-k filtering for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    prompts = load_prompts(args.prompts)
    run_dirs = _find_run_dirs(args.runs_dir, args.run_names)
    ensure_dir(args.out_dir)

    eval_rows: List[Dict] = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        checkpoint_path = pick_checkpoint(run_dir, args.checkpoint_step)
        checkpoint_name = os.path.basename(checkpoint_path)
        model, tokenizer, seq_len, model_name = load_model_and_tokenizer(run_dir, checkpoint_path, device)
        print(f"Scoring run={run_name} model={model_name} checkpoint={checkpoint_name} prompts={len(prompts)}")

        for item in prompts:
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=item.prompt,
                device=device,
                seq_len=seq_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            passed, matched = answer_matches(completion, item.answers, item.match)
            earned = item.score if passed else 0.0
            eval_rows.append(
                {
                    "run_name": run_name,
                    "model": model_name,
                    "checkpoint": checkpoint_name,
                    "prompt_id": item.prompt_id,
                    "category": item.category,
                    "difficulty": item.difficulty,
                    "prompt": item.prompt,
                    "answers": "|".join(item.answers),
                    "match": item.match,
                    "completion": completion,
                    "matched_answer": matched or "",
                    "passed": int(passed),
                    "points_earned": float(earned),
                    "points_possible": float(item.score),
                }
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eval_rows = sort_eval_rows_by_prompt(eval_rows, prompts)
    summary_rows = build_summary(eval_rows)
    category_rows = build_breakdown(eval_rows, key="category")
    difficulty_rows = build_breakdown(eval_rows, key="difficulty")

    write_csv(
        os.path.join(args.out_dir, "eval_results.csv"),
        eval_rows,
        fieldnames=[
            "run_name",
            "model",
            "checkpoint",
            "prompt_id",
            "category",
            "difficulty",
            "prompt",
            "answers",
            "match",
            "completion",
            "matched_answer",
            "passed",
            "points_earned",
            "points_possible",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "summary_table.csv"),
        summary_rows,
        fieldnames=[
            "run_name",
            "model",
            "checkpoint",
            "prompts",
            "passed",
            "accuracy",
            "points_earned",
            "points_possible",
            "weighted_accuracy",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "category_breakdown.csv"),
        category_rows,
        fieldnames=[
            "run_name",
            "category",
            "prompts",
            "passed",
            "accuracy",
            "points_earned",
            "points_possible",
            "weighted_accuracy",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "difficulty_breakdown.csv"),
        difficulty_rows,
        fieldnames=[
            "run_name",
            "difficulty",
            "prompts",
            "passed",
            "accuracy",
            "points_earned",
            "points_possible",
            "weighted_accuracy",
        ],
    )

    markdown = to_markdown_table(
        summary_rows,
        columns=[
            "run_name",
            "model",
            "checkpoint",
            "prompts",
            "passed",
            "accuracy",
            "points_earned",
            "points_possible",
            "weighted_accuracy",
        ],
    )
    with open(os.path.join(args.out_dir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
