# eval.py
# -*- coding: utf-8 -*-

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import make_dataloader
from model.img_encoder import VisionEncoderCLIP
from model.llm_encoder import LlamaEncoder, MultiModalProjector, build_tokenizer
from model.llm_decoder import LlamaDecoderWithVisionPrefix


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PromptTemplate:
    system: str = "You are an industrial anomaly inspection assistant."
    user_normal: str = "The following image is from {dataset}/{category}. Describe whether it is normal or anomalous and why."
    user_anom: str = "The following image is from {dataset}/{category}. Describe the defect type and likely location succinctly."
    target_normal: str = "Prediction: normal. Rationale: no visible defects."
    target_anom: str = "Prediction: anomalous. Rationale: visible defect patterns present."


def build_text_example(meta: Dict[str, Any], label: int, tmpl: PromptTemplate) -> Tuple[str, str]:
    dataset = meta.get("dataset", "unknown")
    category = meta.get("category", "object")
    sys_prompt = tmpl.system
    if label == 0:
        user = tmpl.user_normal.format(dataset=dataset, category=category)
        target = tmpl.target_normal
    else:
        user = tmpl.user_anom.format(dataset=dataset, category=category)
        target = tmpl.target_anom
    prompt = f"<s>[SYSTEM]\n{sys_prompt}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"
    return prompt, target


def tokenize_prompts_and_targets(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    max_length: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    input_id_batches = []
    attn_batches = []
    label_batches = []

    for p, t in zip(prompts, targets):
        pt = tokenizer(p, add_special_tokens=False)
        tt = tokenizer(t, add_special_tokens=False)
        input_ids = pt["input_ids"] + tt["input_ids"] + [tokenizer.eos_token_id]
        input_ids = input_ids[:max_length]
        attn = [1] * len(input_ids)

        prompt_len = min(len(pt["input_ids"]), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            attn += [0] * pad_len
            labels += [-100] * pad_len

        input_id_batches.append(input_ids)
        attn_batches.append(attn)
        label_batches.append(labels)

    input_ids = torch.tensor(input_id_batches, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_batches, dtype=torch.long, device=device)
    labels = torch.tensor(label_batches, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def compute_target_loss(
    llm_dec: LlamaDecoderWithVisionPrefix,
    tokenizer,
    vision_tokens: torch.Tensor,          # (1, V, H)
    prompt: str,
    target: str,
    max_length: int,
    device: torch.device
) -> float:
    tok = tokenize_prompts_and_targets(
        tokenizer,
        prompts=[prompt],
        targets=[target],
        max_length=max_length,
        device=device,
    )
    out = llm_dec(
        vision_prefix=vision_tokens,
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
        labels=tok["labels"],
    )
    # HF loss는 평균 cross-entropy
    return float(out["loss"].item())


def build_parser():
    p = argparse.ArgumentParser()
    # 데이터 설정 (train.py와 동일 패턴)
    p.add_argument("--mvtec_root", type=str, default="")
    p.add_argument("--visa_root", type=str, default="")
    p.add_argument("--mvtec_loco_root", type=str, default="")
    p.add_argument("--goodsad_root", type=str, default="")

    p.add_argument("--single_root", type=str, default="")
    p.add_argument("--single_name", type=str, default="", choices=["mvtec", "visa", "mvtec_loco", "goodsad"])

    # 모델 설정
    p.add_argument("--clip_name", type=str, default="ViT-L-14")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--llm_name", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--num_vision_tokens", type=int, default=32)
    p.add_argument("--projector_hidden", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_length", type=int, default=384)

    # 평가
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--ckpt_path", type=str, required=True, help="Path to best.pt or epoch_xxx.pt")
    p.add_argument("--output_dir", type=str, default="outputs_eval")
    return p


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- DataLoader (test split) -----
    if args.single_root and args.single_name:
        test_loader: DataLoader = make_dataloader(
            root=args.single_root,
            dataset_name=args.single_name,
            split="test",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            return_mask=False,
            shuffle=False,
        )
    elif args.visa_root:
        # VisA만 평가하는 경우 (concat이 아니라 single_root를 사용하는 게 편함)
        test_loader = make_dataloader(
            root=args.visa_root,
            dataset_name="visa",
            split="test",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            return_mask=False,
            shuffle=False,
        )
    else:
        raise ValueError("평가할 데이터셋 root를 하나는 지정해야 합니다. (--single_root/--single_name 또는 --visa_root 등)")

    # ----- Models -----
    vision = VisionEncoderCLIP(
        model_name=args.clip_name,
        pretrained=args.clip_pretrained,
        freeze=True,           # 평가 시에는 freeze
        device=device,
    )

    llm_enc = LlamaEncoder(pretrained_name=args.llm_name, device=device, freeze=True)
    projector = MultiModalProjector(
        vision_dim=vision.get_output_dim(),
        llm_dim=llm_enc.hidden_size,
        num_vision_tokens=args.num_vision_tokens,
        hidden_dim=args.projector_hidden if args.projector_hidden > 0 else None,
        use_ln=True,
    ).to(device)

    llm_dec = LlamaDecoderWithVisionPrefix(
        pretrained_name=args.llm_name,
        device=device,
        freeze_lm=True,   # 평가용: gradient 불필요
    )

    tokenizer = build_tokenizer(args.llm_name, use_fast=True)

    # ----- Load checkpoint -----
    print(f"[Info] Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")

    projector.load_state_dict(ckpt["projector"])
    if "vision" in ckpt:
        vision.load_state_dict(ckpt["vision"])
    if "llm_decoder" in ckpt:
        # 양자화/장치 때문에 strict=False가 더 안전할 수 있음
        llm_dec.llm.load_state_dict(ckpt["llm_decoder"], strict=False)

    vision.eval()
    projector.eval()
    llm_dec.eval()

    tmpl = PromptTemplate()

    total = 0
    correct = 0

    pbar = tqdm(test_loader, desc="Eval", dynamic_ncols=True)

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)
            labels = batch["label"]                                 # (B,)
            metas = batch["meta"]                                   # List[dict]

            # 1) vision prefix
            vision_feat = vision(images)                # (B, Dv)
            vision_tokens = projector(vision_feat)      # (B, V, H)

            # 🔧 LLM의 토큰 임베딩 기준으로 정렬 - RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
            vision_tokens = vision_tokens.to(
                device=llm_dec.token_embed.weight.device,
                dtype=llm_dec.token_embed.weight.dtype,
            )

            B = images.size(0)
            for j in range(B):
                meta_j = metas[j]
                y_true = int(labels[j].item())

                # 각 샘플에 대해 prefix 1개와 두 후보 타깃(normal / anomaly)
                p_norm, t_norm = build_text_example(meta_j, 0, tmpl)
                p_anom, t_anom = build_text_example(meta_j, 1, tmpl)

                vt = vision_tokens[j:j+1]  # (1, V, H)

                loss_norm = compute_target_loss(
                    llm_dec, tokenizer, vt, p_norm, t_norm,
                    max_length=args.max_length, device=device
                )
                loss_anom = compute_target_loss(
                    llm_dec, tokenizer, vt, p_anom, t_anom,
                    max_length=args.max_length, device=device
                )

                y_pred = 0 if loss_norm <= loss_anom else 1

                total += 1
                if y_pred == y_true:
                    correct += 1

            acc = correct / max(1, total)
            pbar.set_postfix(acc=f"{acc*100:.2f}%")

    final_acc = correct / max(1, total)
    print(f"[Eval] total={total}, correct={correct}, accuracy={final_acc*100:.2f}%")

    # 결과 저장
    out_path = os.path.join(args.output_dir, "eval_results.txt")
    with open(out_path, "w") as f:
        f.write(f"total={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"accuracy={final_acc:.6f}\n")

    print(f"[Eval] Saved metrics to {out_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
