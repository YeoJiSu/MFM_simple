# train.py
# -*- coding: utf-8 -*-
import os
import math
import random
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 프로젝트 모듈 ---
from data.dataset import make_concat_dataloader, make_dataloader
from model.img_encoder import VisionEncoderCLIP
from model.llm_encoder import LlamaEncoder, MultiModalProjector, build_tokenizer
from model.llm_decoder import LlamaDecoderWithVisionPrefix

# -----------------------
# 유틸
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic 모드가 느릴 수 있으니 필요 시만 켜세요.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

@dataclass
class PromptTemplate:
    system: str = "You are an industrial anomaly inspection assistant."
    user_normal: str = "The following image is from {dataset}/{category}. Describe whether it is normal or anomalous and why."
    user_anom: str = "The following image is from {dataset}/{category}. Describe the defect type and likely location succinctly."
    # 간단한 타겟 문장들 (데모용)
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
    # 간단한 대화 포맷 (LLM 프롬프팅 규칙은 자유롭게 변경 가능)
    prompt = f"<s>[SYSTEM]\n{sys_prompt}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"
    return prompt, target

def tokenize_prompts_and_targets(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    max_length: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    prompt + target 을 직접 이어붙이고,
    라벨은 prompt 부분을 -100으로 마스킹하여 LM 학습 시 target만 로스에 반영.
    """
    input_id_batches = []
    attn_batches = []
    label_batches = []

    for p, t in zip(prompts, targets):
        pt = tokenizer(p, add_special_tokens=False)
        tt = tokenizer(t, add_special_tokens=False)
        # [prompt tokens] + [target tokens] + eos
        input_ids = pt["input_ids"] + tt["input_ids"] + [tokenizer.eos_token_id]
        # 길이 제한
        input_ids = input_ids[:max_length]
        # 어텐션 마스크
        attn = [1] * len(input_ids)
        # 라벨: prompt 구간은 -100, target 구간은 실제 토큰 id
        prompt_len = min(len(pt["input_ids"]), len(input_ids))  # 잘린 경우 안전 처리
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        # 패딩
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

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# 학습 루프
# -----------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- DataLoader -----
    if args.use_concat_loader:
        roots = {}
        if args.mvtec_root: roots["mvtec"] = args.mvtec_root
        if args.visa_root: roots["visa"] = args.visa_root
        if args.mvtec_loco_root: roots["mvtec_loco"] = args.mvtec_loco_root
        if args.goodsad_root: roots["goodsad"] = args.goodsad_root
        if not roots:
            raise ValueError("No dataset roots provided. Set at least one of --mvtec_root/--visa_root/--mvtec_loco_root/--goodsad_root")
        train_loader = make_concat_dataloader(
            roots=roots,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            return_mask=False,
        )
    else:
        # 단일 데이터셋 학습을 원하는 경우
        if not args.single_root or not args.single_name:
            raise ValueError("For single loader mode, provide --single_root and --single_name (e.g., mvtec, visa, mvtec_loco, goodsad)")
        train_loader = make_dataloader(
            root=args.single_root,
            dataset_name=args.single_name,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            return_mask=False,
        )

    # ----- Models -----
    # Vision encoder (CLIP)
    vision = VisionEncoderCLIP(
        model_name=args.clip_name,
        pretrained=args.clip_pretrained,
        freeze=not args.unfreeze_vision,
        device=device,
    )

    # LLM encoder(크기/토크나이저 확인용) + Projector
    llm_enc = LlamaEncoder(pretrained_name=args.llm_name, device=device, freeze=True)
    projector = MultiModalProjector(
        vision_dim=vision.get_output_dim(),
        llm_dim=llm_enc.hidden_size,
        num_vision_tokens=args.num_vision_tokens,
        hidden_dim=args.projector_hidden if args.projector_hidden > 0 else None,
        use_ln=True,
    ).to(device)

    # LLM decoder with vision prefix
    llm_dec = LlamaDecoderWithVisionPrefix(pretrained_name=args.llm_name, device=device, freeze_lm=not args.unfreeze_llm)

    # Tokenizer (pad 토큰 자동 세팅)
    tokenizer = build_tokenizer(args.llm_name, use_fast=True)

    # ----- Optimizer -----
    # 기본: projector만 학습. 필요 시 vision/llm 파라미터도 추가
    params = []
    params += list(p for p in projector.parameters() if p.requires_grad)
    if args.unfreeze_vision:
        params += list(p for p in vision.parameters() if p.requires_grad)
    if args.unfreeze_llm:
        params += list(p for p in llm_dec.parameters() if p.requires_grad)

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    total_trainable = count_trainable_params(projector) \
        + (count_trainable_params(vision) if args.unfreeze_vision else 0) \
        + (count_trainable_params(llm_dec) if args.unfreeze_llm else 0)

    print(f"[Info] Device: {device}")
    print(f"[Info] Trainable params (total): {total_trainable/1e6:.2f} M")
    print(f"[Info] Projector params: {count_trainable_params(projector)/1e6:.2f} M")
    if args.unfreeze_vision:
        print(f"[Info] Vision params (trainable): {count_trainable_params(vision)/1e6:.2f} M")
    if args.unfreeze_llm:
        print(f"[Info] LLM-decoder params (trainable): {count_trainable_params(llm_dec)/1e6:.2f} M")

    # ----- Training -----
    tmpl = PromptTemplate()
    global_step = 0
    best_loss = float("inf")

    # JISU ADDED: Train Log CSV
    log_path = os.path.join(args.output_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,avg_train_loss\n")

    """
    중간에 학습하다가 서버가 끊겨서, 모델 다시 이어서 학습시킬려고 작성한 코드입니다.
    """
     # ----- Resume -----
    start_epoch = 7 # 미지 
    best_loss = _maybe_load_best_loss(args.output_dir)  # 기존 best.pt 기준
    if args.resume_from and os.path.isfile(args.resume_from):
        print(f"[Resume] loading: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu")

        # 모델 가중치
        projector.load_state_dict(ckpt["projector"], strict=True)
        if "vision" in ckpt:
            try:
                vision.load_state_dict(ckpt["vision"], strict=False)
            except Exception as e:
                print(f"[warn] vision load skipped: {e}")
        if args.unfreeze_llm and "llm_decoder" in ckpt:
            try:
                llm_dec.load_state_dict(ckpt["llm_decoder"], strict=False)
            except Exception as e:
                print(f"[warn] llm_decoder load skipped: {e}")

        # 옵티마이저 (파라미터 구성이 같아야 함)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            _optimizer_to(optimizer, device)
        except Exception as e:
            print(f"[warn] optimizer state not loaded: {e} (파라미터 구성이 달라졌을 수 있음)")

        # 다음 에폭부터 재개
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Resume] start_epoch = {start_epoch}, prev_avg_loss = {ckpt.get('avg_loss','?')}")

    for epoch in range(start_epoch, args.epochs + 1):
        vision.train(args.unfreeze_vision)
        projector.train()
        llm_dec.train(args.unfreeze_llm)

        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)
            labels = batch["label"]                                 # (B,)
            metas = batch["meta"]

            # 1) 비전 임베딩
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                vision_feat = vision(images)  # (B, Dv)
                vision_tokens = projector(vision_feat)  # (B, V, H_llm)

            # 2) 프롬프트/타겟 구성
            prompts, targets = [], []
            for j in range(images.size(0)):
                p, t = build_text_example(metas["dataset"][j] if isinstance(metas, dict) and "dataset" in metas else metas[j],
                                          int(labels[j].item()), tmpl)
                # 위의 라인은 dataset/key 접근의 구조 차이를 흡수하기 위한 방어적 코드
                # dataset.py의 meta는 dict of lists가 아니라 list of dict 구조임
                if isinstance(metas, dict):
                    # DataLoader default collate가 dict of lists로 만들었을 수 있어 방어
                    meta_j = {k: metas[k][j] for k in metas}
                else:
                    meta_j = metas[j]
                p, t = build_text_example(meta_j, int(labels[j].item()), tmpl)
                prompts.append(p)
                targets.append(t)

            tok = tokenize_prompts_and_targets(
                tokenizer,
                prompts=prompts,
                targets=targets,
                max_length=args.max_length,
                device=device
            )

            # 3) LLM forward
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                out = llm_dec(
                    vision_prefix=vision_tokens,
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    labels=tok["labels"],   # prompt 부분 -100, target 부분 token id
                )
                loss = out["loss"]

            # 4) Backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if (i + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            else:
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            running += loss.item()
            avg = running / (i + 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

        # ----- Epoch 끝: 체크포인트 -----
        avg_epoch = running / max(1, len(train_loader))
        print(f"[Epoch {epoch}] avg_loss = {avg_epoch:.15f}")

        # JISU ADDED: Save train log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_epoch:.15f}\n")
        ckpt = {
            "epoch": epoch,
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "avg_loss": avg_epoch,
        }
        if args.save_vision and not args.unfreeze_vision:
            # 동결이라도 저장을 원할 수 있음(재현)
            ckpt["vision"] = vision.state_dict()
        if args.unfreeze_vision:
            ckpt["vision"] = vision.state_dict()
        if args.unfreeze_llm:
            ckpt["llm_decoder"] = llm_dec.state_dict()

        torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"))

        if avg_epoch < best_loss:
            best_loss = avg_epoch
            torch.save(ckpt, os.path.join(args.output_dir, f"best.pt"))
            print(f"[Info] Best ckpt updated: {best_loss:.4f}")

    print("[Done] Training completed.")

# Utils for checkpoint loading
def _maybe_load_best_loss(output_dir: str) -> float:
    best_path = os.path.join(output_dir, "best.pt")
    if os.path.isfile(best_path):
        try:
            ck = torch.load(best_path, map_location="cpu")
            return float(ck.get("avg_loss", float("inf")))
        except Exception:
            pass
    return float("inf")

def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device):
    for st in optimizer.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)



# -----------------------
# 아규먼트
# -----------------------
def build_parser():
    p = argparse.ArgumentParser()
    # 데이터 경로
    p.add_argument("--mvtec_root", type=str, default="", help="Path to MVTec AD root")
    p.add_argument("--visa_root", type=str, default="", help="Path to VisA root")
    p.add_argument("--mvtec_loco_root", type=str, default="", help="Path to MVTec LOCO AD root")
    p.add_argument("--goodsad_root", type=str, default="", help="Path to GoodsAD root")
    p.add_argument("--use_concat_loader", action="store_true", help="Use concatenated loader over all provided roots")

    # 단일 데이터셋 모드 (use_concat_loader=False일 때)
    p.add_argument("--single_root", type=str, default="")
    p.add_argument("--single_name", type=str, default="", choices=["mvtec", "visa", "mvtec_loco", "goodsad"])

    # 모델/토크나이저
    p.add_argument("--clip_name", type=str, default="ViT-L-14")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--llm_name", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--num_vision_tokens", type=int, default=32)
    p.add_argument("--projector_hidden", type=int, default=0, help="0이면 자동(max(vision, llm))")

    # 학습
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")

    # 언프리즈 옵션
    p.add_argument("--unfreeze_vision", action="store_true")
    p.add_argument("--unfreeze_llm", action="store_true")
    p.add_argument("--save_vision", action="store_true", help="동결이어도 비전 상태를 체크포인트에 포함")

    # resume 옵션 - 추가 학습할 때 사용하자.
    p.add_argument("--resume_from", type=str, default="", help="epoch_xxx.pt 체크포인트 경로")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
