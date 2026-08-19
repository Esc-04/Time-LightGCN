"""
sanity_check.py — 본실험 전 필수 통과 관문
담당: 이하람

검증 내용:
  mode='mul', λ=0 일 때 mode='vanilla'와 출력이 완전히 일치하는지 확인.
  max|diff| ≈ 0 이어야 PASS.

실행:
  python sanity_check.py

PASS가 떠야 이후 실험 결과를 신뢰할 수 있습니다.
FAIL이면 model.py의 정규화(차수 재계산) 또는 λ 초기화를 다시 확인하세요.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from model import TimeLightGCN, build_edge_index


def run(data_path="data/gowalla_train.csv"):
    print(f"데이터 로딩: {data_path}")
    train = pd.read_csv(data_path)

    n_users = train["user"].max() + 1
    n_items = train["item"].max() + 1
    print(f"유저: {n_users}, 아이템: {n_items}, 상호작용: {len(train):,}")

    # 양방향 엣지 구성
    edge_index, dt_doubled = build_edge_index(
        train["user"].values,
        train["item"].values,
        n_users,
        dt_days=train["dt_days"].values,
    )

    # 같은 임베딩에서 출발하도록 시드 고정 후 두 모델 생성
    torch.manual_seed(42)
    model_vanilla = TimeLightGCN(n_users, n_items, dim=64, n_layers=3, mode="vanilla")

    torch.manual_seed(42)
    model_mul = TimeLightGCN(n_users, n_items, dim=64, n_layers=3, mode="mul")
    with torch.no_grad():
        model_mul.lam.zero_()   # λ = 0 강제 → exp(-0·dt) = 1 → vanilla와 동일

    print("\n[sanity_check] mul(λ=0) vs vanilla 비교 중...")
    with torch.no_grad():
        E_vanilla = model_vanilla.propagate(edge_index)
        E_mul     = model_mul.propagate(edge_index, dt_days=dt_doubled)

    max_diff = (E_vanilla - E_mul).abs().max().item()
    print(f"max|E_vanilla - E_mul(λ=0)| = {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✓ PASS — mul(λ=0)은 vanilla와 수치적으로 동일합니다.")
        print("  → 이후 성능 변화의 원인을 시간 정보로 귀속시킬 수 있습니다.")
        return True
    else:
        print("✗ FAIL — λ=0에서도 두 모델이 다릅니다.")
        print("  → model.py의 차수 재계산 또는 λ 초기화를 확인하세요.")
        return False


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
