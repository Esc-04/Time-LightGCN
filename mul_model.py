"""
=== 사용법 ===

1. 데이터 로딩
    import pandas as pd
    train = pd.read_csv("gowalla_train.csv")  # user, item, dt_days 컬럼 필요
    n_users = train["user"].max() + 1
    n_items = train["item"].max() + 1

2. 엣지 구성 (dt_days 반드시 넣기)
    edge_index, dt_doubled = build_edge_index(
        train["user"].values,
        train["item"].values,
        n_users,
        dt_days=train["dt_days"].values  # 없으면 에러 남
    )

3. 모델 선언 및 전파
    model = TimeLightGCN(n_users, n_items, mode="mul")
    E = model.propagate(edge_index, dt_days=dt_doubled)

4. 스코어 계산
    scores = model.score(E, u_idx, i_idx)

=== 유의사항 ===

- mode="mul" 로 설정해야 이 모델이 작동합니다.
- dt_days는 gowalla_train.csv의 dt_days 컬럼을 그대로 넣으면 됩니다.
- dt_days를 넣지 않으면 에러가 납니다.
- 이 파일은 유저별 min-max 버전입니다 (add 모델과 동일한 시간 기준).
  전체 min-max 버전은 model_mul_A.py를 사용하세요.
- vanilla 모드는 dt_days 없이 돌아갑니다.
  model = TimeLightGCN(n_users, n_items, mode="vanilla")
  E = model.propagate(edge_index)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class TimeLightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3, mode="vanilla"):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.mode = mode

        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        if mode == "mul":
            self.lam = nn.Parameter(torch.zeros(1))

    def _compute_edge_weight(self, dt_norm):
        return torch.exp(-self.lam * dt_norm)

    def propagate(self, edge_index, dt_days=None):
        row, col = edge_index[0], edge_index[1]
        N = self.n_users + self.n_items

        if self.mode == "vanilla":
            edge_w = torch.ones(row.shape[0], device=row.device)
        elif self.mode == "mul":
            edge_w = self._compute_edge_weight(dt_days.to(row.device))
        elif self.mode == "add":
            raise NotImplementedError()

        deg = torch.zeros(N, device=row.device)
        deg.scatter_add_(0, row, edge_w)
        deg = deg.clamp(min=1e-12)
        norm = edge_w / (deg[row].sqrt() * deg[col].sqrt())

        E0 = self.embedding.weight
        E = E0
        layers = [E0]

        for _ in range(self.n_layers):
            msg = norm.unsqueeze(1) * E[col]
            E_next = torch.zeros_like(E)
            E_next.scatter_add_(0, row.unsqueeze(1).expand(-1, self.dim), msg)
            E = E_next
            layers.append(E)

        E_final = torch.stack(layers, dim=0).mean(dim=0)
        return E_final

    def score(self, E_final, u_idx, i_idx):
        u_emb = E_final[u_idx]
        i_emb = E_final[self.n_users + i_idx]
        return (u_emb * i_emb).sum(dim=-1)


def bpr_loss(pos_scores, neg_scores, l2_reg=0.0, emb_params=None):
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    if l2_reg > 0 and emb_params is not None:
        reg = sum((p ** 2).sum() for p in emb_params)
        loss = loss + l2_reg * reg
    return loss


def build_edge_index(train_u, train_i, n_users, dt_days=None, device="cpu"):
    u = torch.as_tensor(train_u, dtype=torch.long, device=device)
    i = torch.as_tensor(train_i, dtype=torch.long, device=device) + n_users

    row = torch.cat([u, i])
    col = torch.cat([i, u])
    edge_index = torch.stack([row, col], dim=0)

    if dt_days is not None:
        df = pd.DataFrame({"user": train_u, "dt": dt_days})
        df["dt_norm"] = df.groupby("user")["dt"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )
        dt_norm = df["dt_norm"].values.astype(np.float32)
        dt = torch.as_tensor(dt_norm, dtype=torch.float32, device=device)
        dt_doubled = torch.cat([dt, dt])
    else:
        dt_doubled = None

    return edge_index, dt_doubled
