"""
model.py — Time-LightGCN, mul 모드 구현
담당: 이하람 (mul 분기 + sanity check 기준선 vanilla)
"""

import torch
import torch.nn as nn


class TimeLightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3, mode="vanilla"):
        """
        n_users : 유저 수 (0 ~ n_users-1)
        n_items : 아이템 수 (0 ~ n_items-1)
        dim     : 임베딩 차원 (기본 64, LightGCN 원논문과 동일)
        n_layers: 전파 홉 수 K (기본 3)
        mode    : 'vanilla' / 'mul' / 'add'
        """
        super().__init__()
        assert mode in ("vanilla", "mul", "add"), f"알 수 없는 mode: {mode}"

        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.mode = mode

        # 학습 파라미터: 임베딩만. 전파 자체는 parameter-free (LightGCN 철학 유지)
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        if mode == "mul":
            # λ: 시간 감쇠 속도. 초기값 0 → 학습 시작점 = vanilla.
            # 부호 제약 없음 (nn.Parameter 그대로, softplus 등 사용 안 함).
            self.lam = nn.Parameter(torch.zeros(1))

        if mode == "add":
            pass

    def _compute_edge_weight(self, dt_days):
        """
        mul 모드 전용. dt_days → 엣지별 스칼라 가중치 α.

        dt_days: (E,) FloatTensor. 일 단위 Δt. 작을수록 최근.
                 조은서 데이터 기준 0 ~ 585 범위.
        반환   : (E,) α = exp(-λ · dt_days)
                 λ=0이면 α=1 → vanilla와 동일.
                 λ>0이면 오래된 엣지(dt 큰 것)의 α가 작아짐.
        """
        return torch.exp(-self.lam * dt_days)

    def propagate(self, edge_index, dt_days=None):
        """
        LightGCN 전파. mode에 따라 엣지 가중치만 달라지고
        나머지 구조(층 결합, 스코어 방식)는 완전히 동일.

        edge_index : (2, 2E) LongTensor. 양방향 엣지.
                     행 0 = row(메시지 받는 노드),
                     행 1 = col(메시지 보내는 노드).
                     아이템 노드 인덱스 = item_id + n_users.
        dt_days    : (2E,) FloatTensor. mul 모드에서만 사용.
                     양방향 엣지에 맞게 복제된 상태로 넘겨야 함.
        """
        row, col = edge_index[0], edge_index[1]
        N = self.n_users + self.n_items

        # ── 모드별 엣지 가중치 ──────────────────────────────
        if self.mode == "vanilla":
            edge_w = torch.ones(row.shape[0], device=row.device)

        elif self.mode == "mul":
            assert dt_days is not None, \
                "mul 모드는 dt_days가 필요합니다. propagate(edge_index, dt_days=...) 로 호출하세요."
            edge_w = self._compute_edge_weight(dt_days.to(row.device))

        elif self.mode == "add":
            raise NotImplementedError(
            )

        # ── 정규화: 가중치 곱한 뒤 차수 재계산 (핵심 버그 방지) ─
        # vanilla에서 1/√(d_u · d_i) 정규화를 그대로 유지하되,
        # mul에서는 가중 차수 d'_u = Σ_i w_ui 로 다시 계산.
        deg = torch.zeros(N, device=row.device)
        deg.scatter_add_(0, row, edge_w)
        deg = deg.clamp(min=1e-12)
        norm = edge_w / (deg[row].sqrt() * deg[col].sqrt())

        # ── K층 선형 전파 ─────────────────────────────────
        E0 = self.embedding.weight          # (N, dim)
        E = E0
        layers = [E0]

        for _ in range(self.n_layers):
            msg = norm.unsqueeze(1) * E[col]           # (2E, dim)
            E_next = torch.zeros_like(E)
            E_next.scatter_add_(0, row.unsqueeze(1).expand(-1, self.dim), msg)
            E = E_next
            layers.append(E)

        # 층 결합: 균등 평균 α_k = 1/(K+1), LightGCN 원논문 기본값
        E_final = torch.stack(layers, dim=0).mean(dim=0)   # (N, dim)
        return E_final

    def score(self, E_final, u_idx, i_idx):
        """
        유저-아이템 내적 스코어. BPR 손실 계산에 사용.
        u_idx: (B,) 유저 인덱스
        i_idx: (B,) 아이템 인덱스 (n_users offset 없이 0~n_items-1)
        """
        u_emb = E_final[u_idx]
        i_emb = E_final[self.n_users + i_idx]
        return (u_emb * i_emb).sum(dim=-1)


def bpr_loss(pos_scores, neg_scores, l2_reg=0.0, emb_params=None):
    """
    BPR 손실. L2 정규화는 임베딩 E^(0)에만 적용.
    λ나 게이트 파라미터에 L2 걸면 λ가 0으로 끌려가므로 주의.
    """
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    if l2_reg > 0 and emb_params is not None:
        reg = sum((p ** 2).sum() for p in emb_params)
        loss = loss + l2_reg * reg
    return loss


def build_edge_index(train_u, train_i, n_users, dt_days=None, device="cpu"):
    """
    양방향 엣지 인덱스 생성.
    user→item, item→user 두 방향을 모두 포함.
    아이템 노드 인덱스 = item_id + n_users.

    반환:
      edge_index   : (2, 2E) LongTensor
      dt_doubled   : (2E,) FloatTensor or None. dt_days를 양방향으로 복제.
    """
    u = torch.as_tensor(train_u, dtype=torch.long, device=device)
    i = torch.as_tensor(train_i, dtype=torch.long, device=device) + n_users

    row = torch.cat([u, i])
    col = torch.cat([i, u])
    edge_index = torch.stack([row, col], dim=0)

    if dt_days is not None:
        dt = torch.as_tensor(dt_days, dtype=torch.float32, device=device)
        dt_doubled = torch.cat([dt, dt])
    else:
        dt_doubled = None

    return edge_index, dt_doubled
