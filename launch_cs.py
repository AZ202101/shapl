import argparse
import ast
import math
import numpy as np
import cvxpy as cp
from itertools import combinations

# -----------------------------
# Helpers (去节点文件化的兜底逻辑)
# -----------------------------
def _infer_n_from_vlen(v_len: int):
    """
    根据 v 的长度推断特征数 n：
    1) 如果 v_len = 2^n - 1（只含非空集合），返回 (n, False)
    2) 如果 v_len = 2^n   （含空集），返回   (n, True)
    其他情况返回 (None, None)
    """
    if v_len <= 0:
        return None, None
    # 尝试匹配 2^n-1
    n1 = int(round(math.log2(v_len + 1)))
    if (1 << n1) - 1 == v_len:
        return n1, False
    # 尝试匹配 2^n
    n2 = int(round(math.log2(v_len)))
    if (1 << n2) == v_len:
        return n2, True
    return None, None

def _all_nonempty_subsets(indices):
    """
    生成所有非空子集，按“子集大小递增 + 词典序”的确定性顺序。
    返回一个 list，每个元素是一个 list（子集中的元素升序）。
    """
    out = []
    n = len(indices)
    for r in range(1, n + 1):
        for comb in combinations(indices, r):
            out.append(list(comb))
    return out

def bernoulli_matrix(m, n):
    arr = np.random.random([m, n])
    sp = np.random.binomial(1, arr)
    sp = np.float64(sp)
    sp[sp == 1] = 1.0 / np.sqrt(m)
    sp[sp == 0] = -1.0 / np.sqrt(m)
    return sp

def random_combination(iterable, r):
    """从 itertools.combinations(iterable, r) 中随机取 1 个组合"""
    pool = tuple(iterable)
    n = len(pool)
    # 从 C(n, r) 均匀随机：这里简化为随机 r 个下标再排序
    idx = sorted(np.random.choice(np.arange(n), size=r, replace=False))
    return tuple(pool[i] for i in idx)

# -----------------------------
# 主流程（去 nodes 依赖版）
# -----------------------------
def run_cs(v, args):
    """
    论文 QR-CS 主流程，去掉 nodes/* 文件依赖：
    - 自动从 v 的长度推断 n
    - 内存中生成 m_combs（非空子集，确定性顺序）
    - 其它保持与原实现一致
    """
    v = list(v)
    v_len = len(v)

    # 1) 根据 v 的长度推断特征数 n，以及 v 是否包含空集项
    n, has_empty = _infer_n_from_vlen(v_len)
    if n is None:
        raise ValueError(
            f"[launch_cs] 无法从 v 的长度({v_len})推断特征数 n。"
            f"期望 len(v) = 2^n-1（仅非空集合）或 2^n（含空集）。"
        )

    # 如果 v 含空集，则忽略 v[0]（约定空集在最前）
    if has_empty:
        v_nonempty = v[1:]
    else:
        v_nonempty = v

    # 2) 构造所有非空联盟（确定性顺序）
    ind_EV_loc = list(range(n))                     # 特征索引 0..n-1
    m_combs = _all_nonempty_subsets(ind_EV_loc)     # 长度应为 2^n - 1
    if len(m_combs) != len(v_nonempty):
        # 长度不一致时，取两者最小长度对齐，避免越界（给出提示）
        L = min(len(m_combs), len(v_nonempty))
        m_combs = m_combs[:L]
        v_nonempty = v_nonempty[:L]
        print(f"[launch_cs][WARN] |m_combs|({len(m_combs)}) 与 |v|({len(v_nonempty)}) 不一致，已截断对齐。")

    # 3) 读取参数
    m = int(getattr(args, "m", 100))   # CS 测量条数
    t = int(getattr(args, "t", 50))    # 迭代次数
    # g/c 参数与 nodes 相关，此处保留字段但不再使用
    # np.random.seed 固定以可复现（如需）
    np.random.seed(19365)

    # 为 utility 做一个 “子集→索引” 的查找表（避免频繁线性搜索）
    # key 统一用 tuple(sorted(subset))
    subset_to_idx = {tuple(s): i for i, s in enumerate(m_combs)}

    requests = []  # 记录调用过的子集（与原实现一致）

    def utility(coalition, m_combs_local):
        if len(coalition) == 0:
            return 0.0
        coal = tuple(sorted(coalition))
        requests.append(np.array(coal))
        idx = subset_to_idx.get(coal, None)
        if idx is None:
            # 正常不会发生；如发生，返回 0 并报警
            # 这样可避免越界，同时不中断流程
            # 你也可以改成 raise
            # raise KeyError(f"coalition {coal} not found in m_combs.")
            return 0.0
        return float(v_nonempty[idx])

    # 4) 与原流程一致：随机行、构造 φ 的测量、聚合 y、ℓ1 最优化重构
    A = bernoulli_matrix(m, n)
    y = {}

    for cur_t in range(t):
        # 在 [1..n] 之间随机一个子集大小，然后随机取该大小的组合
        row_idx = np.array(random_combination(ind_EV_loc, np.random.randint(1, n + 1)))

        phi_arr = []
        for i in ind_EV_loc:
            permutation = row_idx.copy()
            if i in permutation:
                permutation_w_i = permutation.copy()
                permutation_wo_i = permutation[permutation != i]
            else:
                permutation_wo_i = permutation.copy()
                permutation_w_i = np.append(permutation, i)

            u1 = utility(permutation_w_i, m_combs)
            u2 = utility(permutation_wo_i, m_combs)
            phi = u1 - u2
            phi_arr.append(phi)

        phi_arr = np.array(phi_arr, dtype=float)

        # y_m = A * phi_arr
        y_m = A @ phi_arr
        y[cur_t] = y_m

    y_bar = np.sum(np.array(list(y.values())).T, axis=1) * (1.0 / t)
    # s_bar：全体子集的平均值（论文原式：utility(ind_EV_loc)/n）
    s_bar = utility(ind_EV_loc, m_combs) * (1.0 / n)

    # L1 重构
    eps = 1e-3
    x_l1 = cp.Variable(shape=(n, 1))
    constraints = [cp.norm(A @ (s_bar + x_l1) - y_bar[:, np.newaxis]) <= eps]
    obj = cp.Minimize(cp.norm(x_l1, 1))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    CS_res = s_bar + x_l1.value  # 形状 (n,1)

    # 去重统计（与原实现一致，非必要但保留）
    result = []
    for arr in requests:
        if not any(np.array_equal(arr, c) for c in result):
            result.append(arr)

    print(f"Used size:{len(result)}")
    print(f"Total size:{len(v_nonempty)}")

    # —— 关键：模块调用返回一维 —— #
    return np.asarray(CS_res).reshape(-1)

# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the file (.txt is accepted)', required=True)
    parser.add_argument('--g', help='Grid id', required=True)           # 保留字段，不再强依赖外部文件
    parser.add_argument('--c', help='Case id', required=True)           # 保留字段，不再强依赖外部文件
    parser.add_argument('--m', help='Number of CS measurements', default=100, required=False)
    parser.add_argument('--t', help='Number of CS iterations', default=50, required=False)
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        v = ast.literal_eval(lines[0])

    CS_res = run_cs(v, args)
    res = np.asarray(CS_res).reshape(-1)
    np.set_printoptions(linewidth=np.inf, precision=4)
    print(str(f"CS_res:{res}"))
