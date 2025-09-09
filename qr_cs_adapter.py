# qr_cs_adapter.py
import os
import sys
import re
import json
import math
import tempfile
import itertools
import subprocess
import numpy as np

__version__ = "2025-09-09-r2"

# -------------------------------
# 基础：构造 v（标准顺序，按子集大小递增；只取非空子集）
# -------------------------------

def _all_subsets_indices(d):
    subs = []
    for r in range(0, d + 1):
        for comb in itertools.combinations(range(d), r):
            subs.append(list(comb))
    return subs

def _build_v_by_eval_fn(feature_names, eval_fn):
    """
    用 eval_fn 按子集大小从小到大构造联盟值向量 v（只含非空子集；做 f(S)-f(∅) 归一化）
    返回：长度 2^d - 1 的 list[float]
    """
    d = len(feature_names)
    all_sets = _all_subsets_indices(d)
    f_empty = float(eval_fn([]))  # 空集
    v = []
    for idx_set in all_sets[1:]:  # 跳过空集
        S = [feature_names[i] for i in idx_set]
        v.append(float(eval_fn(S)) - f_empty)
    return v

# -------------------------------
# 工具：把内核的各种返回（dict/list/ndarray/文本）统一成长度 d 的 1D 向量
# -------------------------------

def _normalize_phi(raw, d):
    """
    支持四类返回：
      1) dict: 优先取 'phi' / 'CS_res' / 'result' / 'values' 等键
      2) str:  从 "CS_res:[...]" 或数字串里抓数，尽量取末尾 d 个
      3) list/tuple/ndarray: 直接转 np.array 并 reshape/截断/补零
      4) 其他：尽力 array 化
    """
    # 情况 1：dict
    if isinstance(raw, dict):
        for k in ('phi', 'CS_res', 'result', 'values'):
            if k in raw:
                raw = raw[k]
                break

    # 情况 2：str（CLI 输出）
    if isinstance(raw, str):
        # 先尝试 "CS_res:[...]" 的明确格式
        mobj = re.search(r"CS_res:\s*\[([^\]]+)\]", raw)
        if mobj:
            arr = np.fromstring(mobj.group(1), sep=',', dtype=float)
            return _normalize_phi(arr, d)

        # 退而求其次：抓取所有数字，尽量拿末尾 d 个（规避 g/c/t 混入）
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
        arr = np.array([float(x) for x in nums], dtype=float)
        if arr.size >= d:
            arr = arr[-d:]
        # 不够则在后面补零
        if arr.size < d:
            arr = np.pad(arr, (0, d - arr.size), mode='constant')
        return arr

    # 情况 3/4：可数组化
    arr = np.array(raw, dtype=float).reshape(-1)

    # 截断/补零到 d
    if arr.size > d:
        arr = arr[:d]
    elif arr.size < d:
        arr = np.pad(arr, (0, d - arr.size), mode='constant')
    return arr

# -------------------------------
# 内核 CLI 路径（兜底用）
# -------------------------------

def _run_core_cli(v, args):
    """
    通过子进程调用内核脚本（CLI）。
    传入：v(list[float])，args(dict: m/t/g/c)
    返回：stdout 文本
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "launch_cs.py"),
        os.path.join(this_dir, "launch_cs - 使用QR-CS.py"),
    ]
    core_py = None
    for c in candidates:
        if os.path.isfile(c):
            core_py = c
            break
    if core_py is None:
        raise RuntimeError(f"未找到内核脚本：{candidates}")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "v.json")
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(v, f)

        cmd = [
            sys.executable, core_py,
            "--input", in_path,
            "--m", str(args['m']),
            "--t", str(args['t']),
            "--g", str(args['g']),
            "--c", str(args['c']),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 如需调试可解注：print("[core stderr]", p.stderr)
        return p.stdout

# -------------------------------
# 主函数：qr_cs（单块）
# -------------------------------

def qr_cs(X, y, eval_fn, max_d_for_cs=20, m=200, extra_args=None):
    """
    单块 QR-CS：对 X 的列名（例如 top-k）构造 v 并调用内核求 φ
    必传：eval_fn(S_by_names) -> float（KPI）
    """
    # 取特征名
    feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
    d = len(feature_names)
    if d > max_d_for_cs:
        raise ValueError(f"d={d} 超过 max_d_for_cs={max_d_for_cs}。")

    # 1) 构造 v（只含非空子集）
    if eval_fn is None:
        raise ValueError("必须提供 eval_fn(S_by_names)->float 以构造 v(S)。")
    expected = (1 << d) - 1
    print(f"[adapter] building v... d={d}, expect |v|={expected}")
    v = _build_v_by_eval_fn(feature_names, eval_fn)   # 长度 2^d - 1

    if len(v) != expected:
        raise ValueError(f"v 大小异常: got {len(v)}, expect {expected} (=2^d-1).")
    print(f"[adapter] built v: |v|={len(v)} (should be {expected})")

    # 2) 透传参数（!!! 一定要把 g/c 传给内核，否则会退化成 n=1）
    args = {
        'm': int(m),
        't': int((extra_args or {}).get('t', 50)),
        'g': int((extra_args or {}).get('g', d)),
        'c': int((extra_args or {}).get('c', d)),
    }
    print(f"[adapter] send to core: d={d}, |v|={len(v)}, g={args['g']}, c={args['c']}, t={args['t']}, m={args['m']}")

    # 3) 优先：函数式内核（launch_cs.run_cs）
    raw = None
    phi = None
    try:
        import types
        import launch_cs  # 确保 launch_cs.py 可被 import；否则走 CLI 兜底
        class _Args(types.SimpleNamespace): pass
        a = _Args()
        a.m = args['m']
        a.t = args['t']
        a.g = args['g']
        a.c = args['c']
        raw = launch_cs.run_cs(v, a)
        phi = _normalize_phi(raw, d)

        # 如果函数式返回的维度仍不对，自动兜底跑一遍 CLI
        if phi.size != d:
            print(f"[QR-CS adapter][WARN] func-core 结果维度={phi.size}，期望 d={d}，尝试 CLI 兜底...")
            raw_cli = _run_core_cli(v, args)
            phi = _normalize_phi(raw_cli, d)

    except Exception as e:
        # 4) 兜底：CLI 内核
        raw_cli = _run_core_cli(v, args)
        phi = _normalize_phi(raw_cli, d)

    # 5) 最终安全检查
    if phi.size != d:
        print(f"[QR-CS adapter][WARN] normalize 后尺寸仍异常: {phi.size}, 期望 {d}")
        # 兜底补齐/截断
        if phi.size > d:
            phi = phi[:d]
        else:
            phi = np.pad(phi, (0, d - phi.size), mode='constant')

    # 6) 返回 {局部索引: 值}，局部索引 0..d-1 与 X.columns 的顺序一致
    return {i: float(phi[i]) for i in range(d)}

# -------------------------------
# 分块：Block QR-CS
# -------------------------------

def qr_cs_block(X, y, eval_fn, names_order,
                block_size=12,   # 默认 12
                m=50,
                extra_args=None):
    """
    Block QR-CS：把 names_order 按块切分；对每块调用 qr_cs。
    每块都显式设置 g=c=块内维度，t 从 extra_args 透传，其它额外参数也透传。
    返回：{全局列下标: 值}
    """
    if extra_args is None or "t" not in extra_args:
        raise ValueError("qr_cs_block 需要 extra_args 至少包含 't'，例如 {'t':50}。")

    # 名称到全局下标的映射
    if hasattr(X, "columns"):
        name_to_global_idx = {c: i for i, c in enumerate(X.columns)}
    else:
        name_to_global_idx = {f"f{i}": i for i in range(X.shape[1])}

    out = {}
    total = len(names_order)
    for start in range(0, total, block_size):
        block_names = names_order[start:start + block_size]
        X_block = X[block_names]

        # 块内 eval：沿用外部 eval_fn（其内部用的是全局 X_topk / KPI，不受 X_block 限制）
        def eval_fn_block(S_block_names):
            return eval_fn(S_block_names)

        # 每块固定 g=c=block_size（或块实际维度），t 透传；其余 extra_args 也透传
        g_c = len(block_names)
        block_args = {"g": g_c, "c": g_c, "t": int(extra_args["t"])}
        for k, v in (extra_args or {}).items():
            if k in ("g", "c", "t"):
                continue
            block_args[k] = v

        phi_block = qr_cs(
            X_block, y,
            eval_fn=eval_fn_block,
            max_d_for_cs=g_c,
            m=m,
            extra_args=block_args
        )
        # 把块内局部索引映射回全局下标
        for local_idx, val in phi_block.items():
            name = block_names[int(local_idx)]
            gidx = name_to_global_idx[name]
            out[gidx] = float(val)

    return out
