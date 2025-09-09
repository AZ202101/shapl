# qr_cs_adapter.py
import os
import sys
import re
import tempfile
import itertools
import numpy as np

# -------------------------------
# 基础：构造 v（标准顺序，按子集大小递增）
# -------------------------------

def _all_subsets_indices(d):
    subs = []
    for r in range(0, d + 1):
        for comb in itertools.combinations(range(d), r):
            subs.append(list(comb))
    return subs

def _build_v_by_eval_fn(feature_names, eval_fn):
    """
    构造 “标准顺序” 的 v（长度应为 2^d）：
    v(S) = f(S) - f(∅)
    """
    d = len(feature_names)
    all_sets = _all_subsets_indices(d)
    f_empty = eval_fn([])  # 空集
    v = []
    for idx_set in all_sets:
        S = [feature_names[i] for i in idx_set]
        v.append(float(eval_fn(S) - f_empty))
    return v  # 2^d

# -------------------------------
# 主函数：qr_cs（单块）
# -------------------------------

def _qr_cs_core(X, y, eval_fn, max_d_for_cs=12, m=50, extra_args=None):
    """
    极简适配器（按你的要求）：
      - 不再强制检查 nodes 三件套；
      - 不再强制读取/校验 indices_{g}.txt；
      - 直接把我们构造好的 v 传给 block_launch_cs.py；
      - 若返回 phi 维度与 d 不一致，仅提示并做截断/零填充，不再抛错。
    """
    # 取特征名
    if hasattr(X, "columns"):
        feature_names = list(X.columns)
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    d = len(feature_names)

    if d > max_d_for_cs:
        raise ValueError(
            f"[Block-QR-CS adapter] d={d} 超过 max_d_for_cs={max_d_for_cs}。"
            f"若需继续，请调小特征数或增大该阈值。"
        )

    if eval_fn is None:
        raise ValueError("必须提供 eval_fn(S_by_names)->float 以构造完整 v(S)。")

    # 参数：至少需要 g/c/t（与你的 block_launch_cs.py 对齐）
    need = ("g", "c", "t")
    if not extra_args or any(k not in extra_args for k in need):
        raise ValueError("block_launch_cs.py 需要 --g / --c / --t，请通过 extra_args={'g':...,'c':...,'t':...} 传入。")
    g = int(extra_args["g"])
    c = int(extra_args["c"])
    t = int(extra_args["t"])
    # 与之前不同：不再强制要求 g==c==d。保留你的灵活性。
    # 但给个提醒：
    if (g != d) or (c != d):
        print(f"[QR-CS adapter][WARN] g={g}, c={c}, d={d} 不一致。"
              f"将继续把长度 2^{d} 的 v 传给 block_launch_cs.py，由其内部处理。")

    # 1) 构造 v（标准顺序）
    v = _build_v_by_eval_fn(feature_names, eval_fn)  # 长度 2^d

    # 2) 优先函数式调用 run_cs；失败则 CLI 兜底
    phi = None
    try:
        import types
        import block_launch_cs

        class _Args(types.SimpleNamespace): pass
        args = _Args()
        args.m = m
        args.g = g
        args.c = c
        args.t = t
        # 透传其它参数
        for k, val in (extra_args or {}).items():
            setattr(args, k, val)

        if not hasattr(block_launch_cs, "run_cs"):
            raise AttributeError("block_launch_cs.py 没有暴露 run_cs(v, args) 函数。")

        CS_res = block_launch_cs.run_cs(v, args)
        phi = np.array(CS_res).reshape(-1)

    except Exception as e:
        # CLI 兜底
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "v.txt")
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(str(v) + "\n")

            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "block_launch_cs.py"),
                "--input", in_path,
                "--g", str(g), "--c", str(c), "--t", str(t)
            ]
            if m:
                cmd += ["--m", str(m)]
            for k, val in (extra_args or {}).items():
                if k in ("g", "c", "t"):
                    continue
                cmd += [f"--{k}", str(val)]

            out = subprocess.check_output(cmd, text=True)
        mobj = re.search(r"CS_res:\[([^\]]+)\]", out)
        if not mobj:
            raise RuntimeError(f"[Block-QR-CS] 无法在 block_launch_cs.py 输出中解析 CS_res。原始输出：\n{out}")
        arr = np.fromstring(mobj.group(1), sep=',')
        phi = arr.reshape(-1)

    # 3) 返回前做“软检查”：长度不一致就截断/零填充，并提示
    if phi.shape[0] != d:
        print(f"[Block-QR-CS adapter][WARN] phi 长度({phi.shape[0]}) 与 d({d}) 不一致。将进行适配（截断/零填充）。")
        if phi.shape[0] > d:
            phi = phi[:d]
        else:
            pad = np.zeros(d, dtype=float)
            pad[:phi.shape[0]] = phi
            phi = pad

    return {i: float(phi[i]) for i in range(d)}

# -------------------------------
# 分块：Block QR-CS（默认 12）
# -------------------------------

def block_qrcs(X, y, eval_fn, names_order,
                block_size=12,   # 恢复为 12（你提到原本是 12）
                m=50,
                extra_args=None):
    """
    Block QR-CS：把 names_order 按块切分；对每块调用 qr_cs。
    这里不再强制检查任何 nodes 文件；每块自动设置 g=c=block_size，t 从 extra_args 透传。
    """
    if extra_args is None or "t" not in extra_args:
        raise ValueError("block_qrcs 需要 extra_args 至少包含 't'，例如 {'t':50}。")

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

        def eval_fn_block(S_block_names):
            # 你的 eval_fn 已经接受列名列表，这里直接复用
            return eval_fn(S_block_names)

        # 每块固定 g=c=block_size，t 透传；不做 nodes 检查
        g_c = len(block_names)  # 当前块真实特征数
        block_args = {"g": g_c, "c": g_c, "t": extra_args["t"]}
        # 透传其余参数（若有）
        for k, v in (extra_args or {}).items():
            if k in ("g", "c", "t"):
                continue
            block_args[k] = v

        phi_block = _qr_cs_core(
            X_block, y,
            eval_fn=eval_fn_block,
            max_d_for_cs=len(block_names),
            m=m,
            extra_args=block_args
        )
        for local_idx, val in phi_block.items():
            name = block_names[int(local_idx)]
            gidx = name_to_global_idx[name]
            out[gidx] = float(val)

    return out
