import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import kendalltau, pearsonr
import cvxpy as cp
import random
from block_qr_cs_adapter import block_qrcs
from qr_cs_adapter import qr_cs
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..')))
from shapG.shapley import shapG, cis
import shapG.plot as shapGplot
from shapG.utils import corr_generator, create_minimal_edge_graph, matrix_generator, kl, kl_mi_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
"""
benchmark_feature_importance.py

目的：一次性对比五种特征重要性/“沙普利值”近似方法：
- shapG（图结构 Shapley）
- CIS（Cut Influence Score）
- QR-CS（单块压缩感知近似）
- Block-QR-CS（分块版 QR-CS）
- Random-CS（随机采样 + L1 重构的 CS 近似）
流程概览：
1) 读数据 -> 构图（create_minimal_edge_graph） -> shapG、CIS 得到全量排名
2) 用 shapG 前 k 个特征作为候选集，跑 QR-CS 与 Block-QR-CS（更高效）
3) 跑 Random-CS（对全量特征），以压缩测量 + L1 重构得到 φ
4) 把五种方法的排名输入统一的 KPI 曲线对比函数（逐步丢弃特征评估性能衰减）

"""

# Random-CS：生成伯努利随机矩阵
def _bernoulli_matrix(m, n, rng=np.random):
    A = rng.binomial(1, 0.5, size=(m, n)).astype(np.float64)
    A[A == 1] = 1.0 / np.sqrt(m)
    A[A == 0] = -1.0 / np.sqrt(m)
    return A
def cs_det_shapley_values(X, y, kpi_func, m=100, t=50, eps=1e-3, seed=19365):
    """
        说明：该实现按随机集合 S 估计“个体边际贡献”的压缩观测 y，
        再通过 L1 最小化从 y 中重构出 φ（以 s_bar 为 baseline）。
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = X.shape[1]
    feat_idx = np.arange(n)
    # 压缩测量矩阵
    A = _bernoulli_matrix(m, n, rng=rng)
    # 生成压缩观测并对 t 次迭代求均值
    y_list = []
    for _ in range(t):
        r_size = rng.integers(1, n + 1)
        S = sorted(rng.choice(feat_idx, size=r_size, replace=False).tolist())
        delta = np.zeros(n, dtype=np.float64)
        for i in feat_idx:
            S_wo = [j for j in S if j != i]
            S_w = S if i in S else sorted(S + [i])
            u1 = kpi_func(X, y, [X.columns[k] for k in S_w]) if len(S_w) > 0 else 0.0
            u2 = kpi_func(X, y, [X.columns[k] for k in S_wo]) if len(S_wo) > 0 else 0.0
            delta[i] = float(u1 - u2)
        y_list.append(A @ delta)
    y_bar = np.mean(np.stack(y_list, axis=1), axis=1)   # (m,)
    s_bar = float(kpi_func(X, y, list(X.columns))) / n  # baseline / n

    # L1-最小化：φ = s_bar + x
    x = cp.Variable((n, 1))
    constraints = [cp.norm(A @ (s_bar + x) - y_bar[:, None]) <= eps]
    prob = cp.Problem(cp.Minimize(cp.norm(x, 1)), constraints)
    prob.solve(solver=cp.ECOS, warm_start=True)
    phi = (s_bar + x.value).ravel()
    return phi
# Data readers
def housing_data_reader(filename='./data/housing_price.csv'):
    data = pd.read_csv(filename)
    X = data.drop(['MEDV'], axis=1)
    y = data['MEDV']
    return X, y
def h1n1_data_reader(filename='./data/process_data.csv'):
    data = pd.read_csv(filename)
    X = data.drop(['h1n1_vaccine', 'respondent_id', 'seasonal_vaccine'], axis=1)
    y = data['h1n1_vaccine']
    return X, y


def plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename=None, limit=7):
    """
    Plot the comparison of KPIs for different feature selection methods.

    Parameters:
    - reader: Function to read the dataset.
    - feature_rankings: Dictionary where keys are method names and values are lists of features in order of importance.
    - model: The machine learning model to use (LGBM or MLP).
    - filename: File name to save the plot.
    - limit: Maximum number of features to consider.

    Returns:
    - Dictionary containing results for each method.
    """
    # Define model specific parameters
    random_state = [42, 42]
    test_size = [0.2, 0.3]

    # Generate results file name
    model_name = type(model).__name__
    results_file = f"{model_name}_dict_results.pkl"

    # Load or calculate results
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded results for {model_name} from disk.")
    else:
        X, y = reader()
        results = {}
        # Calculate initial metric (without dropping features)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state[0]
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        initial_metric = r2_score(y_test, y_pred)

        # Process each ranking method
        for method, feature_order in feature_rankings.items():
            # Make sure feature_order contains only column names as strings
            feature_order = [feat if isinstance(feat, str) else feat[0] for feat in feature_order]
            if limit:
                feature_order = feature_order[:limit]
            metrics = [initial_metric]
            features = [[]]
            deltas = []
            for i in range(1, len(feature_order) + 1):
                features_to_drop = feature_order[:i]
                # Check if all features exist in dataframe
                missing_cols = [col for col in features_to_drop if col not in X.columns]
                if missing_cols:
                    print(f"Warning: Columns {missing_cols} not found in dataset. Skipping.")
                    continue
                reduced_X = X.drop(columns=features_to_drop)
                x_train, x_test, y_train, y_test = train_test_split(
                    reduced_X, y, test_size=test_size[1], random_state=random_state[1]
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                new_metric = r2_score(y_test, y_pred)
                deltas.append(metrics[-1] - new_metric)
                metrics.append(new_metric)
                features.append(features_to_drop)
            # Calculate weighted slope for comparison
            beta = 0.8
            weight = [beta ** i for i in range(len(deltas))]
            results[method] = {
                'Features': features,
                'Metrics': metrics,
                'Slope': np.dot(deltas, weight) if deltas else 0
            }
        # Save results to disk
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results for {model_name} to disk.")

    # Create the plot
    plt.figure(figsize=(12, 8))
    metric_name = "$R^2$"
    for method, data in results.items():
        label = f'{method} $S$={data["Slope"]:.4f}'
        plt.plot(
            range(len(data['Metrics'])),
            data['Metrics'],
            label=label,
            alpha=0.6
        )
    plt.xlabel('Number of Features Dropped')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} after dropping features based on different XAI methods ({model_name})')
    plt.legend()
    plt.grid()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()
    return results

# 绘制算法的 Shapley 值对比图
def benchmark_feature_importance(reader, model, filename=None, limit=7):
    """
    Benchmark feature importance using different methods.

    Parameters:
    - reader: Function to read the dataset.
    - model: The machine learning model to use (LGBM or MLP).
    - filename: File name to save the plot.
    - limit: Maximum number of features to consider.
    """
    X, y = reader()
    W = matrix_generator(X)
    A, W_new = create_minimal_edge_graph(W, reverse=True, version='v3')
    G = nx.Graph(A)

    # 计算三种沙普利的值
    t0 = time.time()
    shapley_values = shapG(
        G, m=3, f=lambda G, S: classification_kpi(X, y, S),
        approximate_by_ratio=False, scale=False
    )
    t1 = time.time()
    shapg_time = t1 - t0

    t0 = time.time()
    cis_values = cis(
        G, f=lambda G, S: classification_kpi(X, y, S)
    )
    t1 = time.time()
    cis_time = t1 - t0


    sorted_shapley_g = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    k_for_qrcs = 18
    topk_names = []
    for node, _ in sorted_shapley_g:
        try:
            idx = int(node)
            if 0 <= idx < len(X.columns):
                topk_names.append(X.columns[idx])
        except (ValueError, TypeError):
            if node in X.columns:
                topk_names.append(node)
        if len(topk_names) >= k_for_qrcs:
            break

    t0 = time.time()
    block_qr_cs_values_glb = block_qrcs(
        X, y,
        eval_fn=lambda S: classification_kpi(X, y, S),
        names_order=topk_names,
        block_size=4,
        m=50,
        extra_args={'t': 50}
    )
    block_qr_cs_pairs = [
        (topk_names[int(i)], v)
        for i, v in sorted(block_qr_cs_values_glb.items(), key=lambda x: x[1], reverse=True)
    ]
    t1 = time.time()
    block_qrcs_time = t1 - t0
    t0 = time.time()
    qr_cs_local = qr_cs(
        X[topk_names], y,
        eval_fn=lambda S: classification_kpi(X, y, S),  # 注意：eval_fn 传入的是列名集合 S
        max_d_for_cs=len(topk_names),
        m=50,
        extra_args={'t': 50, 'g': len(topk_names), 'c': len(topk_names)}
    )
    # 将局部索引映射回列名，得到(列名,φ)排序列表
    qr_cs_pairs = [
        (topk_names[int(i)], v)
        for i, v in sorted(qr_cs_local.items(), key=lambda x: x[1], reverse=True)
    ]
    t1 = time.time()
    qrcs_time = t1 - t0
    # Random-CS(随机子集产生压缩观测，L1 重构 φ；对全量特征运行)
    t0 = time.time()
    phi_cs = cs_det_shapley_values(
        X, y,
        kpi_func=lambda X_, y_, cols: classification_kpi(X_, y_, cols),
        m=100, t=50, eps=1e-3, seed=19365
    )
    cs_order = np.argsort(phi_cs)[::-1]
    random_cs_pairs = [(X.columns[i], float(phi_cs[i])) for i in cs_order]
    t1 = time.time()
    randomcs_time = t1 - t0


    # === 构造 feature_rankings ===
    feature_rankings = {}
    # ShapG
    sorted_shapley_g = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['shapG'] = []
    for node, _ in sorted_shapley_g:
        try:
            idx = int(node)
            if 0 <= idx < len(X.columns):
                feature_rankings['shapG'].append(X.columns[idx])
        except (ValueError, TypeError):
            if node in X.columns:
                feature_rankings['shapG'].append(node)
    # CIS
    sorted_cis = sorted(cis_values.items(), key=lambda x: x[1], reverse=True)
    feature_rankings['CIS'] = []
    for node, value in sorted_cis:
        try:
            idx = int(node)
            if 0 <= idx < len(X.columns):
                feature_rankings['CIS'].append(X.columns[idx])
        except (ValueError, TypeError):
            if node in X.columns:
                feature_rankings['CIS'].append(node)
    # QR-CS(单块)
    feature_rankings['QR-CS'] = [name for name, _ in qr_cs_pairs]
    # Block-QR-CS(分块)
    feature_rankings['Block-QR-CS'] = [name for name, _ in block_qr_cs_pairs]
    #Random-CS(随机压缩感知)
    feature_rankings['Random-CS'] = [name for name, _ in random_cs_pairs]


    # Model importance

    # Plot
    results = plot_KPI_comparison_by_dict(reader, feature_rankings, model, filename, limit)

    print(f"[Time] shapG: {shapg_time:.2f}s, CIS: {cis_time:.2f}s, QR-CS: {qrcs_time:.2f}s, Block-QR-CS: {block_qrcs_time:.2f}s", f"Random-CS: {randomcs_time:.2f}s")

    random_cs_values = dict(random_cs_pairs)
    qr_cs_values = dict(qr_cs_pairs)
    block_qr_cs_values = dict(block_qr_cs_pairs)
    return shapley_values, cis_values, qr_cs_values, block_qr_cs_values, random_cs_values, results


# Classification KPI
from sklearn.model_selection import train_test_split

def classification_kpi(X, y, S):
    cols = list(S)
    if len(cols) == 0:
        return 0.0
    X_train, X_test, y_train, y_test = train_test_split(
        X[cols], y, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        ))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)



if __name__ == "__main__":
    import time
    start = time.time()

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )

    shapley_values, cis_values, qr_cs_values, block_qr_cs_values, random_cs_values, results = \
        benchmark_feature_importance(housing_data_reader, model, filename='housing_benchmark_mlp.png')

    print("Shapley values:", shapley_values)
    print("CIS values:", cis_values)
    print("QR_CS values:", qr_cs_values)
    print("Block_QR_CS values:", block_qr_cs_values)
    print("Random_CS values:", random_cs_values)

    end = time.time()
    print(f"Elapsed: {end - start:.2f} seconds")


