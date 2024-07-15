import numpy as np
import pandas as pd
from calculate_NLL import calculate_NLL
from pybads import BADS

# 実験データの読み込みと実験条件
r_pos_list = np.array([-10, -5, 0, 5, 10])
D = pd.read_csv("simulated_data_15_5_0.2.csv")

# モデルフィッティング用パラメータ類の設定
n_sims = 10000 # pr_from_sで使う，シミュレーション回数
min_p = 0.01 # 確率として設定する最小値
param_names = ['sig_a', 'sig_v', 'p_common']
n_params = len(param_names)

# 最適化用パラメータ類の設定
#n_ops = 100 # 最適化計算を繰り返す回数．大きい値にするのは最後に行う．
n_ops = 5 # 最適化計算を繰り返す回数
params_upper = np.array([100, 100, 1]) # パラメータ上限のハードリミット
params_lower = np.array([0, 0, 0]) # パラメータ下限のハードリミット
params_plausible_upper = np.array([30, 10, 0.99]) # パラメータの妥当な範囲の上限
params_plausible_lower = np.array([1, 0.01, 0.01]) # パラメータの妥当な範囲の下限


# 最適化する関数を定義する．フィッティングするパラメータのリストparamsを引数とする関数として定義する必要がある．
# ここでは無名関数lambdaを使っているが，通常の関数にしても問題ない．
func = lambda params: calculate_NLL(D, r_pos_list, params, n_sims, min_p)

rng = np.random.default_rng()

x_results = np.zeros((n_ops, n_params))
fval_results = np.zeros(n_ops)

# 初期値を変えながら最適化を繰り返す．
for i_ops in range(n_ops):
    print(f"-----------{i_ops+1}/{n_ops}-----------")
    # 初期値の設定．ここではparams_plausible_lowerとupperで指定した値の間からランダムに選ぶ．
    x0 = rng.random(n_params) * (params_plausible_upper - params_plausible_lower) + params_plausible_lower
    #x0 = [10, 2, 0.5]

    # BADSを用いた最適化の実行
    bads = BADS(func, x0, params_lower, params_upper, params_plausible_lower, params_plausible_upper)
    optimize_result = bads.optimize()

    x_results[i_ops, :] = optimize_result["x"] # 求められたパラメータ
    fval_results[i_ops] = optimize_result["fval"] # そのパラメータにおける関数値（NLL）

# フィッティングの結果の概要を表示
result_df = pd.DataFrame(np.hstack([fval_results.reshape(-1, 1), x_results]), columns = ['NLL'] + param_names)
print("パラメータフィッテングの概要")
print(result_df.describe())

# ベストなパラメータを求める
best_ind = np.argmin(fval_results)
best_fval = fval_results[best_ind]
best_x = x_results[best_ind, :]

print()
print("最適パラメータ")
print("負の対数尤度：", best_fval)
print("BIC：", n_params * np.log(D.shape[0]) + 2 * best_fval)
print("パラメーター値：", best_x)

