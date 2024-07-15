import numpy as np
import pandas as pd
from shat_from_x import shat_from_x
from r_from_shat import r_from_shat

def create_sim(r_pos_list, stim, params):
    '''
    一人分のシミュレーションの行動データを作る．
    r_pos_list：回答ボタンに対応する位置．
    stim：提示する刺激． 試行数×2要素の行列．0列目はs_a, 1列目はs_v．
    params：モデルパラメータ．[sig_a, sig_v, p_common]の順であるとする．
    
    出力：行動データr
    '''

    rng = np.random.default_rng()

    n_trial = stim.shape[0]
    sig_a = params[0]
    sig_v = params[1]

    # ノイズ込の観測値を生成
    x_a = stim[:, 0] + rng.normal(0, sig_a, size=n_trial)
    x_v = stim[:, 1] + rng.normal(0, sig_v, size=n_trial)
    x = np.vstack([x_a, x_v]).T

    s_hat = shat_from_x(x, params)
    r = r_from_shat(s_hat, r_pos_list, params)

    return r


# パラメータ類を設定して行動データ生成

if __name__ == "__main__":

    # パラメータの設定
    params = np.array([12, 3, 0.8])
    r_pos_list = np.array([-10, -5, 0, 5, 10])
   
    # 実験条件の設定
    s_a_list = [-10, -5, 0, 5, 10]    
    s_v_list = [-10, -5, 0, 5, 10]
    n_reps = 10 # 各s_aとs_vの組み合わせを何回繰り返すか

    # 刺激の全組み合わせの生成
    import itertools
    stim_prod = np.array(list(itertools.product(s_a_list, s_v_list)))

    # 全組み合わせをn_reps繰り返す
    stim_rep = np.tile(stim_prod, (n_reps,1))

    # 試行をランダムに並べ直す（やらなくてもいい）
    rng = np.random.default_rng()
    stim = rng.permutation(stim_rep)

    # シミュレーションデータの生成
    r_sim = create_sim(r_pos_list, stim, params)

    # pandasのデータフレームにしてCSVで保存
    data_df = pd.DataFrame(np.hstack([stim, r_sim]), columns=['s_a', 's_v', 'r_a', 'r_v'])
    data_df.to_csv(f"simulated_data_{params[0]}_{params[1]}_{params[2]}.csv", index=False)
    print("ファイルにセーブしました")





    