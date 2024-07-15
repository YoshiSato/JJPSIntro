import numpy as np
import pandas as pd
from pr_from_s import pr_from_s

def calculate_NLL(D, r_pos_list, params, n_sims, min_p):
    '''
    D：計測データ．試行数*4列の行列で，0列目がs_a，1列目がs_v, 2列目が聴覚反応のボタン番号（0からカウントスタート）, 3列目が視覚反応のボタン番号．
    r_pos_list：回答ボタンに対応する位置．
    params：モデルパラメータ．[sig_a, sig_v, p_common]の順であるとする．
    n_sims：シミュレーション試行数
    min_p：確率の最小値
    
    出力：モデルの負の対数尤度．
    '''

    n_r = len(r_pos_list)

    # データ処理のためにpandasを用いる．
    df = pd.DataFrame(D, columns=["s_a", "s_v", "r_a", "r_v"])
    
    # s_aとs_vの組み合わせを抽出して試行数をカウント．
    df_unique = df.groupby(["s_a", "s_v"]).size().reset_index(name="freq")

   
    # 各刺激の組み合わせについて繰り返し
    nll = 0
    for ind in df_unique.index:
        
        row = df_unique.loc[ind, :]
        
        # s_aとs_vに対応するデータを抜き出す
        df_stim = df[(df["s_a"] == row["s_a"]) & (df["s_v"] == row["s_v"])]

        # モデルの予測分布を計算
        pr = pr_from_s(row.loc[["s_a", "s_v"]].values, r_pos_list, params, n_sims, min_p)

        #aとvに関してそれぞれNLLを計算し足し合わせる
        for i_s, s in enumerate(["a", "v"]):

            # 計測データの頻度を計算
            n = np.zeros(n_r)
            for i in range(n_r):
                n[i] = np.sum(df_stim["r_" + s] == i)

            # 負の対数尤度を計算
            nll += -(np.sum(n * np.log(pr[:, i_s])))

    return(nll)



# 動作確認テスト
if __name__ == "__main__":
    D = [[0, 0, 3, 2], [5, 10, 4, 4], [-5, -5, 1, 0], [0, 0, 4, 3], [5, 10, 3, 2], [-10, -5, 2, 1]]
    params = np.array([10, 2, 0.5])
    r_pos_list = np.array([-10, -5, 0, 5, 10])
    n_sims = 100
    min_p = 0.01
   
    print(calculate_NLL(D, r_pos_list, params, n_sims, min_p))










