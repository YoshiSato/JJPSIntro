import numpy as np
from shat_from_x import shat_from_x
from r_from_shat import r_from_shat

def pr_from_s(s, r_pos_list, params, n_sims, min_p):
    '''
    刺激sから反応rの分布をシミュレートする関数．
    s：2要素[s_a, s_v]の1次元配列．
    r_pos_list：回答ボタンに対応する位置．
    params：[sig_a, sig_v, p_common]の順であるとする．
    n_sims：シミュレーション試行数
    min_p：確率の最小値

    出力：
    反応の確率分布のシミュレーション値．rの要素数*2列の行列で，0列目がr_a，1列目がr_v．
    '''
    sig_a = params[0]
    sig_v = params[1]

    s_a = s[0]
    s_v = s[1]

    n_r = len(r_pos_list)

    # sにノイズを加えてxを生成する
    rng = np.random.default_rng()
    x_a = s_a + rng.normal(0, sig_a, size=n_sims)
    x_v = s_v + rng.normal(0, sig_v, size=n_sims)
    x = np.vstack([x_a, x_v]).T

    # xからrを計算
    s_hat = shat_from_x(x, params)
    r_sim = r_from_shat(s_hat, r_pos_list, params)

    # 確率分布を計算
    pr = np.zeros((n_r, 2)) # rの選択肢数 * 2つの感覚 AとV
    for i_s in range(2):
        for i_r in range(n_r):
            pr[i_r, i_s] = np.sum(r_sim[:, i_s] == i_r) / n_sims
    
    # 最小確率 min_p と最大確率 1-min_p を設定
    pr_norm = np.maximum(pr, np.ones(pr.shape) * min_p)
    pr_norm = np.minimum(pr_norm, np.ones(pr.shape) * (1-min_p))

    # 確率分布を再度正規化
    for i_s in range(2):
        pr_norm[:, i_s] = pr_norm[:, i_s] / np.sum(pr_norm[:, i_s])

    return pr_norm


# 動作確認テスト
if __name__ == "__main__":
    s = np.array([0, 10])
    params = np.array([10, 2, 0.5])
    r_pos_list = np.array([-10, -5, 0, 5, 10])
    n_sims = 100
    min_p = 0.01
   
    print(pr_from_s(s, r_pos_list, params, n_sims, min_p))




    

