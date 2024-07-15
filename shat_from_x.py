import numpy as np

def shat_from_x(x, params):
    '''
    xからs_hatを計算する関数．
    x：n行2列の行列で0列目がx_a，1列目がx_v．行は各試行．
    params：[sig_a, sig_v, p_common]の順であるとする．

    出力：
    n行2列の行列s_hatを返す．0列目がs_aの推定値，1列目がs_vの推定値．行は各試行．
    '''

    sig_a = params[0]
    sig_v = params[1]
    p_common = params[2]
    L = 180

    # xはx_aとx_vを合わせて一つの変数にしたものなので，わかりやすいようにx_aとx_vを定義
    x_a = x[:, 0]
    x_v = x[:, 1]

    # P(x_a, x_v | C=1)とP(x_a, x_v | C=2)を計算
    p_xa_xv_c1 = 1/(np.sqrt(2*np.pi*(sig_a**2 + sig_v**2))) * np.exp(-1/2*(x_a - x_v)**2/ (sig_a**2 + sig_v**2)) * 1/L
    p_xa_xv_c2 = 1/L**2

    # P(C=1 | x_a, x_v)とP(C=2 | x_a, x_v)を計算
    p_c1_xa_xv = (p_xa_xv_c1 * p_common) / (p_xa_xv_c1 * p_common + p_xa_xv_c2 * (1 - p_common))
    p_c2_xa_xv = 1 - p_c1_xa_xv
    
    # s_aの推定値とs_vの推定値を計算
    s_a_hat = p_c1_xa_xv * (x_a/sig_a**2 + x_v/sig_v**2) / (1/sig_a**2 + 1/sig_v**2) + p_c2_xa_xv * x_a
    s_v_hat = p_c1_xa_xv * (x_a/sig_a**2 + x_v/sig_v**2) / (1/sig_a**2 + 1/sig_v**2) + p_c2_xa_xv * x_v

    # s_a_hatとs_v_hatをまとめて一つの変数にする
    # s_hatはn行2列の行列．0列目がs_aの推定値，1列目がs_vの推定値．行は試行数．
    s_hat = np.vstack([s_a_hat, s_v_hat]).T

    return s_hat


# 動作確認テスト
if __name__ == "__main__":
    x = np.array([[0, 10]]) # x_a = 0, x_v = 10を渡してみる
    params = np.array([10, 2, 0.9]) # パラメータを適当な値に設定してみる
    
    print(shat_from_x(x, params)) # 上で定義した関数を実行し，値を表示