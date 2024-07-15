import numpy as np

def r_from_shat(s_hat, r_pos_list, params=None):
    '''
    s_hatからrを計算する関数．
    s_hat：n行2列の行列．0列目がs_a_hat，1列目がs_v_hat．行は試行数．
    r_pos_list：回答ボタンに対応する位置．

    出力：
    s_hatに最も近いボタンを選択するものとし，n行2列の配列rを返す．
    '''

    # 配列に対する計算として一度に処理するため，3次元の構造にしてからブロードキャストを利用して距離を計算．
    shat_ext = s_hat[:, :, np.newaxis] # s_hatを3次元配列にする
    r_pos_ext = r_pos_list[np.newaxis, np.newaxis, :] + np.zeros((s_hat.shape[0], s_hat.shape[1], 1)) # ブロードキャストを利用して，r_pos_listをコピーして並べたような3次元配列を作る
    diff_ext = np.abs(shat_ext - r_pos_ext) # s_hatとr_pos_listの差を計算する

    # 距離が一番小さくなるrを求める（rの値は0スタートでカウントされる）
    r = np.argmin(diff_ext, axis=2)
    
    return r


# 動作確認テスト
if __name__ == "__main__":
    s_hat = np.array([[0, 10], [-7, 5], [-12, -3]])
    r_pos_list = np.array([-10, -5, 0, 5, 10])
    
    print(r_from_shat(s_hat, r_pos_list))

