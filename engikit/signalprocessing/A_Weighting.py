import numpy as np

def aweightings(f):
    '''
    聴感補正カーブに基づきフーリエスペクトルに対して聴感補正を行う
    RA(f) = 12194^2 * f^4 / (f^2 + 20.6^2) * (sqrt{(f^2 + 107.7^2) + (f^2 + 737.9^2)} * (f^2 + 12194^2))
    A(f) = 20 log(RA(f)) + 2.00

    :param f:
    :return:
    '''
    if f[0] == 0:
        f[0] = 1
    else:
        pass
    ra = (np.power(12194, 2) * np.power(f, 4)) / \
         ((np.power(f, 2) + np.power(20.6, 2)) * \
          np.sqrt((np.power(f, 2) + np.power(107.7, 2)) * \
                  (np.power(f, 2) + np.power(737.9, 2))) * \
          (np.power(f, 2) + np.power(12194, 2)))
    a = 20 * np.log10(ra) + 2.00
    return a

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f = np.linspace(0, 25600, 4096)  # 周波数軸を作成
    a = aweightings(f)  # 周波数毎の聴感補正カーブを計算
    plt.plot(f, a)