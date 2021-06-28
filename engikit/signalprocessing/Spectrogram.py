import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import chirp
from AnalysisTool.engikit.signalprocessing import FFT_lib, A_Weighting, decibel
from  scipy import fftpack

# 平均化FFTする関数
def fft_ave(data_array, samplerate, Fs, N_ave, acf, no_db_a=True):
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)      # 周波数軸を作成
    a_scale = A_Weighting.aweightings(fft_axis)                # 聴感補正曲線を計算

    # FFTをして配列にdBで追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
    for i in range(N_ave):
        # dB表示しない場合とする場合で分ける
        if no_db_a == True:
            fft_array.append(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)))
        else:
            fft_array.append(decibel.linear2db(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)), 2e-5, "linear"))
    # 型をndarrayに変換しA特性をかける(A特性はdB表示しない場合はかけない）
    if no_db_a == True:
        fft_array = np.array(fft_array)
    else:
        fft_array = np.array(fft_array) + a_scale
    fft_mean = np.mean(np.sqrt(fft_array ** 2), axis=0)          # 全てのFFT波形の平均を計算

    return fft_array, fft_mean, fft_axis




from scipy import  signal
# Fsとoverlapでスペクトログラムの分解能を調整する。
Fs = 4096                                   # フレームサイズ
overlap = 90                                # オーバーラップ率

# 波形を作る
samplerate = 44100
t_max = 5

overlap_rate = 90

x = np.arange(0, t_max, 1/samplerate)
# 周波数が時間で変化する正弦波の生成
# 開始周波数f0, 停止周波数f1, 停止時間t1、スイープ手法method(linear, logarithmicなど)
data = signal.chirp(x, f0=1, f1=500, t1=5, method='linear')
# オーバーラップ抽出された時間波形配列
time_array, N_ave, final_time = FFT_lib.overlapping(data, samplerate, Fs, overlap_rate)


# ハニング窓関数をかける
time_array, acf = FFT_lib.window_func(time_array, Fs, N_ave, window_type="hanning")

# FFTをかける
#fft_array, fft_mean, fft_axis = FFT_lib.fft_average(time_array, samplerate, Fs, N_ave, acf, "AMP")

fft_array, fft_mean, fft_axis = fft_ave(time_array, samplerate, Fs, N_ave, acf, no_db_a=False)



# スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
fft_array = fft_array.T

# ここからグラフ描画
# グラフをオブジェクト指向で作成する。
fig = plt.figure()
ax1 = fig.add_subplot(111)

# データをプロットする。
im = ax1.imshow(fft_array,
                vmin=0, vmax=np.max(fft_array),
                extent=[0, final_time, 0, samplerate],
                aspect='auto',
                cmap='jet')

# カラーバーを設定する。
cbar = fig.colorbar(im)
cbar.set_label('SP [Pa]')

# 軸設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

# スケールの設定をする。
ax1.set_xticks(np.arange(0, 50, 1))
ax1.set_yticks(np.arange(0, 20000, 200))
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 1000)

# グラフを表示する。
plt.show()
plt.close()