import numpy as np
from scipy import signal
from scipy import fftpack

#オーバーラップ処理
def overlap(data, samplerate, Fs, overlap_rate):
    Ts = len(data) / samplerate         #全データ長
    Fc = Fs / samplerate                #フレーム周期
    x_ol = Fs * (1 - (overlap_rate/100))     #オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap_rate/100))) / (Fc * (1-(overlap_rate/100)))) #抽出するフレーム数（平均化に使うデータ個数）

    array = []      #抽出したデータを入れる空配列の定義

    #forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)              #切り出し位置をループ毎に更新
        array.append(data[ps:ps+Fs:1])  #切り出し位置psからフレームサイズ分抽出して配列に追加
    return array, N_ave                 #オーバーラップ抽出されたデータ配列とデータ個数を戻り値にする

#窓関数処理
def window_func(data_array, Fs, N_ave, mode):
    '''
    入力データに対して窓関数を適用する
    FFTのデータ分割及びおーばラップ処理で使用する。
    データを分割すると切り出した部分で波形が急に変化する。
    これを抑えるために窓関数を適用。
    ただし、窓関数を適用すると信号が減衰するため、補正処理を適用する
    :param data_array: 入力データ
    :param Fs: フレームサイズ
    :param N_ave: 分割データ数
    :param mode: 適用する窓関数の種類
    :return:
    '''
    if mode == "hanning":
        window = signal.windows.hann(Fs)  # ハニング
    elif mode == "hamming":
        window = signal.windows.hamming(Fs)# ハミング
    elif mode == "blackman":
        window = signal.windows.blackman(Fs)  # ブラックマン
    elif mode == "bartlett":
        window = signal.windows.bartlett(Fs)  # バートレット
    elif mode == "kaiser":
        alpha = 0  # 0:矩形、1.5:ハミング、2.0:ハニング、3:ブラックマンに似た形
        Beta = np.pi * alpha
        window = signal.windows.kaiser(Fs, Beta)
    else:
        print("Error: input window function name is not sapported. Your input: ", mode)
        print("Hanning window function is used.")
        window = signal.windows.hann(Fs)  # ハニング

    acf = 1 / (sum(window) / Fs)   #振幅補正係数(Amplitude Correction Factor)

    #オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * window #窓関数をかける

    return data_array, acf



#FFT処理
def fft_average(data_array,samplerate, Fs, N_ave, acf, mode):
    '''
    入力データのFFTを行い窓関数補正と正規化、平均化をする
    :param data_array:  入力データ
    :param samplerate: サンプリングレート、サンプリング周波数[Hz]
    :param Fs: フレームサイズ(FFTされるデータの点数)
    :param N_ave: フレーム総数
    :param acf:窓関数補正値
    :param mode: 解析モード


    :return:
        :fft_array: フーリエスペクトル


    '''
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(fftpack.fft(data_array[i])/(Fs/2))) #FFTをして配列に追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。さらに絶対値をとる

    fft_axis = np.linspace(0, samplerate, Fs)   #周波数軸を作成
    fft_array = np.array(fft_array)             #型をndarrayに変換

    if mode == "AMP":
        amp_spectrum_mean = np.mean(fft_array, axis=0)# 平均化された振幅スペクトル
        # ナイキスト定数まで抽出
        fft_axis_out = fft_axis[:int(Fs / 2) + 1]
        fft_spectrum_mean_out = amp_spectrum_mean[:int(Fs // 2) + 1]
    elif mode == "PSD":
        psd_spectrum_mean = np.mean(fft_array ** 2 / (samplerate / Fs), axis=0)  # 平均化されたパワースペクトラム密度
        # ナイキスト定数まで抽出
        fft_axis_out = fft_axis[:int(Fs / 2) + 1]
        fft_spectrum_mean_out = psd_spectrum_mean[:int(Fs // 2) + 1]

    #fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))       #全てのFFT波形のパワー平均を計算してから振幅値とする
    #fft_PSD_mean = np.mean(fft_array ** 2 / (samplerate/ Fs), axis=0)

    else:
        print("Error: input fft mode name is not sapported. Your input: ", mode)
        print("Mode will be AMP.")

    return fft_array, fft_spectrum_mean_out, fft_axis_out


def FFT_main(t, data, split_rate, samplerate, overlap_rate, window_mode, analysis_mode):
    print("Execute FFT")
    Fs =   int(len(t) * split_rate)# フレームサイズ

    # 作成した関数を実行：オーバーラップ抽出された時間波形配列
    split_data, N_ave = overlap(data, samplerate, Fs, overlap_rate)

    # 作成した関数を実行：ハニング窓関数をかける
    time_array, acf = window_func(split_data, Fs, N_ave, mode=window_mode)

    # 作成した関数を実行：FFTをかける
    fft_array, fft_spectrum_mean_out, fft_axis_out = fft_average(time_array, samplerate, Fs, N_ave, acf, analysis_mode)

    return fft_array, fft_spectrum_mean_out, fft_axis_out


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    samplerate = 100#25600 [Hz]

    x = np.arange(0, 50.01, 1/samplerate)
    data = 2 * np.sin(2 * np.pi * 20 * x) + 5 * np.sin(2 * np.pi * 1 * x)

    split_rate = 0.1
    overlap_rate = 70  # オーバーラップ率
    delta_f = samplerate / int(len(x) * split_rate)# 周波数分解能 = サンプリングレート / フレームサイズ


    fft_array, fft_spectrum_mean_out, fft_axis_out = FFT_main(x, data, split_rate, samplerate, overlap_rate, window_mode="hanning", analysis_mode="PSD")
    plt.plot(fft_axis_out, fft_spectrum_mean_out)
    plt.show()


