import numpy as np
from scipy import signal
from scipy import fftpack
from AnalysisTool.engikit.signalprocessing import A_Weighting, decibel

#オーバーラップ処理
def overlapping(data, samplerate, Fs, overlap_rate):
    '''
    入力データに対してオーバーラップ処理を行う
    フレームサイズを定義してデータを切り出すと切り出しができない部分が発生する
    その際の時間も返すように設定
    スペクトログラムを表示する際に使用する

    :param data: 入力データ
    :param samplerate: サンプリングレート[Hz]
    :param Fs: フレームサイズ
    :param overlap_rate: オーバーラップレート[%]

    :return:
        :array: オーバーラップ加工されたデータ
        :N_ave:　オーバーラップ加工されたデータの個数
        :final_time: 最後に切り出したデータの時間
    '''
    Ts = len(data) / samplerate         #全データ点数
    Fc = Fs / samplerate                #フレーム周期
    x_ol = Fs * (1 - (overlap_rate/100))     #オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap_rate/100))) / (Fc * (1-(overlap_rate/100)))) #抽出するフレーム数（平均化に使うデータ個数）

    array = []      #抽出したデータを入れる空配列の定義

    #forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)              #切り出し位置をループ毎に更新
        array.append(data[ps:ps+Fs:1])  #切り出し位置psからフレームサイズ分抽出して配列に追加
    final_time = (ps + Fs) / samplerate
    return array, N_ave, final_time #オーバーラップ抽出されたデータ配列とデータ個数を戻り値にする

#窓関数処理
def window_func(data_array, Fs, N_ave, window_type):
    '''
    入力データに対して窓関数を適用する
    FFTのデータ分割及びオーバーラップ処理で使用する。
    データを分割すると切り出した部分で波形が急に変化する。
    これを抑えるために窓関数を適用。
    ただし、窓関数を適用すると信号が減衰するため、補正処理を適用する
    :param data_array: 入力データ
    :param Fs: フレームサイズ
    :param N_ave: 分割データ数
    :param mode: 適用する窓関数の種類

    :return:
        :data_array: 窓関数が適用されたデータ
        :acf: 窓関数補正値
    '''
    if window_type == "hanning":
        window = signal.windows.hann(Fs)  # ハニング
    elif window_type == "hamming":
        window = signal.windows.hamming(Fs)# ハミング
    elif window_type == "blackman":
        window = signal.windows.blackman(Fs)  # ブラックマン
    elif window_type == "bartlett":
        window = signal.windows.bartlett(Fs)  # バートレット
    elif window_type == "kaiser":
        alpha = 0  # 0:矩形、1.5:ハミング、2.0:ハニング、3:ブラックマンに似た形
        Beta = np.pi * alpha
        window = signal.windows.kaiser(Fs, Beta)
    else:
        print("Error: input window function name is not sapported. Your input: ", window_type)
        print("Hanning window function is used.")
        window = signal.windows.hann(Fs)  # ハニング

    acf = 1 / (sum(window) / Fs)   #振幅補正係数(Amplitude Correction Factor)

    #オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * window #窓関数をかける

    return data_array, acf


#FFT処理
def fft_average(data_array,samplerate, Fs, N_ave, acf, spectrum_type):
    '''
    入力データのFFTを行い窓関数補正と正規化、平均化をする
    :param data_array:  入力データ
    :param samplerate: サンプリングレート、サンプリング周波数[Hz]
    :param Fs: フレームサイズ(FFTされるデータの点数)
    :param N_ave: フレーム総数
    :param acf:窓関数補正値
    :param mode: 解析モード


    :return:
        :fft_array: フーリエスペクトル(平均化、正規化及び窓補正済み)
        :fft_spectrum_mean_out: ナイキスト周波数まで抽出したスペクトル(スペクトルの種類は解析モードによる)
        :fft_axis_out: ナイキスト周波数まで抽出した周波数軸

    '''
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf*np.abs(fftpack.fft(data_array[i])/(Fs/2))) #FFTをして配列に追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。さらに絶対値をとる

    fft_axis = np.linspace(0, samplerate, Fs)   #周波数軸を作成
    fft_array = np.array(fft_array)             #型をndarrayに変換

    if spectrum_type == "AMP":
        amp_spectrum_mean = np.mean(fft_array, axis=0)# 平均化された振幅スペクトル
        # ナイキスト定数まで抽出
        fft_axis_out = fft_axis[:int(Fs / 2) + 1]
        fft_spectrum_mean_out = amp_spectrum_mean[:int(Fs // 2) + 1]
    elif spectrum_type == "PSD":
        psd_spectrum_mean = np.mean(fft_array ** 2 / (samplerate / Fs), axis=0)  # 平均化されたパワースペクトラム密度
        # ナイキスト定数まで抽出
        fft_axis_out = fft_axis[:int(Fs / 2) + 1]
        fft_spectrum_mean_out = psd_spectrum_mean[:int(Fs // 2) + 1]

    #fft_mean = np.sqrt(np.mean(fft_array ** 2, axis=0))       #全てのFFT波形のパワー平均を計算してから振幅値とする
    #fft_PSD_mean = np.mean(fft_array ** 2 / (samplerate/ Fs), axis=0)

    else:
        print("Error: input fft mode name is not sapported. Your input: ", spectrum_type)
        print("Mode will be AMP.")

    return fft_array, fft_axis_out, fft_spectrum_mean_out


def fft_average_corr(data_array, samplerate, Fs, N_ave, acf, correction_mode):
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)      # 周波数軸を作成

    if correction_mode == "AWeight":
        '''
        FFT結果のスペクトルをA特性周波数重み付け音圧レベルで補正を適用する
        "音の大きさ"のレベル[dB]を表し、人間の感覚に近い解析結果が得られる
        音圧レベルLp = 10log10(p^2/p0^2) = 20log10(p/2*10^(-5))
        p[Pa] =　音圧
        p0[Pa] = 基準となる音圧
        すなわち、FFT結果の振幅スペクトルを用いる(PSDやPSではない)
        '''
        base_db_val =  2e-5 #　音圧の基準となる実効値(20µPa)

        for i in range(N_ave):
            a_scale = A_Weighting.aweightings(fft_axis)                # 聴感補正曲線を計算
            # FFTでフーリエスペクトルを算出し正規化後、絶対値をとる。さらに窓関数補正をする。その後dBに変換
            fft_array.append(decibel.linear2db(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)), base_db_val, "linear"))

        fft_array_Ascaled = np.array(fft_array) + a_scale# A補正を実施
        fft_array_Ascaled_mean = np.mean(np.sqrt(fft_array_Ascaled ** 2), axis=0)          # 全てのFFT波形の平均を計算
        #fft_corr_array_out = fft_corr_array[:int(Fs / 2) + 1]

        fft_array_out = fft_array_Ascaled
        fft_axis_out = fft_axis[:int(Fs / 2) + 1]
        fft_spectrum_mean_out = fft_array_Ascaled_mean[:int(Fs / 2) + 1]

    if correction_mode == "Simple":
        '''
        dB表示及び補正を適用しない
        ※窓関数補正や正規化は実施する
        '''
        for i in range(N_ave):
            fft_array.append(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)))
        fft_array = np.array(fft_array)
        fft_array_out = fft_array

        fft_axis_out = fft_axis[:int(Fs / 2) + 1]
        fft_spectrum_mean = np.mean(np.sqrt(fft_array ** 2), axis=0)  # 全てのFFT波形の平均を計算
        fft_spectrum_mean_out = fft_spectrum_mean[:int(Fs / 2) + 1]

    return fft_array_out, fft_axis_out, fft_spectrum_mean_out




def FFT_main(t, data, Fs, samplerate, overlap_rate, window_type, spectrum_type):
    '''
    FFTを実施するメインコード
    :param t: 時間データ
    :param data: 時間データに対する信号のテータ
    :param Fs: フレームサイズ フレームサイズ = samplerate/周波数分解能
    :param samplerate: サンプリングレート[Hz]
    :param overlap_rate: オーバーラップ率[%]
    :param window_mode: 窓関数の種類
    :param analysis_mode: 解析モードの種類
    :return:
        :fft_array: フーリエスペクトル(平均化、正規化及び窓補正済み)
        :fft_spectrum_mean_out: ナイキスト周波数まで抽出したスペクトル(スペクトルの種類は解析モードによる)
        :fft_axis_out: ナイキスト周波数まで抽出した周波数軸

    '''
    print("Execute FFT")

    # 作成した関数を実行：オーバーラップ抽出された時間波形配列
    split_data, N_ave, final_time = overlapping(data, samplerate, Fs, overlap_rate)

    # 窓関数をかける
    time_array, acf = window_func(split_data, Fs, N_ave, window_type=window_type)

    # 作成した関数を実行：FFTをかける
    fft_array, fft_axis_out, fft_spectrum_mean_out = fft_average(time_array, samplerate, Fs, N_ave, acf, spectrum_type)

    return fft_array, fft_axis_out, fft_spectrum_mean_out, final_time

def FFT_main_corr(t, data, Fs, samplerate, overlap_rate, window_type, correction_mode):
    print("Execute FFT")

    # オーバーラップ抽出された時間波形配列
    time_array, N_ave, final_time = overlapping(data, samplerate, Fs, overlap_rate)

    # 窓関数をかける
    time_array, acf = window_func(time_array, Fs, N_ave, window_type=window_type)

    #特殊な補正つきFFTの実行
    fft_array, fft_axis_out, fft_spectrum_mean_out = fft_average_corr(time_array, samplerate, Fs, N_ave, acf, correction_mode)

    return fft_array, fft_axis_out, fft_spectrum_mean_out, final_time


def plotting(fft_array, fft_axis_out, fft_spectrum_mean_out, final_time, samplerate, t, data, title, path):
    import matplotlib.pyplot as plt
    fft_array = fft_array.T
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
    cbar.set_label('Amplitude [V]')# Sound Pressure[Pa]やAcceleration[m/s^2]など用途に応じて

    # 軸設定する。
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [Hz]')

    # スケールの設定をする。
    ax1.set_xticks(np.arange(0, np.max(t), 1))# 1秒ごと。サンプリングレートに応じて何秒ごとにticksするか考えること
    ax1.set_yticks(np.arange(0, samplerate, 10))# 0から最大周波数。もしくは解析周波数の上限値
    ax1.set_xlim(0, np.max(t))
    ax1.set_ylim(0, samplerate / 2)
    plt.savefig(path + "\\" + title, dpi=300)
    # グラフを表示する。
    #plt.show()
    #plt.close()



if __name__ == '__main__':
    '''
    from matplotlib import pyplot as plt
    samplerate = 100#25600 [Hz]
    x = np.arange(0, 50.01, 1/samplerate)
    data = 2 * np.sin(2 * np.pi * 20 * x) + 5 * np.sin(2 * np.pi * 1 * x)
    split_rate = 0.1
    overlap_rate = 70  # オーバーラップ率
    delta_f = 0.2# 周波数分解能(自分で決める)
    Fs = int(samplerate / delta_f)

    fft_array, fft_axis_out, fft_spectrum_mean_out = FFT_main(x, data, Fs, samplerate, overlap_rate, window_type="hanning", spectrum_type="PSD")
    plt.plot(fft_axis_out, fft_spectrum_mean_out)
    plt.show()
    '''


    from scipy import signal
    import matplotlib.pyplot as plt

    # Fsとoverlapでスペクトログラムの分解能を調整する。
    '''
    全データ点数 = 入力データの長さ / サンプリング周波数
    フレームサイズ = 入力データの長さ×入力データ分割レート(入力データをどのくらいの割合で分割するか)
    フレーム周期 = フレームサイズ / サンプリング周波数 
    周波数分解能 = サンプリング周波数 / フレームサイズ = 44100 / 4096 = 10.76Hz
                = サンプリング周波数 / (入力データの長さ * 入力データ分割レート)
    周波数分解能が決まれば、自動的にオーバーラップに関する設定が決まる。            
    
    このテストの場合は、周波数分解能が10Hz位になるように入力データ分割レートを決める。
    10Hz = サンプリング周波数 / Fs
    Fs = サンプリング周波数 / 10Hz = 44100 / 10 = 4410
    2の累乗の大きさが望ましいため、とりあえず4096にセットしておく
    fft_array_out, fft_axis_out, fft_spectrum_mean_out, final_time = FFT_main_corr(x, data, Fs, samplerate, overlap_rate, "hanning", "Simple")及び
    fft_array, fft_axis_out, fft_spectrum_mean_out = FFT_main(x, data, Fs, samplerate, overlap_rate, window_type="hanning", spectrum_type="AMP")は同じ結果となる
    SFFTによるスペクトログラムを無補正かつリニア表示する場合はFFT_mainを使えばよい
    一方で、何らかの重みづけやdB変換をする場合はFFT_main_corrを使用する。
    '''
    delta_f = 10

    Fs = 4096 # フレームサイズ(4096がデフォルト)
    # 波形を作る
    samplerate = 44100
    t_max = 5

    overlap_rate = 90

    x = np.arange(0, t_max, 1/samplerate)
    # 周波数が時間で変化する正弦波の生成
    # 開始周波数f0, 停止周波数f1, 停止時間t1、スイープ手法method(linear, logarithmicなど)
    data = signal.chirp(x, f0=1, f1=500, t1=5, method='linear')

    fft_array, fft_axis_out, fft_spectrum_mean_out, final_time = FFT_main_corr(x, data, Fs, samplerate, overlap_rate, "hanning", "Simple")


    title = "Test_Spect"
    path = "D:\\PycharmProjects\\AI\\AnalysisTool\\Output\\FFT"
    plotting(fft_array, fft_axis_out, fft_spectrum_mean_out, final_time, samplerate, x, data, title, path)
    print()
