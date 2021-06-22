import numpy as np
import matplotlib.pyplot as plt

def FFT_main(t, x, dt, split_t_r, overlap, window_F, output_FN, y_label, y_unit):
    '''
    入力信号を分割し、オーバーラップさせる。
    その後、窓を適用し信号を変形させる。
    FFT処理を行ったら、
    :param t:
    :param x:
    :param dt:
    :param split_t_r:
    :param overlap:
    :param window_F:
    :param output_FN:
    :param y_label:
    :param y_unit:
    :return:
    '''

    #解析対象のデータをオーバーラップして分割
    split_data = data_split(t, x, split_t_r, overlap)

    #分割した時系列データにそれぞれに対してFFT処理
    FFT_result_list = []
    for split_data_cont in split_data:
        FFT_result_cont = FFT(split_data_cont, dt, window_F)# FFTを実行する。この時、分割された時系列波形にウィンドウを適用
        FFT_result_list.append(FFT_result_cont)


    #各フレームのグラフ化処理
    '''
    IDN = 0
    for split_data_cont, FFT_result_cont in zip(split_data, FFT_result_list):
        IDN = IDN+1
        plot_FFT(split_data_cont[0], split_data_cont[1], FFT_result_cont[0], FFT_result_cont[1], output_FN, IDN, 0, y_label, y_unit)

    '''
    #平均化処理
    fq_ave = FFT_result_list[0][0]
    Amp_Spectrum_ave = np.zeros(len(fq_ave))
    for i in range(len(FFT_result_list)):
        Amp_Spectrum_ave = Amp_Spectrum_ave + FFT_result_list[i][1]
    Amp_Spectrum_ave = Amp_Spectrum_ave/(i+1)

    plot_FFT(t, x, fq_ave, Amp_Spectrum_ave, output_FN, "ave", 1, y_label, y_unit)

    return fq_ave, Amp_Spectrum_ave

def plot_FFT(t, x, fq, Amp_Spectrum, output_FN, IDN, final_graph, y_label, y_unit):
    fig = plt.figure(figsize=(12, 4))
    ax2 = fig.add_subplot(121)
    title1 = "Historical Data"
    plt.plot(t, x)
    plt.xlabel("time [s]")
    plt.ylabel(y_label+"["+y_unit+"]")
    plt.title(title1)

    ax2 = fig.add_subplot(122)
    title2 = "FFT Result"
    plt.xlabel('freqency(Hz)')
    plt.ylabel(y_label+"["+y_unit+"/sqrt(Hz)]")
    plt.xscale("log")
    plt.yscale("log")
    plt.loglog(fq, Amp_Spectrum)
    plt.title(title2)
    plt.grid(which='minor', color='white', linestyle='--')
    plt.grid(which='major', color='white', linestyle='--')


    if final_graph == 0:
        plt.savefig(output_FN[:-4]+"_"+str(IDN)+"_FFT"+output_FN[-4:], dpi=300)
    elif final_graph == 1:
        plt.savefig(output_FN, dpi=300)

    return 0

def FFT(data_input, dt, window_type, FFT_res_mode = "Amp_Spect"):

    N = len(data_input[0])

    #窓の設定
    if window_type == "hanning":
        window = np.hanning(N)          # ハニング
    elif window_type == "hamming":
        window = np.hamming(N)          # ハミング
    elif window_type == "blackman":
        window = np.blackman(N)         # ブラックマン
    elif window_type == "bartlett":
        window = np.bartlett(N)         # バートレット
    elif window_type == "kaiser":
        alpha = 0#0:矩形、1.5:ハミング、2.0:ハニング、3:ブラックマンに似た形
        Beta = np.pi * alpha
        window = np.kaiser(N, Beta)
    else:
        print("Error: input window function name is not sapported. Your input: ", window_type)
        print("Hanning window function is used.")
        hanning = np.hanning(N)          # ハニング

    #窓関数後の信号
    x_windowed = data_input[1]*window# Windowによって入力信号が変形する

    #FFT計算
    F = np.fft.fft(x_windowed)  # 窓関数が適用された時系列信号でFFTを導出。虚数

    #正規化
    #正しいロジックの実装必要
    #F = F / (N / 2)

    # 振幅スペクトルを出す場合
    # OK
    if FFT_res_mode == "Amp_Spect":
        F_abs = np.abs(F)# 絶対値をとる
        Amp_Spectrum = F_abs / (N//2)  # Fをデータ点数/2で割ることで振幅スペクトルを取得

        fq = np.linspace(0, 1.0/dt, N)
        #窓補正
        acf=1/(sum(window)/N)# 1/(sum(窓関数=window)/サンプリング点数)
        Amp_Spectrum = acf*Amp_Spectrum# 最終的な振幅スペクトル。ただしナイキスト定数以降の信号含む。窓補正を使って振幅補正を実施

        #ナイキスト定数まで抽出
        fq_out = fq[:int(N/2)+1]
        Amp_Spectrum_out = Amp_Spectrum[:int(N//2)+1]

        return [fq_out, Amp_Spectrum_out]# ナイキスト定数(N/2)まで抽出した周波数、振幅スペクトル


    # パワースペクトルを出す場合
    # なんかおかしい？fft(data)**2じゃない？
    if FFT_res_mode == "Power_Spect":
        F_abs = np.abs(F)# 絶対値をとる
        Power_Spectrum = (F**2)/(N//2)# Fをデータ点数/2で割ることでパワースペクトルを取得。振幅スペクトルの二乗
        fq = np.linspace(0, 1.0/dt, N)
        #窓補正
        acf=1/(sum(window)/N)
        Power_Spectrum = acf*Power_Spectrum# 最終的なパワースペクトル。ただしナイキスト定数以降の信号含む

        #ナイキスト定数まで抽出
        fq_out = fq[:int(N/2)+1]
        Power_Spectrum_out = Power_Spectrum[:int(N//2)+1]

        return [fq_out, Power_Spectrum_out]# ナイキスト定数(N/2)まで抽出した周波数、パワースペクトル





def data_split(t, x, split_t_r, overlap):

    split_data = []
    one_frame_N = int(len(t)*split_t_r) #1フレームのサンプル数
    overlap_N = int(one_frame_N*overlap) #オーバーラップするサンプル数
    start_S = 0
    end_S = start_S + one_frame_N

    while True:
        t_cont = t[start_S:end_S]
        x_cont = x[start_S:end_S]
        split_data.append([t_cont, x_cont])

        start_S = start_S + (one_frame_N - overlap_N)
        end_S = start_S + one_frame_N

        if end_S > len(t):
            break


    return np.array(split_data)

if __name__ == "__main__":
    t = np.arange(0, 50.01, 0.01)
    #x = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    x = np.sin(2*np.pi*1*t)


    dt = 0.01 #This value should be correct as real.
    output_FN = "test.png"

    split_t_r = 0.1 #1つの枠で全体のどの割合のデータを分析するか
    overlap = 0.5 #オーバーラップ率
    window_type = "hamming" #窓関数選択: hanning, hamming, blackman, bartlet, kaiser
    y_label = "amplitude"
    y_unit = "V"
    FFT_main(t, x, dt, split_t_r, overlap, window_type, output_FN, y_label, y_unit)
