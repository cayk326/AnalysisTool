import numpy
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import sys
from AnalysisTool.engikit.signalprocessing import FFT, wavelet, filter
from AnalysisTool.engikit.util import  prepro

def filitering(df, labellist, fs, samplingrate, fpass, fstop, gpass, gstop, checkflag, mode="butter"):
    filtered_df = df.copy()
    if mode == "butter":
        for idx, labelname in enumerate(labellist):
            filtered_df[labelname] = filter.butterlowpass(x=df[labelname], fpass=fpass, fstop=fstop, gpass=gpass,
                                                          gstop=gstop, fs=fs, dt=samplingrate, checkflag=checkflag,
                                                          labelname=labelname)
    return filtered_df

def analysis(signals_df):
    header_name = signals_df.columns
    analysis_label = header_name.drop('Time')
    samplingrate = signals_df['Time'][1] - signals_df['Time'][0]
    split_t_r = 0.1 # 1つの枠で全体のどの割合のデータを分析するか
    overlap = 0.7# オーバーラップ率
    window_F = 'hamming'# 窓関数: hanning, hamming, blackman
    y_label = 'acceleration'
    y_unit = 'm/s^2'
    fpass = 8
    fstop = 10
    gpass = 3
    gstop = 40
    fs = 1 / samplingrate
    checkflag = True
    fq_ave, F_abs_amp_ave = {}, {}
    for label in analysis_label:
        fq_ave['f_'+label], F_abs_amp_ave['PSD_'+label] = FFT.FFT_main(signals_df["Time"], signals_df[label], samplingrate, split_t_r, overlap, window_F, str(label) + "png", y_label, y_unit)
        print('Analyzing for ' + label + '...')
    FFT_result_df = pd.concat([pd.DataFrame(fq_ave),pd.DataFrame(F_abs_amp_ave)], axis='columns')
    FFT_result_df.to_csv('FFT_result.csv')

    filtered_df = filitering(signals_df, analysis_label, fs, samplingrate, fpass, fstop, gpass, gstop, checkflag, mode="butter")
    print()


def main():
    workdir = r'D:\PycharmProjects\AI\Analyzer'
    inputdir = workdir + '/' + 'Input'
    outputdir = workdir + '/' + 'Output'
    header_pos = 0
    encoding = 'utf-8'
    chunksize = 10000

    all_files_path = prepro.GetAllFileList(inputdir, '*csv')
    data_len = []

    for i in range(len(all_files_path)):
        data_len.append(prepro.get_length_bigdata(all_files_path[i], chunksize=chunksize, encoding=encoding, sep=',', header=header_pos))


    # Analysis logic loop
    for i in range(len(data_len)):
        chunks = pd.read_csv(all_files_path[i], chunksize=chunksize, encoding=encoding, sep=',', header=header_pos)
        signals_df = pd.concat((data for data in chunks), ignore_index=True)
        print('data frame memory size:{0}'.format(sys.getsizeof(signals_df) / 1000000) + '[MB]')


    analysis(signals_df)


if __name__ == '__main__':
    main()