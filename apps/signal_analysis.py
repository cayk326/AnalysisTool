import numpy
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import sys
from AnalysisTool.engikit.signalprocessing import FFT_lib, wavelet, filter, FFT_lib
from AnalysisTool.engikit.util import  prepro, grabinfo

class Analyzer:
    def __init__(self):
        self.confpath = "D:\\PycharmProjects\\AI\\AnalysisTool\\conf\\config.json"
        self.config = grabinfo.jsonfileparser(self.confpath)
        self.all_files_path = None


    def filtering(self, df, samplerate, labellist):
        filtered_df = df.copy()
        if self.config["FilterSetting"]["mode"] == "butter":
            for idx, labelname in enumerate(labellist):
                filtered_df[labelname] = filter.butterlowpass(x=df[labelname], fpass=self.config["FilterSetting"]["fpass"],
                                                              fstop=self.config["FilterSetting"]["fstop"],
                                                              gpass=self.config["FilterSetting"]["gpass"],
                                                              gstop=self.config["FilterSetting"]["gstop"],
                                                              fs=1 / samplerate,
                                                              dt=samplerate,
                                                              checkflag=False,
                                                              labelname=labelname)
        return filtered_df

    def search_header_pos(self, analyzer, columns, env):
        if env == None:
            print("Please select environment")
            return False
        from collections import OrderedDict
        self.config["GenLabelNames"] = OrderedDict()

        for header in self.config["LabelNames"].keys():
            for name in self.config["LabelNames"][header][env]["Name"]:
                print("Search header name..." + name + " in data")
                if name in columns:
                    name_idx = columns.get_loc(name)
                    print("Found " + name + "at " + str(name_idx))
                    self.config["GenLabelNames"][name] = name_idx
                    break
        return


    def analysis(self, signals_df, path):
        header_name = [header for header in self.config["GenLabelNames"].keys()]
        analysis_label = header_name.copy()
        Time = analysis_label.pop(0)#Timeラベルのみ抽出

        sample_time = signals_df[Time][1] - signals_df[Time][0]
        delta_f = self.config["FFTSetting"]["delta_f"]
        overlap_rate = self.config["FFTSetting"]["overlap_rate"]
        window_mode = self.config["FFTSetting"]["window_mode"]["Primary"]
        y_label = "acceleration"
        y_unit = "^(2)"
        samplerate = int(1 / sample_time)
        samplerate_trans = self.config["ResampleSetting"]["samplerate_trans"]


        OUTPUT_PATH = self.config["System"]["OutputFileDir"]

        # ----- Filtering ----- #
        if eval(self.config["Analysis_Flags"]["Filter"]):
            output_df = Analyzer.filtering(self, signals_df, sample_time, analysis_label)
        else:
            output_df = signals_df

        # ----- Export filtered data ----- #
        if eval(self.config["Export_Flags"]["Filter"]):
            print("Export filtered data...")
            output_df.to_csv((path.replace("Input", "Output\\filtered")).replace(".csv", "_filt.csv"), index = False)
            print("Exporting finished!")


        # ----- FFT ----- #
        #fft_spectrum_mean_out, fft_axis_out = [], []
        FFT_result_df = {}
        fft_array, fft_axis_out, fft_spectrum_mean_out, final_time = [], [], [], 0

        Fs = int(samplerate / delta_f)

        
        if eval(self.config["Analysis_Flags"]["FFT"]):
            for label in analysis_label:
                print('FFT Analyzing for ' + label + '...')
                fft_array, fft_axis_out, fft_spectrum_mean_out, final_time = FFT_lib.FFT_main(signals_df[Time].to_list(), signals_df[label].to_list(), Fs, samplerate, overlap_rate, window_mode, "AMP")
                FFT_result_df['f_' + label], FFT_result_df['PSD_' + label] = fft_axis_out, fft_spectrum_mean_out

                FFT_lib.plotting(fft_array, fft_axis_out, fft_spectrum_mean_out, final_time, samplerate,
                                 signals_df[Time].to_list(), signals_df[label].to_list(), label + "_Spectrogram", OUTPUT_PATH + "\\FFT")


        FFT_result_df = pd.DataFrame(FFT_result_df)# Data Frame化する
        FFT_result_df.to_csv((path.replace("Input", "Output\\FFT")).replace(".csv", "_FFT.csv"), index=False)

        resample_df = prepro.resample(output_df, samplerate, 1000, Time, "s")

        '''
        if eval(self.config["System"]["Export_Flags"]["Resample"]):
            print("Resampling")
            resample_df = output_df.copy()
            resample_df = ""
        '''






def main():
    workdir = r'D:\PycharmProjects\AI\Analyzer'
    inputdir = workdir + '/' + 'Input'
    outputdir = workdir + '/' + 'Output'
    header_pos = 0
    encoding = 'utf-8'
    chunksize = 10000

    analyzer = Analyzer()

    # Get all file path #
    all_files_path = prepro.GetAllFileList(analyzer.config["System"]["InputFileDir"], "*.csv")


    for path in all_files_path:
        # ----- Read All files ----- #
        chunks = pd.read_csv(path,
                             chunksize=chunksize,
                             encoding=analyzer.config["System"]["Encoding"],
                             sep=analyzer.config["System"]["Deliminator"]["CSV"],
                             header=analyzer.config["System"]["Header_pos"]["CaseA"])
        signals_df = pd.concat((data for data in chunks), ignore_index = True)
        print("data frame memory_size:{0}".format(sys.getsizeof(signals_df) / 1000000) + "[MB]")

        # ヘッダー名の辞書と入力ファイルのヘッダー名を引き当てる
        analyzer.search_header_pos(analyzer, signals_df.columns, "ENV1")

        analyzer.analysis(signals_df, path)



if __name__ == '__main__':
    main()