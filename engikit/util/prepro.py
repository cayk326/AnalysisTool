import pandas as pd
import natsort
import numpy as np
import os
import glob
import time

def GetAllFileList(file_path, ext):
    all_files = natsort.natsorted((glob.glob(os.path.join(file_path, ext))))
    return all_files

def ConstractDataFrame(all_files, header_pos):
    data_length_list = np.arange(all_files.__len__())
    for i in range(all_files.__len__()):
        data_length_list[i] = pd.read_csv(all_files[i], header=header_pos).__len__()
    each_file = (pd.read_csv(f, header=header_pos) for f in all_files)
    dataframe = pd.concat(each_file, ignore_index=True, sort=False, join='inner')
    return dataframe, data_length_list


def get_length_bigdata(filepath, chunksize, encoding, sep, header):
    chunks = pd.read_table(filepath, chunksize=chunksize, encoding=encoding, sep=sep, header=header)
    return len(pd.concat((data for data in chunks), ignore_index=True))


def calctime(func):
    start = time()
    r = func()
    return {'value': r, 'time': time() - start}


def Offset(df, label, start, end, offsetval):
    print('offsetting...')
    print('start=' + str(start) + '\n' + 'end=' + str(end) + '\n' + 'offsetval=' + str(offsetval) + '\n')
    df.iloc[start:end, label] = df.iloc[start:end, label] - offsetval

    return df

def Isneedoffset(df, labelpos, start, end, mode):

    if mode == 'mean':
        return np.mean(df.iloc[start:end, labelpos]) if np.mean(df.iloc[start:end, labelpos]) != 0 else 0
    elif mode == 'median':
        return np.median(df.iloc[start:end, labelpos]) if np.median(df.iloc[start:end, labelpos]) != 0 else 0
    else:
        print('Isneedoffset could not find certain mode name.')
        return -1


def AddMulNewColumn(df, labelname, lengthlist, val):
    print('Add new label')
    loc = 0
    array = np.full(sum(lengthlist), 0)
    for i in range(lengthlist.__len__()):

        if i == 0:
            print('Apply val into array')
            array[loc:lengthlist[i]] = array[loc:lengthlist[i]] + val[i]
            loc = lengthlist[i]
        else:
            print('Apply val into array')
            array[loc:lengthlist[i] + loc] = array[loc:lengthlist[i] + loc] + val[i]
            loc = loc + lengthlist[i]
    print('merge array against dataframe with label name')
    df[labelname] = array
    return df


def resample(df, fs, fs_trans, Time, unit = "s"):
    df[Time] = pd.to_datetime(df[Time], unit = unit)
    df = df.set_index(Time)

    if fs != 1000:
        df = df.resample('1ms').interpolate()
    else:
        trans_dt = 1 / fs_trans
        trans_dt_ms = int(trans_dt * 1000)
        df = df.asfreq(str(trans_dt_ms) + 'ms')# Resampling
        df = df.resample(str(trans_dt_ms) + 'ms').interpolate()
        df.reset_index(inplace = True)
        df[Time] = df[Time].astype('int64' / 10 ** 9) # ns -> s

    return df
