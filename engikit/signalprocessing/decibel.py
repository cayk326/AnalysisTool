import numpy as np

#リニア値からdBへ変換
def linear2db(x, dBref, mode='linear'):
    '''
    This function transform linear/power data to dB
    :param x: data which you want to transform to dB
    :param dBref: base value of transform.
    :param mode:
    :return: dB data
    dB = a * log10(data / base value)
    '''
    if mode == 'linear':
        return 20 * np.log10(x / dBref)
    elif mode == 'power':
        return 10 * np.log10(x / dBref)



def db2linear(x, dBref, mode='linear'):
    '''

    :param x:
    :param dBref:
    :param mode:
    :return:
    '''

    if mode == 'linear':
        return dBref * np.power(10, x / 20)
    elif mode == 'power':
        return dBref * np.power(10, x / 10)

def calcdB(x, target, D2, mode='linear'):
    '''

    :param x:
    :param target:
    :param D2:
    :param mode:
    :return:
    '''

    if mode == 'linear':
        return linear2db(target, x, mode='linear') + D2
    elif mode == 'power':
        return linear2db(target, x, mode='power') + D2
