def cwt_morlet(t, signal, Fs, omega0, freqs, saveas_name='', viewflag=False, saveflag=False):
    '''

    :param t: 時系列ベクトル
    :param signal: 信号ベクトル
    :param Fs: ウェーブレットスケーリング係数
    :param omega0: マザーウェアーブレット(Morlet)周波数
    :param freqs: 疑似周波数ベクトル
    :param saveas_name:
    :param viewflag:
    :param saveflag:
    :return:
    '''
    from swan import pycwt
    import numpy as np
    r = pycwt.cwt_f(signal, freqs, Fs, pycwt.Morlet(omega0))
    rr = np.abs(r)#もしログにしたければdecibel.linear2db(x, dBref, mode='linear'を使うとよい)
    if viewflag or saveflag:
        heatmap_plot(t, signal, freqs, rr, saveas_name, viewflag, saveflag)

    return rr


def heatmap_plot(t, signal, freqs, rr, saveas_name, viewflag, saveflag):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams['figure.figsize'] = (20, 6)
    fig = plt.figure()

    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.6], sharex = ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    ax1.plot(t, signal, 'k')

    img = ax2.imshow(np.flipud(rr), extent=[t[0], t[-1], freqs[0], freqs[-1]],
                     aspect='auto', interpolation='nearest')# np.flipud -> 上下反転する

    ax2.set_yscale('log')
    ax2.tick_params(which='both', labelleft=False, left=False)
    ax2.tick_params(which='both', labelleft=True, left=True, labelright=False)
    fig.colorbar(img, cax=ax3)

    if viewflag:
        plt.show()
    if saveflag:
        plt.savefig(saveas_name + '.png', dpi=300)