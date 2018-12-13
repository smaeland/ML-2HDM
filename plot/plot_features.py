"""
Div plots
"""

import sys
from argparse import ArgumentParser
from collections import OrderedDict
import math
import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# -- All features
# 'piplus.px' 'piplus.py' 'piplus.pz' 'piplus.e'
# 'pi0plus.px' 'pi0plus.py' 'pi0plus.pz' 'pi0plus.e'
# 'piminus.px' 'piminus.py' 'piminus.pz' 'piminus.e'
# 'pi0minus.px' 'pi0minus.py' 'pi0minus.pz' 'pi0minus.e'
# 'met_x' 'met_y' 'inv_mass' 'transv_mass'
# 'upsilon_plus' 'upsilon_minus' 'upsilonp_x_upsilonm'
# 'triple_corr' 'phistar'
# 'piplus_ip_x' 'piplus_ip_y' 'piplus_ip_z'
# 'piminus_ip_x' 'piminus_ip_y' 'piminus_ip_z'
# 'pip_pim_angle' 'pi0p_pi0m_angle'
# 'pip_dot_pi0p' 'pim_dot_pi0m' 'pim_dot_pip'
# 'pip_dot_pi0m' 'pim_dot_pi0p'
#No range for name pip_pim_angle
#No range for name pi0p_pi0m_angle

plt.rcParams['mathtext.fontset'] = 'cm'

def getrange(name):
    """
    Get nbins and range depending on feature name
    """

    # First matching occurence is returned
    ranges = OrderedDict()
    nbins = 80
    ranges['piminus.pz'] = (nbins, 0, 250)
    ranges['pi0minus.pz'] = (nbins, 0, 250)
    ranges['piplus.pz'] = (nbins, -250, 0)
    ranges['pi0plus.pz'] = (nbins, -230, 0)
    ranges['piplus.px'] = (nbins, 0, 0.815)
    ranges['piminus.px'] = (nbins, -0.745, 0.745)
    ranges['.px'] = (nbins, -0.85, 0.815)
    ranges['.py'] = (nbins, -0.85, 0.815)
    ranges['.e'] = (nbins, 0, 300)
    #ranges['met'] = (nbins, -700, 700)
    ranges['met'] = (nbins, -200, 200)
    ranges['inv_mass'] = (nbins, 0, 700)
    ranges['transv_mass'] = (nbins, 0, 700)
    ranges['upsilon_'] = (nbins, -1, 1)
    ranges['upsilonp_x_upsilonm'] = (nbins, -0.9, 0.9)
    ranges['phistar'] = (nbins, 0, 2*math.pi)
    ranges['pim_dot_pi0p'] = (nbins, -50e3, 0)
    ranges['pip_dot_pi0m'] = (nbins, -50e3, 0)
    ranges['pim_dot_pip'] = (nbins, -50e3, 0)
    ranges['pim_dot_pi0m'] = (nbins, 0, 20e3)
    ranges['pip_dot_pi0p'] = (nbins, 0, 20e3)
    ranges['piplus_ip_'] = (nbins, -1e3, 1e3)
    ranges['piminus_ip_'] = (nbins, -1e3, 1e3)
    ranges['triple_corr'] = (nbins, -220, 220)
    ranges['pip_pim_angle'] = (nbins, -5, 5)
    ranges['_angle'] = (nbins, -5.5, 5.5)

    for key in ranges:
        if key in name:
            return ranges[key]

    print 'No range for name', name
    return (None, None, None)

def plot_feature(datas, datalabels, varname, xlabel, colors, ylabel=r'$\mathrm{Events}$'): #Todo: label: events/binning
    """
    Make a single 1D histogram of input data
    """

    linestyles = ('solid', 'dashed')

    nbins, low, high = getrange(varname)
    if high is not None and low is not None:
        rng = (low, high)
    else:
        rng = None

    try:
        binwidth = (rng[1] - rng[0]) / float(nbins)
    except:
        binwidth = -999

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[4, 1]})
    fig.canvas.set_window_title(varname)

    hists = []
    bins = []
    ratio_limits = [1, 1]

    for k, data in enumerate(datas):

        # Main plot
        hn, binedges, _ = ax1.hist(data, bins=nbins, range=rng,
                                   histtype='step',
                                   linestyle=linestyles[k],
                                   color=colors[k]
                                   )
        ax1.plot([0], linestyle=linestyles[k], label=datalabels[k], c=colors[k])

        hists.append(hn)
        bins.append(binedges)

        # Ratio plot
        if (k > 0):
            ratio = np.array(hn, dtype=np.float) / hists[0]
            ratio = np.concatenate((ratio, [ratio[-1]]))
            ratio = np.nan_to_num(ratio)
            ax2.step(binedges, ratio, color='#20A387')
            if max(ratio) > ratio_limits[1] and max(ratio) < 1e2:
                ratio_limits[1] = max(ratio)
            if min(ratio) < ratio_limits[0]:
                ratio_limits[0] = min(ratio)

    ax1.set_xlim([low, high])
    ax1.set_autoscale_on(False)
    #ax2.set_ylim([ratio_limits[0]*0.7, ratio_limits[1]*1.3])
    ax2.set_ylim([0.2, 1.9])
    ax2.set_autoscale_on(False)
    #ax2.axhline(y=1.0, xmin=low, xmax=high, linewidth=1.0, linestyle='--', color='black')
    ax2.axhline(y=1.0, xmin=-1000, xmax=1000, linewidth=1.0, linestyle='--', color='black')
    #print 'axh: var = %s, low = %f, high = %f' % (varname, low, high)

    fig.subplots_adjust(hspace=0.05, top=0.95, bottom=0.13, left=0.13, right=0.965)

    ax1.legend(loc=('upper left' if varname == 'pi0plus.pz' else 'upper right'))
    plt.xlabel(xlabel, fontsize=36)
    ax1.set_ylabel('Events / %.2f [1/GeV]' % binwidth)
    ax2.set_ylabel('Ratio')

    if varname == 'phistar':
        ax1.set_ylim([0, 1.7*ax1.get_ylim()[1]])
        plt.xlabel(r'$\varphi^{*}$', fontsize=40)
        ax1.set_ylabel('Events / %.2f' % binwidth)

    if 'upsilon_' in varname:
        ax1.set_xlim([-0.95, 0.95])
        ax1.set_ylim([0, 65000])
        plt.xlabel(xlabel, fontsize=40)
        ax1.set_ylabel('Events / %.2f' % binwidth)
    
    if varname == "triple_corr":
        plt.xlabel(r'$\mathscr{O}^{*}$', fontsize=40)
        ax1.set_ylabel('Events / %.2f' % binwidth)

    if varname == "upsilonp_x_upsilonm":
        ax1.set_ylabel('Events / %.2f' % binwidth)
    
    if 'met_' in varname:
        ax1.set_xlim([-200, 200])

    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    fig.show()

if __name__ == '__main__':


    if matplotlib.__version__ <= '1.5.3':
        print 'Can\'t use style sheets with version', matplotlib.__version__
    else:
        plt.style.use('paper.mplstyle')

    # Import data
    PARSER = ArgumentParser(description='Plot and compare input features')
    PARSER.add_argument('inputs', nargs='+', type=str, help='List of input .h5 files')
    PARSER.add_argument('-s4d', action='store_true', help='s4d proceedings plots')

    ARGS = PARSER.parse_args()

    for fname in ARGS.inputs:
        if 'H' in fname:
            hf_H0 = h5py.File(fname)
            data_H0 = hf_H0.get('data')
        elif 'A' in fname:
            hf_A0 = h5py.File(fname)
            data_A0 = hf_A0.get('data')

    if ARGS.s4d:
        C1 = '#992667'  # purple
        C2 = '#078600'  # green 

        FEATURETITLES = [r'$p_{x}^{\pi^{+}}$ (x-component of $\pi^{+}$ momentum)',#
                         r'$\pi^{+} p_{y}$',
                         r'$\pi^{+} p_{z}$',
                         r'$\pi^{+} E$',
                         r'$\pi^{0+} p_{x}$', r'$\pi^{0+} p_{y}$', r'$\pi^{0+} p_{z}$',
                         r'$\pi^{0+} E$', r'$\pi^{-} p_{x}$', r'$\pi^{-} p_{y}$',
                         r'$\pi^{-} p_{z}$',
                         r'$\pi^{-} E$', r'$\pi^{0-} p_{x}$', r'$\pi^{0-} p_{y}$',
                         r'$\pi^{0-} p_{z}$',
                         r'$\pi^{0-} E$',
                         #r'$E_{T}^{\mathrm{miss}}\;x$',
                         r'$E_{x}^{\mathrm{miss}}$ (missing energy in the $x$-direction)',
                         #r'$E_{T}^{\mathrm{miss}}\;y$',
                         r'$E_{y}^{\mathrm{miss}}$',
                         #r'$m_{\tau\tau}^{\mathrm{vis}}$',
                         r'$m_{\tau\tau}^{\mathrm{vis}}$ (invariant mass of the $\tau\tau$ pair)',
                         r'$m_{T}^{\mathrm{tot}}$',
                         r'$\Upsilon^{+}$',
                         r'$\Upsilon^{-}$',
                         r'$\Upsilon^{+} \cdot \Upsilon^{-}$', r'$\mathcal{O}^{*}$',
                         r'$\varphi^{*}_{\mathrm{CP}}$', r'$\pi^{+} IP x$', r'$\pi^{+} IP y$',
                         r'$\pi^{+} IP z$', r'$\pi^{-} IP x$', r'$\pi^{-} IP y$', r'$\pi^{-} IP z$',
                         r'$\mathrm{atan2}(pi^{-} p_{y}, pi^{-} p_{x})$',
                         r'$\mathrm{atan2}(pi^{0-} p_{y}, pi^{0-} p_{x})$',
                         r'$pi^{+} \cdot pi^{0+}$', r'$pi^{-} \cdot pi^{0-}$',
                         r'$pi^{-} \cdot pi^{+}$', r'$pi^{+} \cdot pi^{0-}$',
                         r'$pi^{-} \cdot pi^{0+}$', ',Target',]


    else:
        C1 = '#440154' # purple
        C2 = '#39568C' # blue

        FEATURETITLES = data_H0.attrs['feature_titles']

    FEATURENAMES = data_H0.attrs['feature_names']
    assert np.array_equiv(FEATURENAMES, data_A0.attrs['feature_names'])

    COLORS = (C1, C2)

    # -- Pick which features to plot
    import fnmatch
    for n, _ in enumerate(FEATURENAMES):

        features_to_plot = [
            'piminus.px',
            #'piplus.px',
            'pi0minus.pz',
            #'pi0plus.pz',
            'transv_mass',
            #'*_mass',
            #'met_x',
            #'inv_mass',
            'upsilon_minus',
            #'upsilonp_x_upsilonm',
            'triple_corr',
            #'piminus.px', 'pi0minus.pz',
            #'piplus.px', #'pi0plus.pz',
            'phistar',
            ]

        matches = [fnmatch.fnmatch(FEATURENAMES[n], pattern) for pattern in features_to_plot]

        if True not in matches:
            continue

        #if not FEATURENAMES[i] in features_to_plot: continue

        #if not 'mass' in FEATURENAMES[i]: continue

        x_H0 = np.concatenate(data_H0[:, [n]])
        x_A0 = np.concatenate(data_A0[:, [n]])
        plotname = FEATURENAMES[n]
        xtitle = FEATURETITLES[n].replace('0-', '0(-)').replace('0+', '0(+)')
        xtitle += ' [GeV]' if ('p_{' in xtitle or 'm_{' in xtitle or 'E' in xtitle) else ''

        plot_feature([x_H0, x_A0], [r'$H$', r'$A$'], plotname, xtitle, COLORS)

    plt.show()
