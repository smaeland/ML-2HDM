

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
import pickle

plt.style.use('../../plot/paper.mplstyle')
from matplotlib import rcParams



def comparison(datasets, method):
        
    defaultfontsize = rcParams['font.size']
    rcParams['font.size'] = 14

    f, (axes) = plt.subplots(2, sharex=True, sharey=False, figsize=(5.5, 5.3))

    preds = []
    for dset in datasets:
        with open(dset + "/results_predictions_" + method +".pkl") as hin:
            preds.append(pickle.load(hin))

    for i, pred in enumerate(preds):
        
        diff = pred[0.7]
        thisdiff = np.array(diff)
        mean = np.mean(thisdiff)
        std = np.std(thisdiff)
        
        if 'mle' in method and i == 0:
            thisdiff = thisdiff[np.where((thisdiff >= 0.7-1.1) & (thisdiff <= 0.7+1.1))]
            mean = np.mean(thisdiff)
            std = np.std(thisdiff)
        
        nbins = 50 if i == 0 else 100
        xlow, xhigh = -0.3, 1.7
        axes[i].axvspan(xlow, 0, alpha=0.15, color='gray')
        axes[i].axvspan(1, xhigh, alpha=0.15, color='gray')
        axes[i].hist(diff, bins=nbins, range=(xlow, xhigh), color='#39568C', alpha=0.7, label=r'$\alpha$ (test) = 0.7f')
        textloc = (0.2, 0.8) if mean > 0.8 else (0.75, 0.8)
        axes[i].text(textloc[0]+0.0216, textloc[1], r'$n = %d$' % (20 if i == 0 else 500), transform=axes[i].transAxes, fontsize=14)
        axes[i].text(textloc[0]+0.0216, textloc[1]-0.10, r'$\alpha = 0.7$', transform=axes[i].transAxes, fontsize=14)
        axes[i].text(textloc[0], textloc[1]-0.22, r'$\mu_{\alpha} = %.2f$' % mean, transform=axes[i].transAxes, fontsize=14)
        axes[i].text(textloc[0], textloc[1]-0.32, r'$\sigma_{\alpha} = %.2f$' % std, transform=axes[i].transAxes, fontsize=14)
        axes[i].set_xlim([xlow, xhigh])
        
        if i == 0:
            axes[i].set_ylim([0, 700])
            axes[i].yaxis.set_ticks(np.arange(0, 700, 200))
        else:
            axes[i].set_ylim([0, 950])
            axes[i].yaxis.set_ticks(np.arange(0, 950, 200))
            

        xvals = np.linspace(xlow, xhigh, 200)
        axes[i].plot(xvals, norm.pdf(xvals, loc=mean, scale=std)*len(thisdiff)/nbins*(xhigh-xlow), color='#440154', linewidth=1.5)
        
        axes[i].set_ylabel('Test sets / %.2f' % ((xhigh-xlow)/nbins))

        f.subplots_adjust(hspace=0, top=0.99, right=0.99, left=0.17)
        plt.xlabel(r'$\hat{\alpha}$')
        plt.ylabel('Test sets / %.2f' % ((xhigh-xlow)/nbins))
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        f.show()

        f.savefig('results_%s_20_vs_500_events.pdf' % method)
        rcParams['font.size'] = defaultfontsize
        
comparison(['../without_background/results_20events_10000sets', '../without_background/results_500events_10000sets'], 'phistar')
comparison(['../without_background/results_20events_10000sets', '../without_background/results_500events_10000sets'], 'mle')
plt.show()
