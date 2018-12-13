# pylint: disable=C0303

"""
Run comparison of the different classification methods on signal only
"""

import numpy as np
from argparse import ArgumentParser
from glob import glob
from sklearn import utils
from sklearn.preprocessing import binarize
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import time
from scipy.stats import chi2, norm

from neuralnet.utilities import get_dataset_from_path, select_features
from measure_yields.basepredictors import Prediction
from measure_yields.phistar_predictor import PhistarPredictorBinned, PhistarPredictorUnbinned
from measure_yields.naive_network_predictor import NaiveNnPredictor
from measure_yields.strumkes_predictor import MostLikelyNnPredictorBinned, MaxLikelihoodNnPredictorBinned, MaxLikelihoodNnPredictorUnbinned, MostLikelyNnPredictorUnbinned

plt.style.use('../../plot/paper.mplstyle')
from matplotlib import rcParams



def create_mixed_set(ratio, ntotalevents, X_H_in, X_A_in, sample_theta, sample_ntot=False, shuffle=True, offset=0):
    """ 
    Create test set for given nA/nTot ratio
    
    Arguments:
        ratio: theta (= nA/nTot)
        ntotalevents: Total number of events
        X_H_in: Data for H (np.array)
        X_A_in: Data for A (np.array)
        sample_theta: Sample the theta (ratio) value
        sample_ntot: Sample total number of events from a Poisson distribution
            abount the input 'ntotalevents'
        shuffle: Shuffle so that returned data is random
        
    Returns:
        np.array
    """
    
    if sample_ntot:
        ntotalevents = np.random.poisson(ntotalevents)
    
    nA = int(np.round(ratio*ntotalevents))
    nH = int(np.round(ntotalevents - nA))
    
    if sample_theta:
        sampled_vals = np.random.random(ntotalevents).reshape(-1, 1)
        sampled_vals_int = binarize(sampled_vals, threshold=ratio).flatten()
        nH = int(np.sum(sampled_vals_int))
        nA = int(ntotalevents - nH)

    if shuffle:
        X_H = utils.shuffle(X_H_in)[offset:nH+offset]
        X_A = utils.shuffle(X_A_in)[offset:nA+offset]
    else:
        X_H = X_H_in[offset:nH+offset]
        X_A = X_A_in[offset:nA+offset]

    X = np.vstack((X_H, X_A))
    Y = np.concatenate((np.zeros(nH), np.ones(nA)))
    Y = to_categorical(Y, num_classes=2)

    return X, Y






if __name__ == '__main__':
    
    # Global var for S4D proceedings plot style
    global PLOTS_FOR_S4D
    PLOTS_FOR_S4D = False
    
    parser = ArgumentParser(description='Case 1')
    parser.add_argument('-tst', '--test', help='Predict for test set', action='store_true')
    parser.add_argument('-val', '--validation', help='Predict for validation set', action='store_true')
    parser.add_argument('-r', '--recreate', help='Recreate templates', action='store_true')
    parser.add_argument('-p', '--plot', help='Plot validation results', action='store_true')
    parser.add_argument('-b', '--binned', help='Run binned fits rather than unbinned', action='store_true')
    parser.add_argument('-rp', '--result_path', help='Directory to read results from', type=str, default='.')
    pargs = parser.parse_args()
    
    filepath_train = ''
    filepath_validation = ''
    filepath_test = ''
    
    # Instantiate predictors
    if pargs.test or pargs.validation or pargs.recreate:
    
        phi_pred = PhistarPredictorUnbinned('phistar', ['H', 'A'])
        if pargs.binned:
            phi_pred = PhistarPredictorBinned('phistar', 10, ['H', 'A'])
        
        pref = '../../neuralnet/train/'

        naive_nn = NaiveNnPredictor('NaiveNn_theta_train_0p5',
                                    pref+'keras_model_100ep.h5',
                                    pref+'scaler_common.pkl')
        
                
        mle_nn = MaxLikelihoodNnPredictorUnbinned(title='MLE_unbinned',
            saved_model=pref+'keras_model_100ep.h5',
            saved_scaler=pref+'scaler_common.pkl',
            sample_list=['H', 'A'])
        
        if pargs.binned:

            raise Exception('Binned fits not propely implemented, '
                'but code left here for reference')

            mle_nn = MaxLikelihoodNnPredictorBinned(title='MLE_binned',
                saved_model=pref+'keras_model_ratio_0p50.h5',
                saved_scaler='../../neuralnet/train/scaler_common.pkl',
                sample_list=['H', 'A'])
        
    
    # Create predictor templates
    if pargs.recreate:
        
        # Create templates from validation data, which the predictor has not
        # seen before
        if pargs.binned:
            print 'Creating training templates'
            phi_pred.create_train_templates('../../generate_samples/450GeV/validation')
            mle_nn.create_train_templates('../../generate_samples/450GeV/validation')
        
        else:
            phi_pred.create_pdfs('../../generate_samples/450GeV/validation')
            mle_nn.create_pdfs('../../generate_samples/450GeV/validation')
        
    
    # Test set specifications
    thetas_to_test = [0.5, 0.7, 0.9]
    n_expected_events = 100
    n_test_sets = 10000
    
    
    # Open test data
    if pargs.test or pargs.validation:
        XH, YH, features = get_dataset_from_path(
            '../../generate_samples/450GeV/test', 
            pattern='model_H_merged',
            include_mass=False)
            
        XA, YA, _ = get_dataset_from_path(
            '../../generate_samples/450GeV/test',
            pattern='model_A_merged',
            include_mass=False)
    
    
    # Run a single prediction, create plots
    if pargs.test:
        
        #for theta in thetas_to_test:
        for theta in [0.7]:
            
            print 'Running test on theta =', theta
            
            # Create a deterministic test set
            ofs = 180
            if PLOTS_FOR_S4D:
                ofs = 210
            X_test, Y_test = create_mixed_set(theta, n_expected_events, XH, XA,
                                              sample_theta=False, sample_ntot=False, shuffle=False, offset=ofs) # OK for a=0.5: 190
            true_prediction = Prediction(nH=np.sum(Y_test[:, 0]),
                                         nA=np.sum(Y_test[:, 1]),
                                         nBgnd=0)
            
            # Phistar prediction
            X_test_phistar = phi_pred.select_features(X_test, features)
            nll_curve_phistar = phi_pred.predict_and_plot(X_test_phistar)
            
            # MLE-NN prediction
            X_test_nn, _ = select_features(X_test, features, False)
            nll_curve_mle_nn = mle_nn.predict_and_plot(X_test_nn)
            
            # Plot likelihood ratio curves
            fignll = plt.figure()
            plt.axhline(1.0, color='black', alpha=0.7, linestyle='dashed', linewidth=1.5) 
            plt.plot(nll_curve_phistar[0], 2.0*np.array(nll_curve_phistar[1]), label=r'Fit to $\varphi^*$')
            plt.plot(nll_curve_mle_nn[0], 2.0*np.array(nll_curve_mle_nn[1]), label='Fit to NN output')
            plt.ylim(0, 13)
            plt.xlim(0, 1)
            plt.xlabel(r'$\alpha$')
            plt.ylabel(r'$-2\ln\frac{L(\alpha)}{L(\hat{\alpha})}$')
            plt.legend(loc='upper right')
            plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.17)
            fignll.show()
            fignll.savefig('nll_comparison.pdf')
            
            plt.show()
            

    # Run on lots of sets, compute mean and std.dev.
    if pargs.validation:
        
        all_true_thetas = {}
        all_preds_phi = {}
        all_preds_nn = {}
        all_preds_mle = {}
        all_loglambdas_phi = {}
        all_loglambdas_mle = {}
        
        all_nn_choice = {}
        
        # Loop over thetas
        for theta in thetas_to_test:
            
            # Predictions
            theta_truths = []
            theta_predictions_phi = []
            theta_predictions_nn = []
            theta_predictions_mle = []
            
            # -log Lambda
            loglambda_phi = []
            loglambda_mle = []
            
            # Network choice
            nn_choice = []
            
            starttime = time.time()
            
            # Loop over test sets
            for i in range(n_test_sets):
                
                if i > 0 and i % 10 == 0:
                    thistime = time.time()
                    print 'Theta = %.1f: %.0f%% completed  (%.0f msec per testset)' % (theta, i/float(n_test_sets)*100, 1000*(thistime-starttime)/float(i))
                    
                
                # Randomised test set
                X_test, Y_test = create_mixed_set(theta, n_expected_events, XH, XA,
                                                  sample_theta=True, sample_ntot=False, shuffle=True)
                
                true_theta = float(np.sum(Y_test, axis=0)[1])/len(Y_test)
                theta_truths.append(true_theta)


                # Phistar prediction
                X_test_phistar = phi_pred.select_features(X_test, features)
                prediction_phi, nll_phi, lambda_phi = phi_pred.predict(X_test_phistar, theta_true=true_theta)
                theta_predictions_phi.append(prediction_phi)
                loglambda_phi.append(lambda_phi)
                
                # Naive NN prediction
                X_test_nn, _ = select_features(X_test, features, False)
                prediction_nn = naive_nn.predict(X_test_nn)
                theta_predictions_nn.append(prediction_nn)
                
                # MLE-NN prediction
                prediction_mle, _, lambda_mle, status_mle  = mle_nn.predict(X_test_nn, theta_true=true_theta)
                if status_mle == 0:
                    theta_predictions_mle.append(prediction_mle)
                    loglambda_mle.append(lambda_mle)
            
            all_true_thetas[theta] = theta_truths
            all_preds_phi[theta] = theta_predictions_phi
            all_preds_nn[theta] = theta_predictions_nn
            all_preds_mle[theta] = theta_predictions_mle
            
            all_loglambdas_phi[theta] = loglambda_phi
            all_loglambdas_mle[theta] = loglambda_mle
            
            print 'min / max NN output = %.5f / %.5f' % (mle_nn.min_nn_output, mle_nn.max_nn_output)
        
        
        # Save output
        with open('results_true_thetas.pkl', 'w') as hout:
            pickle.dump(all_true_thetas, hout, -1)
        with open('results_predictions_phistar.pkl', 'w') as hout:
            pickle.dump(all_preds_phi, hout, -1)
        with open('results_predictions_naive_nn.pkl', 'w') as hout:
            pickle.dump(all_preds_nn, hout, -1)
        with open('results_predictions_mle.pkl', 'w') as hout:
            pickle.dump(all_preds_mle, hout, -1)
        
        with open('results_loglambda_phi.pkl', 'w') as hout:
            pickle.dump(all_loglambdas_phi, hout, -1)
        with open('results_loglambda_mle.pkl', 'w') as hout:
            pickle.dump(all_loglambdas_mle, hout, -1)
        
        print 'Done'
        
    
    if pargs.plot:
        
        # Awesome comparison plot
        def comparison_plot(diff, name, diff_smeared=None):
            
            defaultfontsize = rcParams['font.size']
            rcParams['font.size'] = 14

            thetas = diff.keys()
            nplots = len(thetas)
            assert nplots > 1, 'Can\'t index a single plot'
            f, (axes) = plt.subplots(nplots, sharex=True, sharey=True, figsize=(5.5, 8))
            for i in range(nplots):

                xlow, xhigh = -0.3, 1.7
                ylow, yhigh = 0, 950
                nbins = 50
                remove_outliers = False
                if n_expected_events == 500:
                    ylow, yhigh = 0, 1950
                if n_expected_events == 20:
                    ylow, yhigh = 0, 950
                    remove_outliers = True
                
                print 'Theta = %.1f: %d entries' % (thetas[i], len(diff[thetas[i]]))
                thisdiff = np.array(diff[thetas[i]])
                mean = np.mean(thisdiff)
                std = np.std(thisdiff)
                if remove_outliers:
                    thisdiff = thisdiff[np.where((thisdiff >= thetas[i]-1.1) & (thisdiff <= thetas[i]+1.1))]
                if ' nn ' in name:
                    mean = np.mean(thisdiff)
                    std = np.std(thisdiff)
                print ' Result %s: mu = %.5f, sigma = %.5f' % (name, mean, std)

                xvals = np.linspace(xlow, xhigh, 200)
                
                
                # Plot smeared results in the background (i.e. first)
                if diff_smeared is not None:
                    thisdiff_smear = np.array(diff_smeared[thetas[i]])
                    mean_smear = np.mean(thisdiff_smear)
                    std_smear = np.std(thisdiff_smear)
                    axes[i].hist(diff_smeared[thetas[i]], bins=nbins, range=(xlow, xhigh), color='#20A387', alpha=0.4)
                    axes[i].plot(xvals, norm.pdf(xvals, loc=mean_smear, scale=std_smear)*len(thisdiff)/nbins*(xhigh-xlow), color='#20A387', linewidth=1.5, label='With det. res.')


                axes[i].axvspan(xlow, 0, alpha=0.15, color='gray')
                axes[i].axvspan(1, xhigh, alpha=0.15, color='gray')
                axes[i].hist(diff[thetas[i]], bins=nbins, range=(xlow, xhigh), color='#39568C', alpha=0.5)
                textloc = (0.17, 0.85) if mean > 0.8 else (0.75, 0.85)
                axes[i].text(textloc[0]+0.0216, textloc[1], r'$n = 100$', transform=axes[i].transAxes, fontsize=12)
                axes[i].text(textloc[0]+0.0216, textloc[1]-0.10, r'$\alpha = %.1f$' % thetas[i], transform=axes[i].transAxes, fontsize=12)
                axes[i].text(textloc[0], textloc[1]-0.22, r'$\mu_{\alpha} = %.2f$' % mean, transform=axes[i].transAxes, fontsize=12)
                axes[i].text(textloc[0], textloc[1]-0.32, r'$\sigma_{\alpha} = %.2f$' % std, transform=axes[i].transAxes, fontsize=12)
                axes[i].set_xlim([xlow, xhigh])
                axes[i].set_ylim([ylow, yhigh])
                axes[i].yaxis.set_ticks(np.arange(ylow, yhigh, 200))
            
                #xvals = np.linspace(xlow, xhigh, 200)
                axes[i].plot(xvals, norm.pdf(xvals, loc=mean, scale=std)*len(thisdiff)/nbins*(xhigh-xlow), color='#440154', linewidth=1.5, label='No det. res.' )

                if diff_smeared is not None:
                    axes[i].text(textloc[0], textloc[1]-0.445, r'$\mu_{\alpha}^{det} = %.2f$' % mean_smear, transform=axes[i].transAxes, fontsize=12)
                    axes[i].text(textloc[0], textloc[1]-0.545, r'$\sigma_{\alpha}^{det} = %.2f$' % std_smear, transform=axes[i].transAxes, fontsize=12)
                    
                    if i == 0 and 'phistar' in name:
                        axes[i].legend(bbox_to_anchor=(0.2, 1.0), loc='upper left', fontsize=12)


            f.subplots_adjust(hspace=0, top=0.99, right=0.99, left=0.17)
            f.canvas.set_window_title(name)
            plt.xlabel(r'$\hat{\alpha}$')
            plt.ylabel('Test sets / %.2f' % ((xhigh-xlow)/nbins))
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            #plt.tight_layout()
            f.show()

            f.savefig(name.replace(' ', '_')+'.pdf') 
            rcParams['font.size'] = defaultfontsize
        

        # Different comparison plot
        def alternate_comparison_plot(diff, name):
            fig = plt.figure()
            for theta, vals in diff.iteritems():
                mean = np.mean(vals)
                std = np.std(vals)
                
                plt.hist(vals, bins=20, range=(0,1), alpha=0.5, label=r'$\alpha$ (test) = %.1f' % theta)
                plt.axvline(theta, color='gray', linewidth=0.5)
                plt.xlim([0,1])
                plt.xlabel(r'Predicted $\alpha$')
                plt.ylabel('Events')
                
            fig.canvas.set_window_title(name)
            fig.show()
        

        # Comparison of two methods in one plot
        def s4d_plot(diffphi, diffmle, theta):

            rcParams['xtick.major.pad'] = 12
            rcParams['ytick.major.pad'] = 12
            
            xlow, xhigh = -0.3, 1.7
            ylow, yhigh = 0, 950
            nbins = 50
            opacity = '10'

            fig = plt.figure()
            ax = plt.gca()
            plt.axvspan(xlow, 0, alpha=0.15, color='gray')
            plt.axvspan(1, xhigh, alpha=0.15, color='gray')

            plt.hist(diffphi, bins=nbins, range=(xlow, xhigh), histtype='step',
                color='#E64A19', fill=True, fc='%s%s' % ('#E64A19', opacity),
                linewidth=1, label=r'$\varphi^*$')
            plt.hist(diffmle, bins=nbins, range=(xlow, xhigh), histtype='step',
                color='#2359B0', fill=True, fc='%s%s' % ('#2359B0', opacity),
                linewidth=1, label=r'$\mathrm{MLE}$')
            
            xvals = np.linspace(xlow, xhigh, 200)
            mu, sigma = np.mean(diffphi), np.std(diffphi)
            plt.plot(xvals, norm.pdf(xvals, loc=mu, scale=sigma)*len(diffphi)/nbins*(xhigh-xlow), color='#E64A19', linewidth=1)
            
            mu, sigma = np.mean(diffmle), np.std(diffmle)
            plt.plot(xvals, norm.pdf(xvals, loc=mu, scale=sigma)*len(diffmle)/nbins*(xhigh-xlow), color='#2359B0', linewidth=1)

            plt.text(0.78, 0.74, r'$n = 100$', transform=ax.transAxes)
            plt.text(0.78, 0.67, r'$\alpha = %.1f$' % theta, transform=ax.transAxes)

            plt.xticks(np.arange(xlow, xhigh, 0.4))
            plt.xlim([xlow, xhigh])
            plt.legend(loc='upper right')
            #fig.subplots_adjust(hspace=0, top=0.99, right=0.99, left=0.17)
            fig.canvas.set_window_title('S4D results, theta = %.1f' % theta)
            plt.xlabel(r'$\hat{\alpha}$')
            plt.ylabel('Test sets')
            plt.tight_layout()
            fig.show()

            fig.savefig('s4d_results_theta_%s.pdf' % str(theta).replace('.', 'p'))


        # Load data
        with open(pargs.result_path + '/results_predictions_phistar.pkl', 'rb') as hin:
            all_preds_phi = pickle.load(hin)
        with open(pargs.result_path + '/results_predictions_naive_nn.pkl', 'rb') as hin:
            all_preds_nn = pickle.load(hin)
        with open(pargs.result_path + '/results_predictions_mle.pkl', 'rb') as hin:
            all_preds_mle = pickle.load(hin)
        with open(pargs.result_path + '/results_predictions_phistar_smeared.pkl', 'rb') as hin:
            all_preds_phi_smeared = pickle.load(hin)
        with open(pargs.result_path + '/results_predictions_mle_smeared.pkl', 'rb') as hin:
            all_preds_mle_smeared = pickle.load(hin)
        
        comparison_plot(all_preds_phi, 'phistar unbinned', all_preds_phi_smeared)
        comparison_plot(all_preds_mle, 'single nn fit unbinned', all_preds_mle_smeared)


        
        # Kyle fig 2b
        def loglambda_plot(values, name):
            thetas = values.keys()
            nplots = len(thetas)

            rcParams['xtick.major.pad'] = 12
            rcParams['ytick.major.pad'] = 12
            
            # For now plot only theta = 0.7
            fig = plt.figure()
            plt.hist(np.array(values[0.7])*2.0, normed=True, bins=75, range=(0, 6.5), color='#20A387')

            xvals = np.linspace(0, 6.5, 300)
            plt.plot(xvals, chi2.pdf(xvals, df=1.0), color='#440154', alpha=0.8, label=r'$\chi^{2}$, n.d.f = 1')
            
            plt.xlabel(r'$-2 \ln \frac{L(\alpha = 0.7)}{L(\hat{\alpha})}$')
            plt.ylabel('Probability density / %.2f' % (6.5/75))
            fig.canvas.set_window_title(name)
            plt.legend(loc='upper right')
            plt.xlim([0, 6.5])
            plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.17)
            fig.show()
            fig.savefig('loglambda_chisquare_mle_nn.pdf')
        
        with open(pargs.result_path + '/results_loglambda_phi.pkl', 'rb') as hin:
            all_loglambdas_phi = pickle.load(hin)
        with open(pargs.result_path + '/results_loglambda_mle.pkl', 'rb') as hin:
            all_loglambdas_mle = pickle.load(hin)
        
        loglambda_plot(all_loglambdas_mle, 'mle')
        

        # Naive network bias compared to phistar bias
        with open(pargs.result_path + '/results_true_thetas.pkl', 'rb') as hin:
            all_true_thetas = pickle.load(hin)

        xlow, xhigh = -0.3, 1.9
        
        # A: Naive NN results
        colors = ['#440154', '#39568C', '#20A387']
        lstyles = ['solid', 'solid', 'solid']
        opacity = '00'
        
        if PLOTS_FOR_S4D:
            colors = ['#E64A19', '#FFFFFF', '#2359B0']
            opacity = 10


        # B
        fig = plt.figure()
        plt.axvspan(xlow, 0, alpha=0.15, color='gray')
        plt.axvspan(1, xhigh, alpha=0.15, color='gray')
        
        for theta, col, sty in zip(all_preds_nn.keys(), colors, lstyles):    
            
            if PLOTS_FOR_S4D and theta == 0.7:
                continue

            diff_theta = np.array(all_preds_nn[theta])
            plt.hist(diff_theta, bins=200, range=(xlow, xhigh), color=col, 
                     histtype='step', fill=True, fc='%s%s' % (col, opacity),
                     linestyle=sty, linewidth=1.0,
                     label=r'$\alpha$ = %.1f' % theta)
            
            plt.axvline(theta, color=col, linewidth=1.0, linestyle='dashed')

        plt.xticks(np.arange(xlow, xhigh, 0.4))
        plt.xlim([xlow, xhigh])
        plt.legend(loc='upper right')
        fig.canvas.set_window_title('NN bias')
        plt.xlabel(r'$\hat{\alpha}$')
        plt.ylabel('Test sets')
        plt.tight_layout()
        
        
        plt.show()
            
