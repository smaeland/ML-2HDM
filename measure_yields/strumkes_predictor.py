# pylint: disable=C0303

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import sqrt
from array import array
from copy import deepcopy
import pickle
import time
from keras.models import load_model
from collections import namedtuple
from sklearn import utils
from keras.utils import to_categorical
from multiprocessing import Process
import subprocess

#from measure_yields.utilities import TemplatePredictor, Prediction, Histogram
from neuralnet.utilities import select_features
from neuralnet.utilities import get_dataset_from_path
from measure_yields.basepredictors import BasePredictor, TemplatePredictor, \
    Histogram, Prediction, stdout_redirected_to
from measure_yields.utilities import data_already_scaled, get_adapted_bin_edges

from ROOT import RooRealVar, RooDataHist, RooDataSet, RooArgSet, TCanvas, TFile
from ROOT import RooKeysPdf, RooAddPdf, RooArgList, RooWorkspace
from ROOT.RooFit import Binning, Save, PrintLevel



#plt.style.use('../../plot/paper.mplstyle')
from matplotlib import rcParams





class MaxLikelihoodNnPredictorBinned(TemplatePredictor):
    """
    Predict based on a binned maximum likelihood fit to the output of a single
    neural network. Uses HistFactory for the template fit implementation.
    
    Arguments:
        title: String
        nbins: Number of bins in histograms
        saved_model: Pre-trained Keras model (.h5 file)
        saved_scaler: Pickled StandardScaler (.pkl file)
        samplelist: List of samples to create templates for, e.g. ['A', 'H', 'bgnd']
    """
    
    def __init__(self, title, nbins, saved_model, saved_scaler, samplelist):
        
        super(MaxLikelihoodNnPredictorBinned, self).__init__(title)
        
        # Load the NN model
        self.model = load_model(saved_model)
        
        # Load the data scaler
        assert os.path.exists(saved_scaler)
        with open(saved_scaler) as hfile:
            self.scaler = pickle.load(hfile)
        
        # Histogram settings
        self.histrange = [0.0, 1.0]

        self.samples = samplelist
        self.templates = {'H' : None, 'A' : None, 'bgnd' : None}
        
        # Get templates
        for sample in self.templates:
            filename = '%s/histo_%s_%s.pkl' % (self.outdir, self.title, sample)
            if os.path.exists(filename):
                print 'Found histogram file:', filename
                with open(filename) as hfile:
                    self.templates[sample] = pickle.load(hfile)
                    self.binedges = self.templates[sample].bin_edges
            else:
                print 'Could not locate', filename
        
        # Colors
        self.colors = {
            "H": "#440154",
            "A": "#39568C",
            "data": "#FFFFFF",
            "model": "#20A387"
        }

   

    def create_template(self, x, sample, title, istrain=True, save=False):
        """
        Create a histogram of network output for a given sample type
        
        Arguments:
            x: Data, un-scaled
            sample: Either 'H', 'A' or 'bgnd'
            title: String
            istrain: Templates for training data are scaled to unit integral
            save: Store the template
        """
        
        assert sample in ['H', 'A', 'bgnd'], 'Sample %s not recognised' % sample
        assert not data_already_scaled(x), 'Data already scaled?'
        
        # Preprocess the inputs 
        sx = self.scaler.transform(x)
        
        # Make predictions 
        preds = self.model.predict(sx)   # (?, n_class) matrix, floats

        # Choose prediction column
        column_to_predict = 0   # -> H
        if sample == 'A':
            column_to_predict = 1   # -> A
        elif sample == 'bgnd':
            column_to_predict = 2
            
        # Fill the histogram   
        hist = Histogram(preds[:, column_to_predict], self.binedges, self.title, title)
        
        # Scale to unit integral
        if istrain:
            hist.th1.Scale(1.0/hist.th1.Integral())
        
        if save:
            with open('%s/histo_%s_%s.pkl' % (self.outdir, self.title, title), 'w') as hout:
                pickle.dump(hist, hout, -1)
        
        return hist




    def create_train_templates(self, paths_to_data, adaptive_binning=False):
        """
        Create templates from training files. Backgrounds merged into one.
        
        Arguments:
            paths_to_data: List of paths to directories containing training
                data. Typically only one, but can also train on multiple
                datasets from different models (i.e. different masses)
            adaptive_binning: Use narrower bin width in dense regions
        """
        
        if not isinstance(paths_to_data, list):
            paths_to_data = [paths_to_data]
        
        files_H = []
        files_A = []
        for path in paths_to_data:
            # These are merged, so only one of each
            if not path.endswith('/'): path += '/'
            files_H.append(path+'model_H_merged.h5')
            files_A.append(path+'model_A_merged.h5')

        # Find background samples, put into one template
        files_bgnd = []
        for path in paths_to_data:
            for sample in self.samples:
                if sample not in ['H', 'A']:
                    files_bgnd.append(path+'/model_%s_merged.h5' % sample)
        
        # Make the templates
        dataH_X, _, feats = self.read_array_from_files(files_H)
        dataH_X, _ = select_features(dataH_X, feats, include_mass_variables=False)
        
        dataA_X, _, feats = self.read_array_from_files(files_A)
        dataA_X, _ = select_features(dataA_X, feats, include_mass_variables=False)
        
        dataH_X = self.scaler.transform(dataH_X)
        dataA_X = self.scaler.transform(dataA_X)
        
        print 'WARNING: LIMITING TEMPLATE EVENTS'
        dataH_X = dataH_X[:5000]
        dataA_X = dataA_X[:5000]

        # Need all data merged in order to do adaptive binning
        data_all_X = np.vstack((dataH_X, dataA_X))
        
        if files_bgnd:
            dataBgnd_X, _, feats = self.read_array_from_files(files_bgnd)
            dataBgnd_X, _ = select_features(dataBgnd_X, feats, include_mass_variables=False)
            dataBgnd_X = self.scaler.transform(dataBgnd_X)
            data_all_X = np.vstack((data_all_X, dataBgnd_X))
        
        # Get reasonable histogram binning
        if adaptive_binning:
            temp_preds = self.model.predict(data_all_X)[:, 1]
            self.binedges = get_adapted_bin_edges(temp_preds, fullrange=(0,1))
        else:
            self.binedges = np.linspace(0, 1, self.nbins+1)
        
        print ' DBG: bin edges:', self.binedges
        
        # Now create the templates
        self.templates['H'] = self.create_template(dataH_X, 'H', 'H', istrain=True, save=True)
        self.templates['A'] = self.create_template(dataA_X, 'A', 'A', istrain=True, save=True)
        if files_bgnd:
            self.templates['bgnd'] = self.create_template(dataBgnd_X, 'bgnd', 'bgnd', istrain=True, save=True)
        
        # Plot templates
        """
        fig = plt.figure()
        plt.hist(self.model.predict(dataH_X)[:,1].flatten(), bins=self.binedges, normed=True, histtype='step', label='H')
        plt.hist(self.model.predict(dataA_X)[:,1].flatten(), bins=self.binedges, normed=True, histtype='step', label='A')
        plt.legend(loc='best')
        fig.show()
        plt.show()
        """
        print 'Created train templates for', self.title







class MaxLikelihoodNnPredictorUnbinned(BasePredictor):
    """
    Predict based on an unbinned maximum likelihood fit to the output of a
    single neural network. Uses kernel density estimation (KDE) to create pdfs
    for network output distributions for H and A
    
    Arguments:
        title: String
        saved_model: Pre-trained Keras model (.h5 file)
        saved_scaler: Pickled StandardScaler (.pkl file)
        samplelist: List of samples to create templates for, e.g. ['A', 'H', 'bgnd']
    """
    
    def __init__(self, title, saved_model, saved_scaler, sample_list):
        
        super(MaxLikelihoodNnPredictorUnbinned, self).__init__(title)
        
        # Load the NN model
        self.model = load_model(saved_model)
        
        # Load the data scaler
        assert os.path.exists(saved_scaler)
        with open(saved_scaler) as hfile:
            self.scaler = pickle.load(hfile)
        
        # RooFit variables
        self.roopred = None
        self.theta_min, self.theta_max = -10, 10

        self.samples = sample_list
        self.pdfs = {'H' : None, 'A' : None, 'bgnd' : None}
        
        # Load existing pdfs
        filename = '%s/pdfs_%s.root' % (self.outdir, self.title)
        if os.path.exists(filename):
            print 'Found pdf file:', filename
            fin = TFile(filename)
            
            ws = fin.Get('ws')
            self.roopred = ws.var('roopred')
            self.pdfs['H'] = ws.pdf('keysH')
            self.pdfs['A'] = ws.pdf('keysA')
        
        # Tweak ROOT
        from ROOT.Math import MinimizerOptions
        MinimizerOptions.SetDefaultMinimizer("Minuit2");
        MinimizerOptions.SetDefaultStrategy(2);
        
        # Debug
        self.max_nn_output = 0.0
        self.min_nn_output = 1.0

        # Colors
        self.colors = {
            "H": "#440154",
            "A": "#39568C",
            "data": "#FFFFFF",
            "model": "#20A387"
        }
        
    def create_pdfs(self, paths_to_data, save=True):
        """
        Create KDE pdfs for network output distribution
        """
        
        if not isinstance(paths_to_data, list):
            paths_to_data = [paths_to_data]
        
        files_H = []
        files_A = []
        for path in paths_to_data:
            # These are merged, so only one of each
            if not path.endswith('/'): path += '/'
            files_H.append(path+'model_H_merged.h5')
            files_A.append(path+'model_A_merged.h5')

        # Find background samples
        files_bgnd = []
        for path in paths_to_data:
            for sample in self.samples:
                if sample not in ['H', 'A']:
                    files_bgnd.append(path+'/model_%s_merged.h5' % sample)
        
        # Read in the data
        dataH_X, _, feats = self.read_array_from_files(files_H)
        dataH_X, _ = select_features(dataH_X, feats, include_mass_variables=False)
        dataH_X = self.scaler.transform(dataH_X)
        
        rho = 0.8
        #ntemplateevents = 1000000
        ntemplateevents = 500000
        #ntemplateevents = 50000
        print 'Limiting events for pdf creation to %d' % ntemplateevents
        # 700k -> 4h 42m
        # 500k -> 2h 30m
        # 200k -> 0h 27m
        # 100k -> 0h  7m
        #  50k -> 0h  2m
        dataH_X = dataH_X[:ntemplateevents]
        
        # Predict, fill a RooDataHist
        starttime = time.time()
        predsH = self.model.predict(dataH_X)[:,1]
        
        self.roopred = RooRealVar('roopred', 'roopred', 0, 1)
        roopreddataH = RooDataSet('roopreddataH', 'roopreddataH', RooArgSet(self.roopred))
        for pred in predsH:
            self.roopred.setVal(pred)
            roopreddataH.add(RooArgSet(self.roopred))
        
        # Create the KDE pdfs
        def createKDE(self, data, htype):
            starttime = time.time()
            keys = RooKeysPdf('keys%s' % htype, 'keys%s' % htype, self.roopred, data)
            self.pdfs[htype] = keys
            print 'Creating KDE pdf for %s took %s' % (htype, time.strftime("%H:%M:%S", time.gmtime(time.time()-starttime)))

        from ROOT.RooKeysPdf import NoMirror
        keysH = RooKeysPdf('keysH', 'keysH', self.roopred, roopreddataH, NoMirror, rho)
        self.pdfs['H'] = keysH
        
        # Do the same for A
        dataA_X, _, feats = self.read_array_from_files(files_A)
        dataA_X, _ = select_features(dataA_X, feats, include_mass_variables=False)
        dataA_X = self.scaler.transform(dataA_X)
        dataA_X = dataA_X[:ntemplateevents]
        starttime = time.time()
        predsA = self.model.predict(dataA_X)[:,1]
        
        roopreddataA = RooDataSet('roopreddataA', 'roopreddataA', RooArgSet(self.roopred))
        for pred in predsA:
            self.roopred.setVal(pred)
            roopreddataA.add(RooArgSet(self.roopred))
        
        keysA = RooKeysPdf('keysA', 'keysA', self.roopred, roopreddataA, NoMirror, rho)
        self.pdfs['A'] = keysA

        if save:
            ws = RooWorkspace('ws', 'ws')
            getattr(ws, 'import')(self.roopred)
            getattr(ws, 'import')(self.pdfs['H'])
            getattr(ws, 'import')(self.pdfs['A'])
            ws.writeToFile('%s/pdfs_%s.root' % (self.outdir, self.title))
            



    def predict(self, x, theta_true):
        """
        Run an unbinned ML fit to make predictions
        """
        
        # Create RooDataSet
        xs = self.scaler.transform(x)
        preds = self.model.predict(xs)[:, 1]

        min_nn_output_local, max_nn_output_local = np.min(preds), np.max(preds)
        if min_nn_output_local < self.min_nn_output:
            self.min_nn_output = min_nn_output_local
        if max_nn_output_local > self.max_nn_output:
            self.max_nn_output = max_nn_output_local
        
        roodata = RooDataSet('data', 'data', RooArgSet(self.roopred))
        for pred in preds:
            self.roopred.setVal(pred)
            roodata.add(RooArgSet(self.roopred))

        
        # Fit
        theta = RooRealVar('theta', 'theta', 0.5, self.theta_min, self.theta_max)
        
        model = RooAddPdf('model', 'model',
                          RooArgList(self.pdfs['A'], self.pdfs['H']),
                          RooArgList(theta))
        
        
        with stdout_redirected_to('%s/minuit_output.log' % self.outdir):
            res = model.fitTo(roodata, Save(True))
            nll = res.minNll()

        fitstatus = res.status()
        fitstatus |= (not subprocess.call(['grep', 'p.d.f value is less than zero', 'output_MLE_unbinned/minuit_output.log']))

        fitted_theta = theta.getValV()
        
        # Get Lambda(theta_true | theta_best)
        logl = model.createNLL(roodata)
        
        theta.setVal(theta_true)
        nll_theta_true = logl.getValV()
        nll_ratio = nll_theta_true - nll
  
        return fitted_theta, nll, nll_ratio, fitstatus



    def predict_and_plot(self, x):
        """
        Do the same as predict(), but add plots
        
        Return -logL ratio, for external plotting
        """

        rcParams['xtick.major.pad'] = 12
        rcParams['ytick.major.pad'] = 12
        
        
        xs = self.scaler.transform(x)
        preds = self.model.predict(xs)[:, 1]
        
        roodata = RooDataSet('data', 'data', RooArgSet(self.roopred))
        for pred in preds:
            self.roopred.setVal(pred)
            roodata.add(RooArgSet(self.roopred))

        theta = RooRealVar('theta', 'theta', 0.5, self.theta_min, self.theta_max)
        model = RooAddPdf('model', 'model',
                          RooArgList(self.pdfs['A'], self.pdfs['H']),
                          RooArgList(theta))
        
        
        #with stdout_redirected_to():
        print '\n\nNEURAL NETWORK FIT'
        res = model.fitTo(roodata, PrintLevel(10))
        
        fitted_theta = theta.getValV()
            
        # Histogram binning for data points
        nbins = 14
        
        xvals = np.linspace(0, 1, 300)
        yvals_H = []
        yvals_A = []
        
        # Get points for pdf curves
        for xval in xvals:
            self.roopred.setVal(xval)
            yvals_H.append(self.pdfs['H'].getValV(RooArgSet(self.roopred)))
            yvals_A.append(self.pdfs['A'].getValV(RooArgSet(self.roopred)))
        
        yvals_H = np.array(yvals_H)
        yvals_A = np.array(yvals_A)
        
        # Plot pdfs by themselves
        fig = plt.figure()
        plt.plot(xvals, yvals_H, color=self.colors["H"], label=r'$p_{H}(y)$')
        plt.plot(xvals, yvals_A, color=self.colors["A"], label=r'$p_{A}(y)$')
        plt.fill_between(xvals, 0, yvals_H, color=self.colors["H"], alpha=0.2)
        plt.fill_between(xvals, 0, yvals_A, color=self.colors["A"], alpha=0.2)
        
        plt.xlim([0, 1])
        plt.ylim([0, 11])
        plt.xlabel(r'Network output ($y$)')
        plt.ylabel('Probability density')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.show()
        fig.savefig('mle_nn_pdfs.pdf')
        
        # Scale to event yield
        yvals_H *= roodata.numEntries()*(1-fitted_theta)/float(nbins)
        yvals_A *= roodata.numEntries()*(fitted_theta)/float(nbins)
        yvals_sum = yvals_H + yvals_A
        
        
        # Make plot of fit to data
        fig, ax = plt.subplots(1)
        histentries, binedges = np.histogram(preds, bins=nbins, range=(0, 1))
        bincenters = (binedges[:-1] + binedges[1:])/2.0
        yerr = np.sqrt(histentries)
        plt.errorbar(bincenters, histentries, xerr=np.diff(binedges)*0.5, yerr=yerr, linestyle='None', ecolor='black', label='Data')
        plt.plot(xvals, yvals_H, color=self.colors["H"], label=r'$p_{H}(y)$')
        plt.plot(xvals, yvals_A, color=self.colors["A"], label=r'$p_{A}(y)$')
        plt.plot(xvals, yvals_sum, color=self.colors["model"], label=r'$p(y \,|\, \alpha = %.2f)$' % fitted_theta)
        plt.fill_between(xvals, 0, yvals_H, color=self.colors["H"], alpha=0.2)
        plt.fill_between(xvals, 0, yvals_A, color=self.colors["A"], alpha=0.2)
        
        # Set correct legend order
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[3], handles[2], handles[0], handles[1]]
        labels = [labels[3], labels[2], labels[0], labels[1]]
        ax.legend(handles, labels, loc='upper right')
        
        plt.xlabel(r'Network output ($y$)')
        plt.ylabel('Events / %.2f' % (1.0/nbins))
        
        axes = plt.gca()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, max(histentries)*2.3])
        plt.tight_layout()
        fig.show()
        fig.savefig('mle_nn_fit.pdf')
        
        
        # Create likelihood curve
        logl = model.createNLL(roodata)
        
        xvals = np.linspace(0, 1, 200)
        yvals = []
        ymin = 999.
        for xval in xvals:
            theta.setVal(xval)
            yvals.append(logl.getValV())
            if yvals[-1] < ymin:
                ymin = yvals[-1]
        
        yvals = np.array(yvals)
        yvals -= ymin
        
        # Return points for the NLL curve
        return xvals, yvals
        
        





class MostLikelyNnPredictorBinned(BasePredictor):
    """
    Do maximum likelihood fits to the output of several networks, predict based
    on the result of the best fit
    
    Arguments:
        title: String
        nbins: Number of bins in histograms
        saved_models: List of pre-trained Keras models (.h5 files)
        saved_scaler: Pickled StandardScaler (.pkl file)
        sample_list: List of samples to consider, e.g. ['H', 'A', 'bgnd']
        mode: Choose NLL-weighted average of all networks ('weighted'), or pick
            the one with lowest NLL value ('pick-best')
    """
    
    
    def __init__(self, title, nbins, saved_models, saved_scaler, sample_list, mode='pick-best'):
        
        super(MostLikelyNnPredictorBinned, self).__init__(title)
        
        assert mode in ['weighted', 'pick-best'], 'Invalid mode: %s' % mode
        self.mode = mode
        
        self.predictors = []
        for modelfile in saved_models:
            thistitle = modelfile.split('/')[-1].replace('.h5', '')
            mlpred = MaxLikelihoodNnPredictor(
                title=thistitle,
                nbins=nbins, saved_model=modelfile, saved_scaler=saved_scaler,
                samplelist=sample_list)
            self.predictors.append(mlpred)


    def create_train_templates(self, paths):
        """ Create train templates for all models """
        for pred in self.predictors:
            pred.create_train_templates(paths)



    def predict(self, x, plot=False):
        
        allpreds = []
        allnlls = []
        
        for predictor in self.predictors:
            
            preds, nll = predictor.predict(x, plot=plot)
            allpreds.append(preds)
            allnlls.append(nll)
        
        if self.mode == 'weighted':
            nAs = []; nHs = []; nBgnds = []
            for p in allpreds:
                nAs.append(p.nA)
                nHs.append(p.nH)
                nBgnds.append(p.nBgnd)
            
            result = Prediction(nH=np.average(nHs, weights=allnlls),
                                nA=np.average(nAs, weights=allnlls),
                                nBgnd=np.average(nBgnds, weights=allnlls))
        
        elif self.mode == 'pick-best':
            
            result = preds[allnlls.index(min(allnlls))]
        
        print ' DBG: nlls are:', allnlls
        
        return result




class MostLikelyNnPredictorUnbinned(BasePredictor):
    """
    Do maximum likelihood fits to the output of several networks, predict based
    on the result of the best fit
    
    Arguments:
        title: String
        saved_models: List of pre-trained Keras models (.h5 files)
        saved_scaler: Pickled StandardScaler (.pkl file)
        sample_list: List of samples to consider, e.g. ['H', 'A', 'bgnd']
        mode: Choose NLL-weighted average of all networks ('weighted'), or pick
            the one with lowest NLL value ('pick-best')
    """
    
    
    def __init__(self, title, saved_models, saved_scaler, sample_list, mode='pick-best'):
        
        super(MostLikelyNnPredictorUnbinned, self).__init__(title)
        
        assert mode in ['weighted', 'pick-best'], 'Invalid mode: %s' % mode
        self.mode = mode
        
        self.predictors = []
        for modelfile in saved_models:
            thistitle = modelfile.split('/')[-1].replace('.h5', '')
            mlpred = MaxLikelihoodNnPredictorUnbinned(
                title=thistitle,
                saved_model=modelfile, saved_scaler=saved_scaler,
                sample_list=sample_list)
            self.predictors.append(mlpred)


    def create_pdfs(self, paths):
        """ Create pdfs for all models """
        for pred in self.predictors:
            pred.create_pdfs(paths)



    def predict(self, x, plot=False):
        
        allpreds = []
        allnlls = []
        
        for predictor in self.predictors:
            
            preds, nll, _ = predictor.predict(x, theta_true=0) # TODO
            allpreds.append(preds)
            allnlls.append(nll)
        
        if self.mode == 'weighted':
            result = np.average(allpreds, allnlls)
            
        elif self.mode == 'pick-best':
            
            result = allpreds[allnlls.index(min(allnlls))]

        
        return result, allnlls.index(min(allnlls))




