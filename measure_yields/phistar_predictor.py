# pylint: disable=C0303

"""
Onedimensional fit to phi* to extract A/H yields. Makes a plot of the phi* 
templates and the final fit model. 

"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import pi, sqrt
from array import array
from copy import deepcopy
import pickle
import time

from measure_yields.basepredictors import BasePredictor, TemplatePredictor, Prediction, Histogram, stdout_redirected_to

from ROOT import RooRealVar, RooDataSet, RooArgSet, TCanvas, TFile
from ROOT import RooGenericPdf, RooAddPdf, RooArgList, RooWorkspace
from ROOT.RooFit import Binning, Save

from matplotlib import rcParams






class PhistarPredictorBinned(TemplatePredictor):
    """
    Binned maximum likelihood fit to the phistar observable. For documentation
    of the fit itself, see TemplatePredictor
    
    Arguments:
        nbins: Number of bins for histograms
        samplelist: List of samples to create templates for, e.g. ['A', 'H',
            'bgnd']
    
    """
    
    def __init__(self, title, nbins, samplelist):
        
        super(PhistarPredictorBinned, self).__init__(title)

        self.samples = samplelist
        
        self.histrange = [0, 2*pi] 
        self.nbins = nbins
        self.binedges = np.linspace(0, 2*pi, self.nbins+1)
        self.bincenters = self.binedges[:-1] + (pi/self.nbins)

        self.templates = {
            'H' : None,
            'A' : None,
            'bgnd' : None
            }
        
        # Get templates
        for sample in self.templates:
            filename = '%s/histo_%s_%s.pkl' % (self.outdir, self.title, sample)
            if os.path.exists(filename):
                print 'Found histogram file:', filename
                with open(filename) as hfile:
                    self.templates[sample] = pickle.load(hfile)
            else:
                print 'Could not locate', filename
        
        # Colors
        self.colors = {
            "H": "#440154",
            "A": "#39568C",
            "data": "#FFFFFF",
            "model": "#20A387"
        }
   


    def create_template(self, x, title, istrain=True, save=False):
        """
        Create a histogram
        
        Arguments:
            x: Numpy array with phistar, shape (?, 1)
            title: string
            istrain: If input is training data, the template is scaled to
                unit integral
            save: Store the histogram
        """ 
        
        assert x.shape[1] == 1, 'Input array x has shape %s' % x.shape.__str__()
        
        hist = Histogram(x, self.binedges, self.title, title)
        
        # Scale to unit integral
        if istrain:
            hist.th1.Scale(1.0/hist.th1.Integral())
        
        if save:
            with open('%s/histo_%s_%s.pkl' % (self.outdir, self.title, title), 'w') as hout:
                pickle.dump(hist, hout, -1)
        
        return hist
        
   

    def select_features(self, x, features):
        """ 
        Pick out the column with phistar
        """
        
        col = np.where(features == 'phistar')[0][0]
        x = x[:, [col]]
        
        return x





class PhistarPredictorUnbinned(BasePredictor):
    """
    Unbinned maximum likelihood fit to the phistar observable.
    
    Arguments:
        title: A string
        samplelist: List of samples to create templates for, e.g. ['A', 'H',
            'bgnd']
    
    """
    
    def __init__(self, title, samplelist):
        
        super(PhistarPredictorUnbinned, self).__init__(title)
        
        self.samples = samplelist
        
        self.pdfs = {'H' : None, 'A' : None, 'bgnd' : None}
        
        # RooFit variables
        self.phistar = None
        self.ampl = None
        self.offset = None
        self.theta_min, self.theta_max = -10, 10
        
        
        # Load saved pdfs
        filename = '%s/pdfs_%s.root' % (self.outdir, self.title)
        if os.path.exists(filename):
            print 'Found pdf file:', filename
            fin = TFile(filename)
            ws = fin.Get('ws')
            self.phistar = ws.var('phistar')
            self.ampl = ws.var('ampl')
            self.offset = ws.var('offset')
            self.pdfs['H'] = ws.pdf('scalar_pdf')
            self.pdfs['A'] = ws.pdf('pseudoscalar_pdf')
            
        # Colors
        self.colors = {
            "H": "#440154",
            "A": "#39568C",
            "data": "#FFFFFF",
            "model": "#20A387"
        }

    def select_features(self, x, features):
        """ 
        Pick out the column with phistar
        """
        
        col = np.where(features == 'phistar')[0][0]
        x = x[:, [col]]
        
        return x
    
    
    def create_pdfs(self, paths_to_data, save=True):
        """ 
        Fit amplitude and offset of sinusoidal pdfs
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

        # TODO: Background samples: uniform pdf
        
        
        self.phistar = RooRealVar('phistar', 'phistar', 0, 2*pi)
        self.ampl = RooRealVar('ampl', 'ampl', 1, 0, 10)        # Must remain in scope
        self.offset = RooRealVar('offset', 'offset', 1, 0, 10)  # - " -
        
        # Read in the data
        # Need only to fit pdf parameters for H, since the pdf for A is
        # equal but with a 180 deg phase shift.
        starttime = time.time()
        dataH_X, _, feats = self.read_array_from_files(files_H)
        dataH_X = self.select_features(dataH_X, feats)
        
        print 'Limiting events for pdf creation to 1e6'
        dataH_X = dataH_X[:1000000]
        
        templatedataH = RooDataSet('templatedataH', 'templatedataH', RooArgSet(self.phistar))
        for phi in dataH_X:
            self.phistar.setVal(phi)
            templatedataH.add(RooArgSet(self.phistar))
                
        # Reset parameters
        self.ampl = RooRealVar('ampl', 'ampl', 1, 0, 10)
        self.offset = RooRealVar('offset', 'offset', 1, 0, 10)
        
        # Pdf for scalar decay
        scalar_pdf = RooGenericPdf('scalar_pdf', 'scalar_pdf', 'ampl*sin(phistar-1.570796) + offset', RooArgList(self.ampl, self.phistar, self.offset))
        scalar_pdf.fitTo(templatedataH)
        
        self.ampl.setConstant()
        self.offset.setConstant()
        
        # Fitting only the scalar pdf is enough to get the pseudoscalar parameters too
        pseudoscalar_pdf = RooGenericPdf('pseudoscalar_pdf', 'pseudoscalar_pdf', 'ampl*sin(phistar+1.570796) + offset', RooArgList(self.ampl, self.phistar, self.offset))
        
        self.pdfs['H'] = scalar_pdf
        self.pdfs['A'] = pseudoscalar_pdf
        
        print 'Creating phistar pdf took %s' % time.strftime("%H:%M:%S", time.gmtime(time.time()-starttime))
        
        if save:
            ws = RooWorkspace('ws', 'ws')
            getattr(ws, 'import')(self.phistar)
            getattr(ws, 'import')(self.ampl)
            getattr(ws, 'import')(self.offset)
            getattr(ws, 'import')(self.pdfs['H'])
            getattr(ws, 'import')(self.pdfs['A'])
            ws.writeToFile('%s/pdfs_%s.root' % (self.outdir, self.title))
        
        
        
    def predict(self, x, theta_true):
        """
        Run the unbinned ML fit
        """
        
        # Data
        roodata = RooDataSet('data', 'data', RooArgSet(self.phistar))
        for xval in x:
            self.phistar.setVal(xval)
            roodata.add(RooArgSet(self.phistar))
        
        theta = RooRealVar('theta', 'theta', 0.5, self.theta_min, self.theta_max)
        
        # The combined pdf
        model = RooAddPdf('model', 'model',
                          RooArgList(self.pdfs['A'], self.pdfs['H']),
                          RooArgList(theta))
        
        with stdout_redirected_to('%s/minuit_output.log' % self.outdir):
            res = model.fitTo(roodata, Save(True))
            nll = res.minNll()
        
        fitted_theta = theta.getValV()
        
        # Get Lambda(theta_true | theta_best)
        with stdout_redirected_to():
            logl = model.createNLL(roodata)
        
        theta.setVal(theta_true)
        nll_theta_true = logl.getValV()
        nll_ratio = nll_theta_true - nll
        
        return fitted_theta, nll, nll_ratio
        
        
    

    def predict_and_plot(self, x): 
        """
        Do the same as predict(), but add plots
        
        Return -logL ratio, for external plotting
        """

        rcParams['xtick.major.pad'] = 12
        rcParams['ytick.major.pad'] = 12
        
        roodata = RooDataSet('data', 'data', RooArgSet(self.phistar))
        for xval in x:
            self.phistar.setVal(xval)
            roodata.add(RooArgSet(self.phistar))
        theta = RooRealVar('theta', 'theta', self.theta_min, self.theta_max)
        
        model = RooAddPdf('model', 'model',
                          RooArgList(self.pdfs['A'], self.pdfs['H']),
                          RooArgList(theta))
        
        #with stdout_redirected_to():
        print '\n\nPHISTAR FIT'
        model.fitTo(roodata)
        fitted_theta = theta.getValV()        
        
        # Histogram binning for data points
        nbins = 10
        
        xvals = np.linspace(0, 2*pi, 200)
        yvals_H = []
        yvals_A = []
        
        # Get points for pdf curves
        for xval in xvals:
            self.phistar.setVal(xval)
            yvals_H.append(self.pdfs['H'].getValV(RooArgSet(self.phistar)))
            yvals_A.append(self.pdfs['A'].getValV(RooArgSet(self.phistar)))
        
        yvals_H = np.array(yvals_H)
        yvals_A = np.array(yvals_A)

        # Plot pdfs by themselves
        #print 'integral H =', np.trapz(yvals_H, dx=1.0/200.0)
        fig = plt.figure()
        plt.plot(xvals, yvals_H, color=self.colors['H'], label=r'$p_{H}(\varphi^{*})$')
        plt.plot(xvals, yvals_A, color=self.colors['A'], label=r'$p_{A}(\varphi^{*})$')
        plt.fill_between(xvals, 0, yvals_H, color=self.colors['H'], alpha=0.2)
        plt.fill_between(xvals, 0, yvals_A, color=self.colors['A'], alpha=0.2)
        plt.xlim([0, 2*pi])
        plt.ylim([0, 0.295])
        plt.xlabel(r'$\varphi^*$')
        plt.ylabel('Probability density')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.show()
        fig.savefig('phistar_pdfs.pdf')

        # Scale to event yield
        yvals_H *= roodata.numEntries()*(1-fitted_theta)/float(nbins)*2*pi
        yvals_A *= roodata.numEntries()*(fitted_theta)/float(nbins)*2*pi
        yvals_sum = yvals_H + yvals_A
        
        # Make plot
        fig, ax = plt.subplots(1)
        histentries, binedges = np.histogram(x, bins=nbins, range=(0, 2*pi))
        bincenters = (binedges[:-1] + binedges[1:])/2.0
        yerr = np.sqrt(histentries)
        plt.errorbar(bincenters, histentries, xerr=np.diff(binedges)*0.5, yerr=yerr, linestyle='None', ecolor='black', label='Data')
        plt.plot(xvals, yvals_H, color=self.colors['H'], label=r'$p_{H}(\varphi^{*})$')
        plt.plot(xvals, yvals_A, color=self.colors['A'], label=r'$p_{A}(\varphi^{*})$')
        plt.plot(xvals, yvals_sum, color=self.colors['model'], label=r'$p(\varphi^{*} \,|\, \alpha = %.2f)$' % fitted_theta)
        plt.fill_between(xvals, 0, yvals_H, color=self.colors['H'], alpha=0.2)
        plt.fill_between(xvals, 0, yvals_A, color=self.colors['A'], alpha=0.2)
        
        # Set correct legend order
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[3], handles[2], handles[0], handles[1]]
        labels = [labels[3], labels[2], labels[0], labels[1]]
        ax.legend(handles, labels, loc='upper right')
        
        plt.xlabel(r'$\varphi^{*}$')
        plt.ylabel('Events / %.2f' % ((2*pi)/nbins))
        
        axes = plt.gca()
        axes.set_xlim([0, 2*pi])
        axes.set_ylim([0, max(histentries)*1.8])
        
        plt.tight_layout()
        fig.show()
        fig.savefig('phistar_fit.pdf')
        
        # Create likelihood curve
        logl = model.createNLL(roodata)
        
        # Extract curve
        xvals = np.linspace(0, 1, 200)
        yvals = []
        ymin = 999.
        for xval in xvals:
            theta.setVal(xval)
            yvals.append(logl.getValV())
            if yvals[-1] < ymin:
                ymin = yvals[-1]
        
        # Shift minimum to zero
        yvals = np.array(yvals)
        yvals -= ymin
        
        # Return points for the NLL curve
        return xvals, yvals
