# pylint: disable=C0303

"""
Base classes for predictors
"""

import os, sys
import numpy as np
from collections import namedtuple
import h5py
from glob import glob
import pickle
from contextlib import contextmanager
from array import array
from copy import deepcopy
import ROOT
from math import sqrt

# Import matplotlib if available
try:
    import matplotlib.pyplot as plt
    disable_graphics = False
except ImportError:
    disable_graphics = True


"""
Simple tuple containing predicted results
The predict() function in all predictors should return this
"""
Prediction = namedtuple('Prediction', ['nH', 'nA', 'nBgnd'])

# Return theta instead



class Histogram(object):
    """ 
    A ROOT TH1 histogram constructed from a numpy hist 
    
    Arguments:
        arr: Values to fill in histogram, list or numpy array
        binedges: See documentation of numpy.histogram()
        title: Histogram title (str)
        sample: Additional nametag to ensure similar histograms have unequal
            titles
    """
    
    # TODO: normed = True
    
    def __init__(self, arr, binedges, title, sample):
        
        self.title = title
        print ' DBG: creating hist with binedges:', binedges
        self.np_hist, self.bin_edges = np.histogram(arr, bins=binedges)
        self.np_hist = self.np_hist.astype(float)
        
        # Construct TH1
        self.th1 = ROOT.TH1F('h_%s_%s' % (title, sample), 'h_%s_%s' % (title, sample),
                             len(self.np_hist), array('d', self.bin_edges))
        for ibin, binval in enumerate(self.np_hist):
            self.th1.SetBinContent(ibin+1, binval)
            self.th1.SetBinError(ibin+1, np.sqrt(binval))


    def save_to_file(self):
        """ Picle it """
        
        with open('histo_%s.pkl' % self.title, 'w') as hout:
            pickle.dump(self, hout, -1)





class BasePredictor(object):
    """
    Abstract base class for predictors.
    """
    

    def __init__(self, title='None'):
        self.title = title
        
        self.outdir = 'output_%s' % title
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
    
    def predict(self, x):
        """
        Make predictions for input np.array
        Pass column names to allow predictor to pick out the correct features
        
        Return Prediction instance
        """
        raise NotImplementedError
    
    
    def read_array_from_files(self, filelist):
        """
        Read a list of .h5 files and return a numpy array with data, an array
        with targets, and a list of features
        names
        """
        
        X = np.array([])
        Y = np.array([])
        features = None
        
        for fin in filelist:
            if not 'validation' in fin.lower():
                print 'Processing file:', fin
            
            hf = h5py.File(fin, 'r')
            data = hf.get('data')
            features = data.attrs['feature_names']
            data = np.array(data)
            
            # Add up the data
            if len(X):
                assert data.shape[1] == X.shape[1]
                X = np.vstack((X, data))
            else:
                X = data
        
        # Extract targets
        Y = np.array(X[:, -1])
        X = np.delete(X, -1, axis=1) # remove targets from data
        
        return X, Y, features
    




class TemplatePredictor(BasePredictor):
    """
    Base predictor for template fits.
    
    Arguments:
        title: string
        adapted_bins: Adapt template bin edges to data
    """

    def __init__(self, title='None', adapted_bins=True):
        
        self.title = title
        self.templates = {}
        self.samples = []
        self.path_to_data = None
        

        # Dummy histogram settings
        self.histrange = [0, 1]
        self.nbins = 1
        self.binedges = [0, 1]
        self.bincenters = 0.5
        
        self.adapted_bins = adapted_bins


    def create_template(self, x, title, istrain=True, save=False):
        """ Create histograms to be used for templates """    
        raise NotImplementedError
    
    
    
    def create_train_templates(self, paths_to_data):
        """
        Create templates from training files. Background samples are merged into
        a single histogram.
        
        Arguments:
            paths_to_data: List of paths to directories containing training
                data. Typically only one, but can also train on multiple
                datasets from different models (i.e. different masses)
        """
        
        if not isinstance(paths_to_data, list):
            paths_to_data = [paths_to_data]
        
        files_H = []
        files_A = []
        for path in paths_to_data:
            print 'looking at', path
            # These are merged, so only one of each
            if not path.endswith('/'): path += '/'
            files_H.append(path+'model_H_merged.h5')
            files_A.append(path+'model_A_merged.h5')

 
        # Find background samples, put into one template
        files_bgnd = []
        for path in paths_to_data:
            for sample in self.samples:
                if sample not in ['H', 'A']:
                    files_bgnd.append(path+'/train/model_%s_merged.h5' % sample)
        
        # Make the templates
        dataH_X, _, feats = self.read_array_from_files(files_H)
        dataH_X = self.select_features(dataH_X, feats)
        print 'WARNING: LIMITING TEMPLATES'
        dataH_X = dataH_X[:50000]
        self.templates['H'] = self.create_template(dataH_X, 'H', istrain=True, save=True)
        
        dataA_X, _, feats = self.read_array_from_files(files_A)
        dataA_X = self.select_features(dataA_X, feats)
        dataA_X = dataA_X[:50000]
        self.templates['A'] = self.create_template(dataA_X, 'A', istrain=True, save=True)
    
        if files_bgnd:
            dataBgnd_X, _, feats = self.read_array_from_files(files_bgnd)
            dataBgnd_X = self.select_features(dataBgnd_X, feats)
            self.templates['bgnd'] = self.create_template(dataBgnd_X, 'bgnd', istrain=True, save=True)
    
        print 'Created train templates'
    
    
        
    def predict(self, x, plot=False, aux_yield=None):
        """
        Make predictions for data points x from a single observable.
        
        Arguments:
            x:  Numpy array of shape (?,)
            plot: Plot the resulting fit
            aux_yield: Aid the fit by providing an auxiliary measurement. See
                fit()
        """

        # Make sure we have valid templates
        assert self.templates['H'] is not None, 'No valid H template -- run first with \'-r/--recreate\''
        assert self.templates['A'] is not None, 'No valid A template -- run first with \'-r/--recreate\''
        
        datahist = Histogram(x, self.binedges, self.title, 'data')
        
        return self.fit(datahist,
                        histscalar=self.templates['H'],
                        histpseudo=self.templates['A'],
                        histbgnd=self.templates['bgnd'],
                        aux_measurement_for_bgnd_yield=aux_yield,
                        plot=plot)

    
    
    def fit(self, histdata, histscalar, histpseudo, histbgnd=None, 
            aux_measurement_for_bgnd_yield=None, plot=False):
        """
        Run the template fit to extract yields. All input histograms should be
        normalised to unity, so that a single normfactor can be applied which
        corresponds to the exact expected yield.

        In case we run with backgrounds, one can try to stabilise the fit by
        adding an auxiliary measurement for the background yield, which is
        enabled by providing a number as the aux_measurement_for_bgnd_yield
        argument. In this case, an auxiliary measurement channel is created,
        containing only background. The number of events in the aux measurement
        is a poisson variation around the input value.
        
        Arguments:
            histdata: Histogram containing test set
            histscalar: Template histogram for H
            histpseudo: Template histogram for A
            histbgnd: Combined template histogram for all backgrounds (optional)
            aux_measurement_for_bgnd_yield: int
            plot: Make a plot
        
        Returns:
            Tuple containing Prediction instance, and negative log likelihood
            value
        """
        
        # Get normalised TH1's
        hdata = deepcopy(histdata.th1)
        hscalar = deepcopy(histscalar.th1)
        hpseudo = deepcopy(histpseudo.th1)
        if histbgnd:
            hbgnd = deepcopy(histbgnd.th1)
        
        print 'hdata bins before:'
        lol = hdata.GetXaxis().GetXbins()
        for i in range(lol.GetSize()):
            print ' ', lol.At(i)

        # Make equidistant bins, since HistFactory can't deal with variable binning
        if 'keras' in self.title:
            print 'equidistansing!'
            def make_equidistant(hist):
                bincontents = []
                binerrors = []
                entries = hist.GetEntries()
                
                for i in range(1, hist.GetNbinsX()+1):
                    bincontents.append(hist.GetBinContent(i))
                    binerrors.append(hist.GetBinError(i)) 
                
                newhist = ROOT.TH1F(hist.GetName()+'_equi', hist.GetTitle()+' (equi)', hist.GetNbinsX(), -0.5, hist.GetNbinsX()-0.5)

                for i in range(1, newhist.GetNbinsX()+1):
                    newhist.SetBinContent(i, bincontents[i-1])
                    newhist.SetBinError(i, binerrors[i-1])

                hist = newhist

            make_equidistant(hdata)
            make_equidistant(hscalar)
            make_equidistant(hpseudo)

        print 'hdata bins after:'
        lol = hdata.GetXaxis().GetXbins()
        for i in range(lol.GetSize()):
            print ' ', lol.At(i)
        
        # Reasonable initial guess
        n_init = hdata.Integral()/2
        if histbgnd:
            n_init = hdata.Integral()/3
        
        meas = ROOT.RooStats.HistFactory.Measurement('meas', 'meas')
        meas.SetOutputFilePrefix('fitresults/')
        meas.SetPOI('nH')
        meas.AddPOI('nA')

        meas.SetLumi(1)
        #meas.SetLumiRelErr(0.1)
        meas.AddConstantParam('Lumi')
        meas.SetExportOnly(True)
        
        chan = ROOT.RooStats.HistFactory.Channel('chan')
        #chan.SetStatErrorConfig(0.03, "Poisson")
        chan.SetData(hdata)
        
        scalar = ROOT.RooStats.HistFactory.Sample('scalar')
        scalar.SetHisto(hscalar)
        scalar.SetNormalizeByTheory(False)
        scalar.AddNormFactor('nH', n_init, 1.0, 10000.0)
        scalar.ActivateStatError()
        chan.AddSample(scalar)
        
        pseudoscalar = ROOT.RooStats.HistFactory.Sample('pseudoscalar')
        pseudoscalar.SetHisto(hpseudo)
        pseudoscalar.SetNormalizeByTheory(False)
        pseudoscalar.AddNormFactor('nA', n_init, 1.0, 10000.0)
        pseudoscalar.ActivateStatError()
        chan.AddSample(pseudoscalar)
        
        if histbgnd:
            bgnd = ROOT.RooStats.HistFactory.Sample('background')
            bgnd.SetHisto(hbgnd)
            bgnd.SetNormalizeByTheory(False)

            bgnd_norm = ROOT.RooStats.HistFactory.NormFactor()
            bgnd_norm.SetName('nBgnd')
            bgnd_norm.SetLow(100); bgnd_norm.SetHigh(3000); bgnd_norm.SetVal(1500)
            bgnd.AddNormFactor(bgnd_norm) 
            bgnd.ActivateStatError()
            chan.AddSample(bgnd)


        meas.AddChannel(chan)
        
        
        if aux_measurement_for_bgnd_yield:
            aux_chan = ROOT.RooStats.HistFactory.Channel('aux')
            
            aux_data = ROOT.TH1F('aux_data', 'aux_data', 1, 0, 1)
            aux_data.SetBinContent(1, np.random.poisson(aux_measurement_for_bgnd_yield))
            aux_data.SetBinError(1, sqrt(aux_measurement_for_bgnd_yield))
            aux_chan.SetData(aux_data)

            aux_hist = ROOT.TH1F('aux_hist', 'aux_hist', 1, 0, 1)
            aux_bgnd = ROOT.RooStats.HistFactory.Sample('background')
            aux_bgnd.SetHisto(aux_hist)
            aux_bgnd.SetNormalizeByTheory(False)
            aux_bgnd.AddNormFactor(bgnd_norm)

            aux_chan.AddSample(bgnd)
            meas.AddChannel(aux_chan)


        
        with stdout_redirected_to('%s/histfactory_output.log' % self.outdir):

            factory = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast()
            workspace = factory.MakeCombinedModel(meas)
            #ROOT.RooStats.HistFactory.FitModel(workspace)
            
            # 'Manual' fit
            mc = workspace.obj('ModelConfig')
            data = workspace.data('obsData') 
            pdf = mc.GetPdf()
            fitres = pdf.fitTo(data, ROOT.RooFit.Minos(True), ROOT.RooFit.PrintLevel(1), ROOT.RooFit.Save(True))
            fitres.Print()
            nll = fitres.minNll()

            meas.PrintTree()
            

        numH = workspace.var('nH').getValV()
        numH_err = workspace.var('nH').getError()
        
        numA = workspace.var('nA').getValV()
        numA_err = workspace.var('nA').getError()
        
        if histbgnd:
            numBgnd = workspace.var('nBgnd').getValV()
            numBgnd_err = workspace.var('nBgnd').getError()
        else:
            numBgnd = -1
            numBgnd_err = 0


        # Draw plot
        if plot and disable_graphics:
            print 'WARNING: Graphics disabled'
            
        elif plot and not disable_graphics:
            fig, ax = plt.subplots(1)
            
            binwidths = np.diff(self.binedges)
            binweights = np.array(((self.binedges[-1]-self.binedges[0])/len(binwidths))/binwidths)

            # Get bin centers
            bincenters = []
            for i in range(len(self.binedges)-1):
                bincenters.append(self.binedges[i] + (self.binedges[i+1]-self.binedges[i])*0.5)
            self.bincenters = bincenters

            # Data
            ydata = histdata.np_hist
            yerrdata = np.sqrt(ydata)
            plt.errorbar(self.bincenters, ydata*binweights, xerr=binwidths/2.0, yerr=yerrdata,
                         linestyle='None', label='Data')

            # Scale templates with result from fit
            yscalar = histscalar.np_hist
            yscalar *= numH/np.sum(yscalar)

            ypseudo = histpseudo.np_hist
            ypseudo *= numA/np.sum(ypseudo)

            if histbgnd:
                ybgnd = histbgnd.np_hist
                ybgnd *= numBgnd/np.sum(ybgnd)

            # Combined model
            ycomb = yscalar + ypseudo
            if histbgnd:
                ycomb += ybgnd
            
            
            # Plot
            plt.bar(left=self.binedges[:-1], width=np.diff(self.binedges), height=yscalar*binweights,
                    alpha=0.3, align='edge', label=r'$H$ template')

            plt.bar(left=self.binedges[:-1], width=np.diff(self.binedges), height=ypseudo*binweights,
                    alpha=0.3, align='edge', label=r'$A$ template')

            if histbgnd:
                plt.bar(left=self.binedges[:-1], width=np.diff(self.binedges), height=ybgnd*binweights,
                        alpha=0.3, align='edge', label=r'Bgnd template')


            #plt.plot(self.bincenters, ycomb, ls='steps',
            #         label='Fit')
            plt.plot(self.binedges, np.insert(ycomb, 0, ycomb[0])*np.insert(binweights, 0, binweights[0]), ls='steps',
                     label='Fit')

            # Set correct legend order
            handles, labels = ax.get_legend_handles_labels()
            handles = [handles[1], handles[2], handles[3], handles[0]]
            labels = [labels[1], labels[2], labels[3], labels[0]]

            if histbgnd:
                handles, labels = ax.get_legend_handles_labels()
                handles = [handles[1], handles[2], handles[3], handles[4], handles[0]]
                labels = [labels[1], labels[2], labels[3], labels[4], labels[0]]

            plt.xlabel(r'$\varphi^{*}$')
            plt.ylabel('Events')
            ax.legend(handles, labels, loc='upper right')


            axes = plt.gca()
            axes.set_xlim([self.histrange[0], self.histrange[1]])
            axes.set_ylim([0, max(ycomb*binweights)*1.5])

            fig.show()


        return Prediction(numH, numA, numBgnd), nll

    
    
    def select_features(self, data, feats):
        raise NotImplementedError
    
    

@contextmanager
def stdout_redirected_to(to=os.devnull):
    """
    Redirect stdout to a file, useful for C libraries printing lots of stuff (yes
    HistFactory, looking at you). Lifted from http://stackoverflow.com/questions/
    5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/17954769#17954769

    Usage:
    import os
    with stdout_redirected_to(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

