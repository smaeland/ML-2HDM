# pylint: disable=C0303, C0103

## Class for a 2HDM model, contains parameters and computes xsec / BR

import os
import subprocess
import argparse
import h5py
import numpy as np
from glob import glob
from lhatool import LHA, Block, Entry
from dispatcher import Dispatcher


class Model(object):
    """ All properties of a model and methods to run Pythia, 2HDMC, SusHi """
    
    
    def __init__(self, param_dict, gen_settings_dict,
                 lumi, massrange,
                 backgrounds,
                 outdir,
                 title,
                 ignore_higgsbounds=False):
        """
        paramdict: Model parameters in a dict, i.e {'mH': 450, ...}
        gen_settings_dict: Settings for sample generation
        lumi: luminosity in fb-1
        """
        super(Model, self).__init__()
        
        # Sanity check
        expected_params = set(['mh', 'mH', 'mA', 'mC',
                               'sin_ba', 'm12_2', 'tanb',
                               'lambda_6', 'lambda_7'])
        diff = set(param_dict.keys()).symmetric_difference(expected_params)
        if diff:
            print 'Problem with parameter(s):', diff
            exit(-1)
        
        self.params = param_dict
        self.title = title
        self.lumi = lumi
        self.massrange = massrange
        self.ignore_higgsbounds = ignore_higgsbounds
        
        self.gen_settings = gen_settings_dict
        
        # Samples: Signal and backgrounds
        if backgrounds is None:
            backgrounds = []
        self.backgrounds = backgrounds
        self.samples = ['H', 'A'] + backgrounds
        
        
        # Properties
        self.xsec = {'A': None, 'H': None}
        self.xsec_err = {'A': None, 'H': None}
        
        self.br_tau_pipi = 0.2552*0.2552
        
        self.br_tautau = {
            'A': None,
            'H': None,
            'Z': 3.337e-2,
            'ttbar': 11.38e-2*11.38e-2,
            'VV': 11.38e-2*11.38e-2,        # Assuming WW dominates
            }
        
        self.efficiency = {'A': None, 'H': None}
        self.nexpected = {'A': None, 'H': None}
        
        self.targets = {
            'H': 0,
            'A': 1,
            'Z': 2,
            'ttbar': 2,
            'VV': 2,
            }
        
        
        # Goto work dir
        self.originalpath = os.getcwd()
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        os.chdir(outdir)
        self.outdir = os.getcwd()
        
        print 'self.outdir:', self.outdir
        
        # Create train/test/validate directories
        for d in ['train', 'test', 'validation']:
            if not os.path.exists(d):
                os.mkdir(d)
        
        # I/O
        self.lhafile = None
        self.sushi_input = {'A': None, 'H': None}
        self.sushi_output = {'A': None, 'H': None}
        
        self.trainfiles = {'A': None, 'H': None}
        
        # TODO: improve cmnd file locations
        self.cmndfiles = {
            'A': None,
            'H': None,
            'Z': self.originalpath + '/../../pythia/processes/Z0_tautau_hadhad.cmnd',
            'VV': self.originalpath + '/../../pythia/processes/diboson_tautau_hadhad.cmnd',
            'ttbar': self.originalpath + '/../../pythia/processes/ttbar_tautau_hadhad.cmnd'}
        
        
        # Try to retrieve existing files
        if os.path.exists(self.title + '.lha'):
            self.lhafile = os.path.abspath(self.title + '.lha')
            print 'Located LHA file:', self.lhafile
        
        for key in self.samples:
            
            if key in ['H', 'A']:
            
                # SusHi inputs
                if os.path.exists('%s_%s_sushi.in' % (self.title, key)):
                    self.sushi_input[key] = os.path.abspath('%s_%s_sushi.in' % (self.title, key))
                    print 'Located SusHi input file:', self.sushi_input[key]
                # SusHi outputs
                if os.path.exists('%s_%s_sushi.out' % (self.title, key)):
                    self.sushi_output[key] = os.path.abspath('%s_%s_sushi.out' % (self.title, key))
                    print 'Located SusHi output file:', self.sushi_output[key]
            
            # Pythia cmnd files
            if os.path.exists('%s_%s.cmnd' % (self.title, key)):
                self.cmndfiles[key] = os.path.abspath('%s_%s.cmnd' % (self.title, key))
                print 'Located Pythia command file:', self.cmndfiles[key]
            
            # Train sets for this model
            if os.path.exists('train/%s_%s_merged.h5' % (self.title, key)):
                self.trainfiles[key] = os.path.abspath('train/%s_%s_merged.h5' % (self.title, key))
                print 'Located train set:', self.trainfiles[key]

        os.chdir(self.originalpath)

        
        
    
    def compute_decay_table(self):
        """
        Run 2HDMC to calculate BRs and widths
        Also runs HiggsBounds/Signals to check points.
        """
        
        outfile = '%s/%s.lha' % (self.outdir, self.title)
        
        cmnd = [os.environ['TWOHDMCCODEPATH'] + '/calculate_point',
                self.params['mh'],
                self.params['mH'],
                self.params['mA'],
                self.params['mC'],
                self.params['sin_ba'],
                self.params['lambda_6'],
                self.params['lambda_7'],
                self.params['m12_2'],
                self.params['tanb'],
                outfile,
                int(self.ignore_higgsbounds)
                ]
        
        cmnd = [str(c) for c in cmnd]
        print 'Running command:', cmnd
        
        self.lhafile = outfile
        
        return subprocess.check_call(cmnd)
        
    
    
    def prepare_sushi_input(self):
        """
        Read LHA file from 2HDMC, and create a .in file for SusHi
        
        higgstype:
            11: h
            12: H
            21: A
        """
        
        for higgsname, higgstype in {'H': 12, 'A': 21}.iteritems():
        
            # Parse LHA file
            lha = LHA(self.lhafile)
            
            # Add SusHi-specific blocks
            sushi = Block('SUSHI', comment='SusHi specific')
            sushi.add(Entry([1, 2], comment='Select 2HDM'))
            sushi.add(Entry([2, higgstype], comment='h / H / A'))
            sushi.add(Entry([3, 0], comment='p-p collisions'))
            sushi.add(Entry([4, 13000], comment='E_cm'))
            sushi.add(Entry([5, 2], comment='ggH at NNLO'))
            sushi.add(Entry([6, 2], comment='bbH at NNLO'))
            sushi.add(Entry([7, 2], comment='SM EW content'))
            sushi.add(Entry([19, 1], comment='Verbosity'))
            sushi.add(Entry([20, 0], comment='All processes'))
            lha.add_block(sushi)
            
            # 2HDM block
            thdm = Block('2HDM', '2HDM parameters')
            thdm.add(Entry([1], comment='Type I'))
            lha.add_block(thdm)
            
            # Kinematic distribution parameters
            distrib = Block('DISTRIB', comment='Kinematic requirements')
            distrib.add(Entry([1, 0], comment='Sigma total'))
            distrib.add(Entry([2, 0], comment='Disable pT cut'))
            #distrib.add(Entry([21, GENER_SETTINGS['higgs_pt_min']], comment='Min higgs pT'))
            distrib.add(Entry([3, 0], comment='Disable eta cut'))
            #distrib.add(Entry([32, GENER_SETTINGS['higgs_eta_max']], comment='Max eta'))
            distrib.add(Entry([4, 1], comment='Use eta, not y'))
            lha.add_block(distrib)
            
            # PDF selection
            pdfspec = Block('PDFSPEC')
            pdfspec.add(Entry([1, 'MMHT2014lo68cl.LHgrid'], comment='Name of pdf (lo)'))
            pdfspec.add(Entry([2, 'MMHT2014nlo68cl.LHgrid'], comment='Name of pdf (nlo)'))
            pdfspec.add(Entry([3, 'MMHT2014nnlo68cl.LHgrid'], comment='Name of pdf (nnlo)'))
            pdfspec.add(Entry([4, 'MMHT2014nnlo68cl.LHgrid'], comment='Name of pdf (n3lo)'))
            pdfspec.add(Entry([10, 0], comment='Set number'))
            lha.add_block(pdfspec)
            
            # Add charm mass
            lha.get_block('SMINPUTS').add(Entry([8, 1.275], comment='m_c'))
            
            # Write output
            suffix = '_%s_sushi.in' % higgsname
            outname = self.lhafile.replace('.lha', suffix)
            self.sushi_input[higgsname] = outname
            
            lha.write(outname)
        
        return 0
        
    

    def get_cross_section(self, sample):
        """
        For higgses, compute the ggF cross section with SusHi
        For backgrounds, take value from Pythia
        
        Return value in pb
        """
        
        # Path to SusHi 
        sushi_binary = os.environ['SUSHIPATH']
        if not os.path.exists(sushi_binary):
            print 'No known SusHi binary file'
            exit(-1)
        
        if sample in ['H', 'A']:
            
            self.sushi_output[sample] = self.sushi_input[sample].replace('.in', '.out')
            
            # Convert to relative path to shorten the file name, since SusHi
            # can't deal with inputs >60 chars
            relpath_in = os.path.relpath(self.sushi_input[sample], os.getcwd())
            relpath_out = os.path.relpath(self.sushi_output[sample], os.getcwd())
            
            # Run SusHi
            ret = subprocess.check_call([sushi_binary,
                                         relpath_in,
                                         relpath_out])
                                         #self.sushi_input[sample],
                                         #self.sushi_output[sample]])
            
            if ret: return ret
            
            # Parse result 
            lha = LHA(self.sushi_output[sample])
            
            self.xsec[sample] = float(lha.get_block('SUSHIggh').get_entry_by_key(1))
            
            # Compare to Pythia
            with h5py.File(self.trainfiles[sample]) as hf:
                xsec = float(hf.get('data').attrs['cross_section'])
                xsec = xsec * 10e9  # convert from mb to pb
                
                print 'SAMPLE:', sample, ':\tSusHi = %.4e, \t Pythia = %.4e' % (self.xsec[sample], xsec)
            
        elif sample == 'Z':
            self.xsec[sample] = 2.7910 # from FEWZ at LO 

        elif sample in self.backgrounds and sample != 'Z':
            
            # Open train set
            with h5py.File(self.trainfiles[sample]) as hf:
                xsec = float(hf.get('data').attrs['cross_section'])
                self.xsec[sample] = xsec * 10e9  # convert from mb to pb
        
        #print 'Cross section for %s = %.3e pb' % (sample, self.xsec[sample])
        
        return 0
        
    
    
    def create_pythia_cmnd_files(self):
        """
        Create a command file for A/H -> tautau
        Needs higgs masses and decay widths
        """
        
        for higgsname, higgspid in {'H': 35, 'A': 36}.iteritems():
        
            # Get mass and width from 2HDMC LHA file
            lha = LHA(self.lhafile)
            mass = lha.get_block('MASS').get_entry_by_key(higgspid)
            width = lha.get_decay(higgspid).width        
            
            outname = self.lhafile.replace('.lha', '_%s.cmnd' % higgsname)
            self.cmndfiles[higgsname] = outname
        
            # Write command file
            with open(outname, 'w') as outfile:
            
                outfile.write('Beams:eCM = 13000.\n')
                outfile.write('Higgs:useBSM = on\n')
                
                if higgspid == 36:
                    #outfile.write('HiggsBSM:allA3 = on\n')     # All production modes
                    outfile.write('HiggsBSM:ffbar2A3 = on\n')   # quark fusion
                    outfile.write('HiggsBSM:gg2A3 = on\n')      # gluon fusion
                elif higgspid == 35:
                    #outfile.write('HiggsBSM:allH2 = on\n')     # All production modes
                    outfile.write('HiggsBSM:ffbar2H2 = on\n')   # quark fusion
                    outfile.write('HiggsBSM:gg2H2 = on\n')      # gluon fusion
                
                outfile.write('{}:all = A0 A0 1 0 0 {} {}  50.0 0.0\n'.format(higgspid, mass, width))
                outfile.write('{}:onMode = off\n'.format(higgspid))
                outfile.write('{}:onIfMatch = 15 -15\n'.format(higgspid))
                
                outfile.write('15:onMode = off\n')
                outfile.write('15:onIfMatch = 16 111 211\n')
                outfile.write('\n')
                outfile.write('Next:numberShowEvent = 0\n')

        return 0


    


    
    def compute_expected_event_numbers(self, sample):
        
        # Get generation efficiency from train sets
        assert self.trainfiles[sample] is not None
        with h5py.File(self.trainfiles[sample]) as hfile:
            data = hfile.get('data')
            self.efficiency[sample] = float(data.attrs['efficiency'])
            if sample == 'Z':
                self.efficiency[sample] = float(data.attrs['events_accepted'])/float(data.attrs['events_passed_mass_cuts'])
        
        
        # Signal
        if sample in ['H', 'A']:
            
            # Branching ratios
            lha = LHA(self.lhafile)
            self.br_tautau['H'] = float(lha.get_decay(35).get_branching_ratio(15, -15))
            self.br_tautau['A'] = float(lha.get_decay(36).get_branching_ratio(15, -15))
        
            

        # Number of expected events
        pb_to_fb = 10e3
        self.nexpected[sample] = (self.lumi * self.xsec[sample] * pb_to_fb * 
                                  self.br_tautau[sample] * self.br_tau_pipi *
                                  self.efficiency[sample])
        
        self.nexpected[sample] = int(round(self.nexpected[sample]))
        
        res = [sample, self.lumi, self.xsec[sample] * pb_to_fb,
               self.br_tautau[sample], self.efficiency[sample],
               self.nexpected[sample]]
        
        
        #print '\nExpected event numbers:'
        st = ['', 'Lumi (fb-1)', 'xsec (fb)', 'BR', 'efficiency', 'N']
        print '{:4} {:>15} {:>15} {:>15} {:>15} {:>15}'.format(*st)
        print '{:4} {:15.1f} {:15.4e} {:15.4e} {:15.4e} {:15d}'.format(*res)
        

    
    def merge_datasets(self, name):
        
        print 'Merging %s datasets' % name
        assert name in ['train', 'test', 'validation']
        
        def merge(filelist, outname, remove_originals=False):
            
            if len(filelist) < 1:
                print 'Error: No files to merge'
                return
            
            atts = {}
            open_files = []
            X = np.array([])
            
            for fin in filelist:
                
                # Skip old merge file
                if fin == outname:
                    continue
                
                hf = h5py.File(fin, 'r')
                open_files.append(hf)
                data = hf.get('data')

                if not len(data):
                    print 'In merge_datasets(): File', fin, 'is empty'
                    continue
                
                # Set metadata
                if len(atts) == 0:
                    # Copy everything from first file
                    for att, info in data.attrs.iteritems():
                        atts[att] = info
                else:
                    for att, info in data.attrs.iteritems():
                        
                        # Sum up tried, accepted event numbers
                        if att in ['total_events_tried', 'events_accepted', 'events_passed_mass_cuts']:
                            atts[att] += info
                            #print att, 'is now:', atts[att]
                    
                        # Compute efficiency
                        elif att == 'efficiency':
                            atts[att] = float(atts['events_accepted'])/float(atts['total_events_tried'])
                        
                        # Check that the rest matches
                        else:
                            
                            # Don't check these
                            if 'feature' in att or 'cross_section' in att or 'process_name' in att:
                                continue
                            
                            if atts[att] != info:
                                print 'Merge: Attribute', att, 'differs:',
                                print 'existing =', atts[att], '- new = ', info
                    
                
                data = np.array(data)
                
                # Add up the data
                if len(X):
                    assert data.shape[1] == X.shape[1]
                    X = np.vstack((X, data))
                else:
                    X = data
            
            # Create new file
            print 'Creating', outname
            if os.path.exists(outname):
                os.remove(outname)
            hout = h5py.File(outname, 'w')
            dout = hout.create_dataset('data', X.shape, data=X)
            for att, info in atts.iteritems():
                dout.attrs[att] = info
            
            hout.close()
            for of in open_files:
                of.close()
                
            if remove_originals:
                for fin in filelist:
                    os.remove(fin)
        
        
        #if name == 'train':
        os.chdir(self.outdir+'/'+name)
        for sample in self.samples:
            files = glob('%s_%s*.h5' % (self.title, sample))
            mname = '%s_%s_merged.h5' % (self.title, sample)
            merge(files, mname, remove_originals=True)
            #merge(files, mname, remove_originals=False)
            
            # Set path to train sets
            self.trainfiles[sample] = os.path.abspath(mname)

        os.chdir('..')
        
        """ Old
        elif name == 'test':
            os.chdir(self.outdir+'/test')
            files = glob('*.h5')
            merge(files, 'all_merged_TEST.h5')
            os.chdir('..')
        
        elif name == 'validation':
            
            os.chdir(self.outdir+'/validation')
            for iset in range(1, self.gen_settings['n_validation_samples'] + 1):
                os.chdir('val{:04d}'.format(iset))
                files = glob('*.h5')
                #print 'now in:', os.getcwd()
                #print 'found files:', files
                merge(files, 'all_merged_VALIDATION_{:04d}.h5'.format(iset),
                      remove_originals=True)
                os.chdir('..')
        """
        
        os.chdir(self.originalpath)        
        
        print 'Done'


