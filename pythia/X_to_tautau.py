"""
Generate and analyse events with ditau final states
"""

import os
from argparse import ArgumentParser
import numpy
import h5py
from cpp.Analysis import Analysis

# Output description
feature_names = ['piplus.px', 'piplus.py', 'piplus.pz', 'piplus.e',
                 'pi0plus.px', 'pi0plus.py', 'pi0plus.pz', 'pi0plus.e',
                 #'nuplus.px', 'nuplus.py', 'nuplus.pz', 'nuplus.e',
                 'piminus.px', 'piminus.py', 'piminus.pz', 'piminus.e',
                 'pi0minus.px', 'pi0minus.py', 'pi0minus.pz', 'pi0minus.e',
                 #'numinus.px', 'numinus.py', 'numinus.pz', 'numinus.e',
                 'met_x', 'met_y',
                 'inv_mass',
                 'transv_mass',
                 'upsilon_plus', 'upsilon_minus',
                 'upsilonp_x_upsilonm',
                 'triple_corr',
                 'phistar',
                 'piplus_ip_x', 'piplus_ip_y', 'piplus_ip_z',
                 'piminus_ip_x', 'piminus_ip_y', 'piminus_ip_z',
                 'pip_pim_angle',
                 'pi0p_pi0m_angle',
                 'pip_dot_pi0p',
                 'pim_dot_pi0m',
                 'pim_dot_pip',
                 'pip_dot_pi0m',
                 'pim_dot_pi0p',
                 'target'
                ]
                
feature_titles = [r'$\pi^{+} p_{x}$', r'$\pi^{+} p_{y}$', r'$\pi^{+} p_{z}$', r'$\pi^{+} E$',
                  r'$\pi^{0+} p_{x}$', r'$\pi^{0+} p_{y}$', r'$\pi^{0+} p_{z}$', r'$\pi^{0+} E$',
                  #r'$\nu^{+} p_{x}$', r'$\nu^{+} p_{y}$', r'$\nu^{+} p_{z}$', r'$\nu^{+} E$',
                  r'$\pi^{-} p_{x}$', r'$\pi^{-} p_{y}$', r'$\pi^{-} p_{z}$', r'$\pi^{-} E$',
                  r'$\pi^{0-} p_{x}$', r'$\pi^{0-} p_{y}$', r'$\pi^{0-} p_{z}$', r'$\pi^{0-} E$',
                  #r'$\nu^{-} p_{x}$', r'$\nu^{-} p_{y}$', r'$\nu^{-} p_{z}$', r'$\nu^{-} E$',
                  r'$E_{T}^{\mathrm{miss}}\;x$', r'$E_{T}^{\mathrm{miss}}\;y$',
                  r'$m_{\tau\tau}^{\mathrm{vis}}$',
                  r'$m_{T}^{\mathrm{tot}}$',
                  r'$\Upsilon^{+}$', r'$\Upsilon^{-}$',
                  r'$\Upsilon^{+} \cdot \Upsilon^{-}$',
                  r'$\mathcal{O}^{*}_{\mathrm{CP}}$',
                  r'$\varphi^{*}_{\mathrm{CP}}$',
                  r'$\pi^{+} IP x$',
                  r'$\pi^{+} IP y$',
                  r'$\pi^{+} IP z$',
                  r'$\pi^{-} IP x$',
                  r'$\pi^{-} IP y$',
                  r'$\pi^{-} IP z$',
                  r'$\mathrm{atan2}(pi^{-} p_{y}, pi^{-} p_{x})$',
                  r'$\mathrm{atan2}(pi^{0-} p_{y}, pi^{0-} p_{x})$',
                  r'$pi^{+} \cdot pi^{0+}$',
                  r'$pi^{-} \cdot pi^{0-}$',
                  r'$pi^{-} \cdot pi^{+}$',
                  r'$pi^{+} \cdot pi^{0-}$',
                  r'$pi^{-} \cdot pi^{0+}$',
                  r'Target',
                 ]





def generate_X_to_tautau(commandfile, target, nevents, outname, massrange, smearing=False, debug=False):
    
    # If the args.commandfile string contains a /, assume we got a relative path
    # Otherwise, append the path to the 'processes' directory
    if not '/' in commandfile:
        commandfile = 'processes/' + commandfile

    # Check that it exists
    if not os.path.exists(commandfile):
        print 'File not found:', commandfile
        #exit(-1)
        return None


    if len(feature_names) != len(feature_titles):
        print 'len(feature_names) =', len(feature_names),
        print '!= len(feature_titles) =', len(feature_titles)

    # Initialise
    ana = Analysis(commandfile, target=target, restframe='visible', random_seed=0, apply_smearing=smearing, debug=debug) # seed = 0 -> use Stdlib time(0)
        
    # Set selection requirements
    assert isinstance(massrange, list)
    ana.set_cuts(min_leading_pt=40., min_subleading_pt=40.,
                 max_eta=2.1,
                 deltaR=0.5,
                 min_MET=0.,
                 mass_lower=massrange[0], mass_upper=massrange[1])
    

    # Generate and analyse events
    data = []
    i = 0
    while ana.get_n_accepted_events() < nevents:
        i += 1
        if (i % 1000 == 0):
            n = ana.get_n_accepted_events()
            print ' %d events accepted (%.0f %%) [%s]' % (n, 100*float(n)/nevents, commandfile.split('/')[-1])
        result = ana.process_event()
        if len(result):
            data.append(result)

    # Extract the selection efficiency and cross section
    eff = ana.get_efficiency()
    tried_events = ana.get_n_tried_events()
    passed_mass_cuts = ana.get_n_passed_mass_cuts()
    accepted_events = ana.get_n_accepted_events()
    xsec = ana.get_cross_section()
    

    data = numpy.array(data)

    # Get process name
    cmndname = commandfile[commandfile.rfind('/')+1:].replace('.cmnd', '')
    if not outname:
        outname = cmndname

    # Get process title (first line in cmnd file)
    with open(commandfile, 'r') as cmndfile:
        process_title = cmndfile.readline()[2:]
        if debug:
            cmndfile.seek(0)
            print 'Contents of command file:\n'
            print cmndfile.read()
            print '\n(end)\n'


    # Save
    if not outname.endswith('.h5'):
        outname += '.h5'
    with h5py.File(outname, 'w') as hf:
        dset = hf.create_dataset('data', data.shape, data=data)
        dset.attrs['process_name'] = cmndname
        dset.attrs['process_title'] = process_title
        dset.attrs['feature_names'] = numpy.array(feature_names)
        dset.attrs['feature_titles'] = numpy.array(feature_titles)
        dset.attrs['efficiency'] = numpy.float32(eff)
        dset.attrs['total_events_tried'] = numpy.uint64(tried_events)
        dset.attrs['events_passed_mass_cuts'] = numpy.uint64(passed_mass_cuts)
        dset.attrs['events_accepted'] = numpy.uint64(accepted_events)
        dset.attrs['cross_section'] = numpy.float32(xsec)

    del ana
    print 'Saved data to', outname
    







if __name__ == "__main__":

    parser = ArgumentParser(description='Generate events using Pythia8.2')
    parser.add_argument('commandfile', help='Input Pythia .cmnd file')
    parser.add_argument('-t', '--target', type=int, help='Target class (0/1)', required=True)
    parser.add_argument('-n', '--nevents', type=int, help='Number of events to generate',
                        default=10)
    parser.add_argument('-o', '--outname', help='Name of output file (default: '
                        'same name as input cmnd file)')
    # parser.add_argument('-rf', '--restframe', help='Name of restframe that features '
    #                     'are computed in. Can be \'lab\', \'truth\', \'visible\','
    #                     'or \'charged\'', default='visible')
    parser.add_argument('-htt', '--requireHtt', help='Require Htautau decay in event', action='store_true')
    parser.add_argument('-s', '--smear', help='Apply detector smearing', action='store_true')
    parser.add_argument('-d', '--debug', help='Print a bit of debug output', action='store_true')
    args = parser.parse_args()


    generate_X_to_tautau(args.commandfile, args.target, args.nevents,
                         args.outname, [0, 10e8], args.smear, args.debug)
