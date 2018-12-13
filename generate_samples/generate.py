# pylint: disable=C0303, C0103

"""
Generate training, test and validation datasets
"""

import os
from argparse import ArgumentParser
import time, datetime

from generate_samples.model import Model
from generate_samples.dispatcher import Dispatcher, JobParams



def submit_set(setname, models, ncpus):
    
    assert setname in ['train', 'validation', 'test'], 'Unrecognised set name'
    
    dispatcher = Dispatcher(ncpus)
    
    for model in models:
        
        # Check if files could be overwritten
        output_path = model.outdir+'/'+setname
        if os.path.isdir(output_path) and os.listdir(output_path):
            cont = raw_input('Files in %s will be overwritten - continue? ' % output_path)
            if cont.lower() != 'y':
                exit(-1)
        
        # Create cmnd files for all models, run 2HDMC
        if setname == 'train':
            model.compute_decay_table()
            model.create_pythia_cmnd_files()
        
        # Create job commands for all samples
        for sample in model.samples:
            
            params = JobParams(
                cmndfile=model.cmndfiles[sample],
                target_val=model.targets[sample],
                outname=model.outdir+'/'+setname+'/' + model.title + '_' + sample + '.h5',
                nevents=model.gen_settings['n_events'][sample],
                massrange=model.massrange,
                smearing=model.gen_settings['smearing']
                )
            
            dispatcher.submit(params, split=model.gen_settings['n_splits'])
            
            print 'Submitting:', params
        
    
    # Wait for train sets to finish
    dispatcher.start()
    dispatcher.close()
    
    # Merge outputs
    for model in models:
        model.merge_datasets(setname)






def run(args, trainmodels, testmodels):
    """
    Generate datasets.
    
    First:
    - Run 2HDMC to check point and get decay widths
    - Create Pythia cmnd files for H and A -> tautau
    - Generate train sets. This is done in parallel.
    
    Train sets contain info about efficiency and Pythia cross section.
    Now,
    - Prepare input LHA file for SusHi
    - Run SusHi to get more accurate signal cross section numbers
    - Compute expected event numbers for all signal / bgnds
    
    With this we can
    - Generate test set
    - Generate all validation sets
    
    Arguments:
        args: argparse instance for controlling what to run 
        trainmodels: list of models to generate training sets for 
        testmodels: list of models to generate test and validation sets for 
    """
    
    starttime = time.time()
    
    
    # Run
    if args.train:
        submit_set('train', trainmodels, args.ncpu)
    
    if args.validation:
        submit_set('validation', trainmodels, args.ncpu)
    
    if args.test:
        submit_set('test', testmodels, args.ncpu)
    
    
    # Compute expected event numbers
    if args.print_expected:
        
        for model in testmodels:
            # Compute the signal cross section
            model.prepare_sushi_input()
        
            for sample in model.samples:
                
                # Compute expectation
                model.get_cross_section(sample)
                model.compute_expected_event_numbers(sample)



    
    print 'All done'
    endtime = time.time()
    print '\nTime elapsed: %s' % str(datetime.timedelta(seconds=endtime-starttime))



def parseargs():
    
    parser = ArgumentParser(description='Generate events!')
    parser.add_argument('-trn', '--train', help='Create train sets', action='store_true')
    parser.add_argument('-tst', '--test', help='Create test sets', action='store_true')
    parser.add_argument('-val', '--validation', help='Create validation sets', action='store_true')
    parser.add_argument('-pex', '--print_expected', help='Print expected event numbers', action='store_true')
    parser.add_argument('-nc', '--ncpu', type=int, help='Number of parallel jobs', default=4)
    pargs = parser.parse_args()
    
    return pargs


## ----------------------------------------------------------------------------- 
if __name__ == '__main__':
    
    
    # Common values for event generation
    GENERATION_SETTINGS = {
    
        # Number of training events
        'n_events' : {
            'H' : 2500000,
            'A' : 2500000,
            'Z' : 20,
            'ttbar': 100,
            'VV': 100,
        },
        
        # Split train sets for faster generation
        'n_splits' : 10,

        # Detector smearing
        'smearing' : False,
        
        # Number of validation sets
        'n_validation_samples' : 2  # TODO: obsolete
    }


    # The models to generate

    # H, A at 450 GeV
    HA_450GeV = Model(
    
        # 2HDM parameters
        param_dict={'mh': 125, 'mH': 450, 'mA': 450, 'mC': 450, 'sin_ba': 0.999,
                    'lambda_6': 0, 'lambda_7': 0, 'm12_2': 16000, 'tanb': 1},
        
        # Settings for event generation
        gen_settings_dict=GENERATION_SETTINGS,
        lumi=300.,
        massrange=[150, 500],
        #backgrounds=['Z', 'ttbar', 'VV'],
        backgrounds=[],
        outdir='450GeV',
        title='model'
    )
    
    
    
    margs = parseargs()
    run(margs, trainmodels=[HA_450GeV], testmodels=[HA_450GeV])


