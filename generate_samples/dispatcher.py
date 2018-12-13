# pylint: disable=C0303, C0103

# Simplified batch manager


import multiprocessing
from collections import namedtuple
from time import sleep
from numpy import random
from pythia.X_to_tautau import generate_X_to_tautau


JobParams = namedtuple('JobParams', 
    ['cmndfile',        # path to Pythia command file
     'target_val',      # target (1,2,3)
     'outname',         # output name
     'nevents',         # number of events to generate
     'massrange',       # list with lower, upper inv mass range
     'smearing'         # boolean, apply smearing or not
     ])


def worker((cmnd_file, target_val, outname, nevents, mass_range, smearing)):
    
    # Wait for random time to avoid pythia start at the same time, hence using same random seed
    random.seed(hash(outname) % 10000)
    sleep(random.uniform(0.0, 3.0))

    print 'Generating %d events from %s' % (nevents, cmnd_file)
    generate_X_to_tautau(cmnd_file, target_val, nevents, outname, mass_range, smearing)



class Dispatcher(object):
    """ Manage a pool of workers """
    
    def __init__(self, n_simultaneous_jobs):
        """
        n_simultaneous_jobs: number of workers in pool
        """
        super(Dispatcher, self).__init__()
        
        self.joblist = []
        self.pool = multiprocessing.Pool(n_simultaneous_jobs)
        
    

    def submit(self, jobparams, split=False):
        """ Submit a job """
        
        if split:
            self.submit_split(jobparams, split)
        else:
            self.submit_single(jobparams)
    
    
    def submit_single(self, jobparams):
        """ Convert jobparams to list, append to job list """
        self.joblist.append(jobparams._asdict().values())
    
    
    def submit_split(self, jobparams, nsplit):
        """ Split a job into multiple and submit """
        
        assert isinstance(nsplit, int)
        
        neach = jobparams.nevents / nsplit
        remainder = jobparams.nevents % nsplit
        
        for i in range(nsplit):
            
            outputname = jobparams.outname.replace('.h5', '_{:03d}'.format(i))
            
            if i == 0:
                jp = jobparams._replace(nevents=(neach+remainder),
                                        outname=outputname)
            else:
                jp = jobparams._replace(nevents=neach, outname=outputname)
            
            self.submit_single(jp)
    
    
    
    def start(self):
        """ Start all submitted jobs """
        print 'Starting jobs'
        self.pool.imap(worker, self.joblist)
    
    
    def close(self):
        """ Close pool, then wait and join """
        print 'Closing pool'
        self.pool.close()
        self.pool.join()





