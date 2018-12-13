# pylint: disable=C0103,C0303 

# Scan the parameter space for valid points, and for each valid point, compute
# (xsec H * BR(H->tau) / (xsec A * BR(A->tau) 

import os
import numpy as np
from argparse import ArgumentParser
import pickle
import multiprocessing
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import csv
import time, datetime
from glob import glob
from generate_samples.lhatool import LHA, Block, Entry


sushi_binary = os.environ['SUSHIPATH']


def write_sushi_input_files(lhafile):
    """ Add SusHi-related blocks to LHA file """ 
    
    outfiles = {}
    
    for higgsname, higgstype in {'H': 12, 'A': 21}.iteritems():
        
        lha = LHA(lhafile)
        
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

        thdm = Block('2HDM', '2HDM parameters')
        #thdm.add(Entry([1], comment='Type I'))
        #thdm.add(Entry([2], comment='Type II'))
        thdm.add(Entry([4], comment='Type IV'))
        lha.add_block(thdm)

        distrib = Block('DISTRIB', comment='Kinematic requirements')
        distrib.add(Entry([1, 0], comment='Sigma total'))
        distrib.add(Entry([2, 0], comment='Disable pT cut'))
        #distrib.add(Entry([21, GENER_SETTINGS['higgs_pt_min']], comment='Min higgs pT'))
        distrib.add(Entry([3, 0], comment='Disable eta cut'))
        #distrib.add(Entry([32, GENER_SETTINGS['higgs_eta_max']], comment='Max eta'))
        distrib.add(Entry([4, 1], comment='Use eta, not y'))
        lha.add_block(distrib)

        pdfspec = Block('PDFSPEC')
        pdfspec.add(Entry([1, 'MMHT2014lo68cl.LHgrid'], comment='Name of pdf (lo)'))
        pdfspec.add(Entry([2, 'MMHT2014nlo68cl.LHgrid'], comment='Name of pdf (nlo)'))
        pdfspec.add(Entry([3, 'MMHT2014nnlo68cl.LHgrid'], comment='Name of pdf (nnlo)'))
        pdfspec.add(Entry([4, 'MMHT2014nnlo68cl.LHgrid'], comment='Name of pdf (n3lo)'))
        pdfspec.add(Entry([10, 0], comment='Set number'))
        lha.add_block(pdfspec)

        lha.get_block('SMINPUTS').add(Entry([8, 1.275], comment='m_c'))

        # Write output
        suffix = '_%s_sushi.in' % higgsname
        outname = lhafile.replace('.lha', suffix)

        lha.write(outname)
        
        outfiles[higgsname] = outname
    
    return outfiles




def worker(mass, tanbeta, q):
    """ Compute cross section * BR for this point  """ 
    
    result = ''
 
    # Prevent crash from destroying the entire scan
    try:

        br_tautau = {}
        xsec = {}
    
        # Run 2HDMC (which only allows writing directly to file)
        try:
            with open(os.devnull, 'w') as DEVNULL:
                output = subprocess.check_output(['./ratio_scanner', str(mass), str(mass), str(tanbeta)])
        except subprocess.CalledProcessError:
            # Invalid point
            return result

             
        # Get file name from output. Open and parse it, get BRs
        lhafilename = output.split('\n')[-2]
        lha = LHA(lhafilename)
        
        br_tautau['H'] = float(lha.get_decay(35).get_branching_ratio(15, -15))
        br_tautau['A'] = float(lha.get_decay(36).get_branching_ratio(15, -15))
        m12_2 = float(lha.get_block('MINPAR').get_entry_by_key(18))
        
        del lha
        
        # Now write SusHi input files
        sushi_inputs = write_sushi_input_files(lhafilename)
        
        # Run SusHi for both A and H
        for higgsname, infile in sushi_inputs.iteritems():
            
            outfile = infile.replace('.in', '.out')
            with open(os.devnull, 'w') as DEVNULL:
                subprocess.check_call([sushi_binary, infile, outfile], stdout=DEVNULL)
            
            # Get cross sections
            lha = LHA(outfile)
            xsec[higgsname] = float(lha.get_block('SUSHIggh').get_entry_by_key(1))
            
            # Remove files
            os.remove(infile)
            os.remove(outfile)
        
        os.remove(lhafilename)
        
        # Output 
        # mH mA m12_2 tanb xsec-H xsec-A xsec-ratio BR-H BR-A BR-ratio tot-ratio
        res = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            mass, mass, m12_2, tanbeta,
            xsec['H'], xsec['A'], (xsec['H']/xsec['A']),
            br_tautau['H'], br_tautau['A'], (br_tautau['H']/br_tautau['A']),
            (xsec['H']/xsec['A'])*(br_tautau['H']/br_tautau['A'])
        )
        
        result += res

    except:
        return result
        
    
    # Clean up remaining lha files
    try:
        for fs in glob('massH_%d_massA_%d_tanb_%f*' % (int(mass), int(mass), tanbeta)):
            os.remove(fs)
    except:
        pass

    # Write result to q
    q.put(result)
    return result
        


def listener(q):
    """ Listen for messages broadcasted on 'q', write to file. """

    f = open(FILENAME, 'wb')
    f.write('mH,mA,m12_2,tanb,xsec-H,xsec-A,xsec-ratio,BR-H,BR-A,BR-ratio,tot-ratio\n')
    while True:
        m = q.get()
        if m == 'stop':
            break
        f.write(str(m))
        f.flush()
    f.close()
    
    


def scan(ncpu):
    """ Run the scan """
    
    starttime = time.time()

    # Number of points to compute
    npoints = 10000
    #npoints = 100

    # Ranges to consider 
    allowed_mass_range = [360, 800]
    allowed_tanb_range = [0.1, 60]

    # Job manager
    manager = multiprocessing.Manager()
    q = manager.Queue()
    pool = multiprocessing.Pool(ncpu)
    
    # Start the listener 
    watcher = pool.apply_async(listener, (q,))
    
    # Submit jobs
    jobs = []
    for i in xrange(npoints):

        # Draw mass and tanb values
        mass = np.random.uniform(allowed_mass_range[0], allowed_mass_range[1])
        tanb = np.random.exponential(scale=10)
        while tanb < allowed_tanb_range[0] or tanb > allowed_tanb_range[1]:
            tanb = np.random.exponential(scale=10)

        job = pool.apply_async(worker, (mass, tanb, q))
        jobs.append(job)
        
    # Collect results from the workers through the pool result queue
    for job in jobs:
        job.get()
    

    # Stop the listener
    q.put('stop')
    pool.close()
    
    print 'Done.'
    endtime = time.time()
    print '\nTime elapsed for %d points: %s' % (npoints, str(datetime.timedelta(seconds=endtime-starttime)))

    
    
def plot(resultsfile):
    """ Plot results from scan """
    
    plt.style.use('../plot/paper.mplstyle')
    
    masses = []
    br_H = []
    br_A = []
    br_ratios = []
    xsec_H = []
    xsec_A = []
    xsec_ratios = []
    total_ratios = []
    tanbs = []

    #  0, 1,    2,   3,     4,     5,         6,   7,   8,       9,       10
    # mH,mA,m12_2,tanb,xsec-H,xsec-A,xsec-ratio,BR-H,BR-A,BR-ratio,tot-ratio

    
    with open(resultsfile, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not len(row):
                continue
            if row[0] == 'mH':
                continue    # Header
            masses.append(float(row[0]))
            xsec_H.append(float(row[4]))
            xsec_A.append(float(row[5]))
            xsec_ratios.append(float(row[6]))
            br_H.append(float(row[7]))
            br_A.append(float(row[8]))
            br_ratios.append(float(row[9]))
            total_ratios.append(float(row[10]))
            tanbs.append(float(row[3]))
    
    # Set up colormap and colorbar
    colormap = plt.cm.get_cmap('viridis')
    formatter = matplotlib.ticker.LogFormatter(10, labelOnlyBase=False, minor_thresholds=(100,1))
    
    """
    # Plot cross section ratio
    fig = plt.figure()
    sc = plt.scatter(masses, xsec_ratios, c=tanbs, vmin=0.5, vmax=60, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    plt.xlabel(r'$m_{A/H}$ (GeV)')
    plt.ylabel(r'$\frac{\sigma(gg\rightarrow H)}{\sigma(gg\rightarrow A)}$')
    fig.show()
    """

    """
    # Plot BR ratio
    fig = plt.figure()
    sc = plt.scatter(masses, br_ratios, c=tanbs, vmin=0.5, vmax=60, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    plt.xlabel(r'$m_{A/H}$ (GeV)')
    plt.ylabel(r'$\frac{\mathcal{B}(H\rightarrow \tau\tau)}{\mathcal{B}(A\rightarrow \tau\tau)}$')
    fig.show()
    """

    """
    # Plot total ratio, H/A
    fig = plt.figure()
    sc = plt.scatter(masses, total_ratios, c=tanbs, vmin=0.5, vmax=60, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    plt.xlabel(r'$m_{A/H}$ (GeV)')
    plt.ylabel(r'$\frac{\sigma(gg\rightarrow H)\times \mathcal{B}(H\rightarrow \tau\tau)}{\sigma(gg\rightarrow A)\times \mathcal{B}(A\rightarrow \tau\tau)}$')
    fig.show()
    """
    
    
  

    # Plot xsec*BR for H
    fig = plt.figure()
    xsec_times_Br_H = np.array(br_H)*np.array(xsec_H)
    sc = plt.scatter(masses, xsec_times_Br_H, c=tanbs, vmin=0.5, vmax=60, s=8, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    plt.yscale('log')
    plt.ylim(1.0e-7, 1)
    plt.xlabel(r'$m_{H}$ [GeV]')
    plt.ylabel(r'$\sigma(gg/b\bar{b}\rightarrow H)\times \mathcal{B}(H\rightarrow \tau\tau)$ [pb]')
    fig.show()
    plt.tight_layout()
    fig.savefig('xsec_vs_mass_H.pdf')
    
    # Plot xsec*BR for A
    fig = plt.figure()
    xsec_times_Br_A = np.array(br_A)*np.array(xsec_A)
    sc = plt.scatter(masses, xsec_times_Br_A, c=tanbs, vmin=0.5, vmax=60, s=6, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    plt.yscale('log')
    plt.ylim(1.0e-7, 1)
    plt.xlabel(r'$m_{A}$ [GeV]')
    plt.ylabel(r'$\sigma(gg/b\bar{b}\rightarrow A)\times \mathcal{B}(A\rightarrow \tau\tau)$ [pb]')
    fig.show()
    plt.tight_layout()
    fig.savefig('xsec_vs_mass_A.pdf')
   
    """
    # Plot xsec*BR for both
    fig = plt.figure()
    sc = plt.scatter(masses, np.array(br_H)*np.array(xsec_H)+np.array(br_A)*np.array(xsec_A), c=tanbs, vmin=0.5, vmax=60, s=20, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    plt.yscale('log')
    plt.ylim(1.0e-9, 1)
    plt.xlabel(r'$m_{A/H}$ (GeV)')
    plt.ylabel(r'$\sigma(gg\rightarrow A)\times \mathcal{B}(A\rightarrow \tau\tau)$ [pb]')
    fig.show()
    """
   
    """
    # Plot total ratio, A/H
    fig = plt.figure()
    total_ratio_A_over_H = xsec_times_Br_A/xsec_times_Br_H
    sc = plt.scatter(masses, total_ratio_A_over_H, c=tanbs, vmin=0.5, vmax=60, s=8, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    #plt.yscale('log')
    plt.ylim(0, 10)
    plt.xlabel(r'$m_{A/H}$ (GeV)')
    plt.ylabel(r'$\frac{\sigma(gg\rightarrow A)\times \mathcal{B}(A\rightarrow \tau\tau)}{\sigma(gg\rightarrow H)\times \mathcal{B}(H\rightarrow \tau\tau)}$')
    fig.show()
    plt.tight_layout()
    """
    
    # Plot theta = nA/(nA+nH)
    fig = plt.figure()
    theta = xsec_times_Br_A/(xsec_times_Br_A+xsec_times_Br_H)
    sc = plt.scatter(masses, theta, c=tanbs, vmin=0.5, vmax=60, s=8, cmap=colormap, norm=matplotlib.colors.LogNorm())
    colorbar = plt.colorbar(ticks=[0.5, 1, 5, 10, 20, 30, 40, 50, 60], format=formatter)
    colorbar.set_label(r'$\tan\beta$')
    #plt.yscale('log')
    plt.ylim(0.335, 1.025)
    plt.xlabel(r'$m_{A/H}$ [GeV]')
    plt.ylabel(r'$\alpha$')
    fig.show()
    plt.tight_layout()
    fig.savefig('alpha_vs_mass.pdf')
    
    plt.show()
    
    
if __name__ == '__main__':
    
    parser = ArgumentParser(description='Scan H/A ratio vs mass')
    parser.add_argument('-s', '--scan', help='Run scan', action='store_true')
    parser.add_argument('-p', '--plot', help='Create plot', action='store_true')
    parser.add_argument('-nc', '--ncpu', type=int, help='Number of parallel jobs', default=8)
    parser.add_argument('-f', '--filename', type=str, help='Input/output file', default='results_ratio_scan.csv')
    pargs = parser.parse_args()
    
    FILENAME = pargs.filename

    if pargs.scan:
        if os.path.exists(FILENAME):
            owr = raw_input('Output file %s exists -- overwrite (y/n)? ' % FILENAME)
            if owr.lower().replace('\n', '') != ('y'):
                print 'exit'
                exit()
        scan(pargs.ncpu)
        
    if pargs.plot:
        plot(FILENAME)
    
