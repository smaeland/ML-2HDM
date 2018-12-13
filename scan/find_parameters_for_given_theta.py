import csv
import numpy as np

def find_theta(resultsfile='results_ratio_scan_sorted.csv'):
    masses = []
    br_H = []
    br_A = []
    br_ratios = []
    xsec_H = []
    xsec_A = []
    xsec_ratios = []
    total_ratios = []
    tanbs = []
    m12s = []

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
            m12s.append(float(row[2]))
            xsec_H.append(float(row[4]))
            xsec_A.append(float(row[5]))
            xsec_ratios.append(float(row[6]))
            br_H.append(float(row[7]))
            br_A.append(float(row[8]))
            br_ratios.append(float(row[9]))
            total_ratios.append(float(row[10]))
            tanbs.append(float(row[3]))

    xsec_times_Br_H = np.array(br_H)*np.array(xsec_H)
    xsec_times_Br_A = np.array(br_A)*np.array(xsec_A)

    theta = xsec_times_Br_A/(xsec_times_Br_A+xsec_times_Br_H)
    
    def isclose(a, b):
        return abs(a-b) < 0.01

    mytheta = 0.9

    for i in range(len(theta)):
        
        if isclose(theta[i], mytheta):
            print 'theta = %.4f, tanb = %.4f, m12 = %.1f' % (theta[i], tanbs[i], m12s[i])
        
        #raw_input('cont')

if __name__ == '__main__':
    find_theta()
