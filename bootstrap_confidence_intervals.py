# Calculates confidence intervals around the mean using the bootstrap

# open file w/jaccard

# sample w/replacement N groups of M
# average each sample
# average the averages

# calculate sqrt(1/M-1 sum( (each sample mean - avg_avg)^2))
# traditionally, the 'whole sample mean' is the average of the estimates
# according to https://thestatsgeek.com/2013/07/02/the-miracle-of-the-bootstrap/

import pandas as pd
import numpy as np
from scipy.stats import norm

from random import gauss
import os, datetime
import sys
import unittest
import time
import argparse

class TestBootstrapCI(unittest.TestCase):
    col = pd.Series([gauss(0,1) for i in range(1000)])

    def test_bootstrap_sterr_of_mean(self):
        for j in range(10):
            m, s, mm, ss = bootstrap_sterr_of_mean(self.col, 500, 1000)
            self.assertTrue(np.abs(mm) < 0.1, msg=mm)  # crude test of function
        
    def test_report_CI(self):
        m, s, mm, ss = bootstrap_sterr_of_mean(self.col, 500, 1000)
        lo, hi = report_CI(mm, ss, 0.05, False)
        self.assertTrue(mm > lo, msg='{} {}'.format(lo, mm))
        self.assertTrue(mm < hi, msg = '{} {}'.format(hi, mm))
        
        
    def comp_series(self, ser1, ser2, eps=1e-10):
        return ((ser1-ser2) < eps).all()
        
        
    def test_read_input(self):
        now = str(datetime.datetime.now())
        f = now.replace(' ',"_").replace(':','_').replace('.','_')
        
        to_remove = []
        
        # without index col or col name
        self.col.to_csv(f, index=False)
        to_remove.append(f)
        read1 = read_input(f)
        self.assertTrue(self.comp_series(self.col, read1))
        
        # with index col and col name
        self.col.name = 'blah'
        f = f + 'a'
        self.col.to_csv(f)
        to_remove.append(f)
        read2 = read_input(f, col_name='blah')
        self.assertTrue(self.comp_series(self.col, read2))
        
        # without header
        f = f+'a'
        self.col.to_csv(f, index=False, header=None)
        to_remove.append(f)
        read3 = read_input(f, header=None)
        self.assertTrue(self.comp_series(self.col, read3))
        
        for i in to_remove:
            os.remove(i)


def bootstrap_sterr_of_mean(col, subgroup_size=None, n_subgroups=1000):
    # Calculate the mean and sterr of a pandas.Series
    # returns (global mean, sterr est using global mean, est mean, sterr est using est mean)
    # using the estimated mean values is standard practice 
    mu_bar = col.mean()
    mus = []
    if not subgroup_size:
        subgroup_size = col.shape[0]

    for i in range(n_subgroups):
        sample = col.sample(subgroup_size, replace=True)
        mus.append(sample.mean())
    mus = pd.Series(mus)
    st_err_glob = np.sqrt(1/(n_subgroups-1) * (mus - mu_bar).pow(2).sum())

    avg_avg = mus.mean()
    st_err = np.sqrt(1/(n_subgroups-1) * (mus - avg_avg).pow(2).sum())
    
    return mu_bar, st_err_glob, avg_avg, st_err
    
    
def parse_args(args=sys.argv[:1]):
    parser = argparse.ArgumentParser(
        description='calculate the bootstrap mean, N groups of M')
    parser.add_argument("input", 
        help='csv with data in columns')
    parser.add_argument("--col_name", 
        help='name of the column to analyze. If unspecified uses the first column', 
        default='')
    parser.add_argument('--no_header', 
        help='input does not have column names as first row', action='store_true')
    parser.add_argument('-M', '--subgroup_size', type=int,
        help='bootstrap subsample size', default=None)
    parser.add_argument('-N', '--n_subgroups', type=int,
        help='the number of subgroups to use', default=1000)
    parser.add_argument('--alpha', default=0.05, type=float,
        help='alpha to use for reporting')
    return parser.parse_args(args)
    
    
def report_CI(mu, sterr, alpha, printout=True):
    # calculate and print a 2-sided confidence interval given a mean, sterr and alpha level
    # returns a tuple-CI

    s = norm.ppf(1-alpha/2)  # how many mus to get to 1-alpha/2 significance (1.96 for alpha=0.05)
    # assumes two sided confidence intervals

    if printout:
        print('mu±sterr: {:.4f}±{:.4f}\n CI: {:.4f} {:.4f}'.format(
            mu, sterr, mu - sterr*s, mu + sterr*s))
    return mu - sterr*s, mu + sterr*s


def read_input(f, header='infer', col_name=''):
    df = pd.read_csv(f, header=header, index_col=False)
    if col_name:
        col = df[col_name]
    else:
        col = df.iloc[:,0]
    return col


def main():
    args = parse_args(sys.argv[1:])
    
    subgroup_size = args.subgroup_size  # M
    n_subgroups = args.n_subgroups # B, N
    if args.no_header:
        header=None
    else:
        header='infer'
    col = read_input(args.input, header, args.col_name)

    mu_bar, st_err_glob, avg_avg, st_err = bootstrap_sterr_of_mean(col, subgroup_size, n_subgroups)
    
    if not subgroup_size:
        subgroup_size = col.shape[0]
    
    print("{}, {} groups of {}, alpha {}".format(args.input,n_subgroups,subgroup_size, args.alpha))
    #print("\nusing global mu")
    #report_CI(mu_bar, st_err_glob, args.alpha)
    #print("\ntraditional")  # use only the traditional formula
    report_CI(avg_avg, st_err, args.alpha)
    

if __name__=="__main__":
    main()
