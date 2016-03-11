"""
Command-line interface for determining base informativity 
using Sublexical Morphology
"""

import paradigms

import argparse
import pickle

#####################################################################
## Parse command line arguments                                    ##
#####################################################################
parser = argparse.ArgumentParser(description = \
         'Inflectional morphology base informativity evaluator')
parser.add_argument('language_name',help='Name of target directory in datasets')

args = parser.parse_args()

#####################################################################
## Main code                                                       ##
#####################################################################

# Load lexicon with grammar into memory
with open('datasets/{}/{}.lexicon'.format(args.language_name, 
                                          args.language_name), 'rb') as lexfile:
    print('Loading the lexicon for {}...'.format(args.language_name))
    lex = pickle.load(lexfile)

	# Find privileged (most informative) base
    print('Evaluating base informativity...')
    results = lex.evaluate_base_informativity()
    
    for base in results:
        mean, percentiles = results[base]
        p25, p50, p75 = percentiles
        print('{}: {} ({}, {}, {})'.format(base, mean, p25, p50, p75))
