"""
Command-line interface for using a grammar to predict novel forms 
using Sublexical Morphology
"""

import paradigms

import argparse
import pickle

#####################################################################
## Parse command line arguments                                    ##
#####################################################################
parser = argparse.ArgumentParser(description = \
         'Inflectional morphology predictor')
parser.add_argument('language_name',help='Name of target directory in datasets')

args = parser.parse_args()

#####################################################################
## Main code                                                       ##
#####################################################################

# Load lexicon with grammar into memory
with open('datasets/{}/{}.lexicon'.format(args.language_name, 
                                          args.language_name), 'rb') as lexfile:
    lex = pickle.load(lexfile)

    # Make predictions
    lex.predict(args.language_name)

    # Save predictions to file
    lex.save_predictions(args.language_name)
