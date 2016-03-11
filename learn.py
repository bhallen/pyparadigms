"""
Command-line interface for learning a grammar using Sublexical Morphology
"""

import paradigms

import argparse


#####################################################################
## Parse command line arguments                                    ##
#####################################################################
parser = argparse.ArgumentParser(description = \
         'Inflectional morphology learner')
parser.add_argument('language_name',help='Name of target directory in datasets')
parser.add_argument('-a', '--max_alignments', help=('Maximum number of'
                                'alignments for a given base--derivative pair'))

args = parser.parse_args()

#####################################################################
## Main code                                                       ##
#####################################################################

# Initialize Lexicon
lex = paradigms.Lexicon()

# Learn grammar
lex.learn('datasets/{}/{}_training.txt'.format(args.language_name, 
                                               args.language_name), 
          'datasets/{}/{}_constraints.txt'.format(args.language_name, 
                                               args.language_name),
          'datasets/{}/{}_features.txt'.format(args.language_name, 
                                               args.language_name),
          pre_reduction_cutoff=100,
          post_reduction_cutoff=100,
          psublexicon_size_cutoff=35)

# Save grammarized Lexicon to datasets/args.language_name
lex.save_lexicon(args.language_name)
