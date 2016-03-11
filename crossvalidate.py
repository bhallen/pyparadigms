"""
Command-line interface for performing leave-one-out cross-validation on
grammars learned using Sublexical Morphology
"""

import paradigms

import argparse
import pickle

import pdb

#####################################################################
## Parse command line arguments                                    ##
#####################################################################
parser = argparse.ArgumentParser(description = \
         'Inflectional morphology predictor')
parser.add_argument('language_name',help='Name of target directory in datasets')
parser.add_argument('-b', '--bypass_learning_psublexicons', action='store_true')

args = parser.parse_args()

#####################################################################
## Main code                                                       ##
#####################################################################

# Read in cross-validation lexeme set
with open('datasets/{}/{}_cv_set.txt'.format(args.language_name, 
                                  args.language_name), 'r') as cv_set_file:
    cv_set_str = cv_set_file.read().strip()
    cv_set = ([lexeme for lexeme in cv_set_str.split('\n') 
              if len(lexeme) > 0])

# Initialize Lexicon first time, just to get the list of cells
lex = paradigms.Lexicon()

# Learn grammar and get a list of cells in the inflectional system
cells = lex.learn('datasets/{}/{}_training.txt'.format(args.language_name, 
                                               args.language_name), 
                  'datasets/{}/{}_constraints.txt'.format(args.language_name, 
                                               args.language_name),
                  'datasets/{}/{}_features.txt'.format(args.language_name, 
                                               args.language_name), 
                  just_return_cells=True)

with open('datasets/{}/{}_cv_results.txt'.format(args.language_name, 
                                   args.language_name), 'a') as cv_results_file:
    ## Write header now. Also useful for making sure multiple runs haven't been
    ## appended onto each other.
    cv_results_file.write('lexeme\tcell\tcandidate\tpredprob\n')


if args.bypass_learning_psublexicons:
    # Load grammarless lexicon from file
    with open('datasets/{}/{}_grammarless.lexicon'.format(
                      args.language_name, args.language_name), 'rb') as lexfile:
        lex = pickle.load(lexfile)

else:
    # Initialize Lexicon
    lex = paradigms.Lexicon()

    # Learn psublexicons but not their grammars
    lex.learn('datasets/{}/{}_training.txt'.format(args.language_name, 
                                                   args.language_name), 
              'datasets/{}/{}_constraints.txt'.format(args.language_name, 
                                                   args.language_name),
              'datasets/{}/{}_features.txt'.format(args.language_name, 
                                                   args.language_name),
              skip_grammars=True,
              pre_reduction_cutoff=50,
              post_reduction_cutoff=50,
              psublexicon_size_cutoff=20)

    # Save the grammar-less lexicon for quick loading later (to datasets/language_name/language_name_grammarless.lexicon)
    lex.save_lexicon(args.language_name, 'grammarless')

# Perform cross-validation
prediction_log = {}
for lexeme in cv_set:

    if lexeme in lex.lexemes: # should be true, but 'kom' is missing from Spanish for now...

        for cell in cells:
            print('To withhold: {}, {}'.format(lexeme, cell))
            
            # learn grammar for target deriv cell without held-out word
            print(lex.psublexicons.keys())
            for ps in lex.psublexicons[cell]:
                ps.learn_grammar('datasets/{}/{}_constraints.txt'.format(
                                                   args.language_name,
                                                   args.language_name), 
                                 word_to_withhold=(lexeme, cell))

            # Predict held-out word
            this_prediction = lex.predict_single_word((lexeme, cell))
            prediction_log[(lexeme, cell)] = this_prediction

            # print(this_prediction)
            # pdb.set_trace()

            # Write this cross-validation result to file
            with open('datasets/{}/{}_cv_results.txt'.format(args.language_name, 
                                       args.language_name), 'a') as cv_results_file:
                for candidate in this_prediction:
                    if this_prediction[candidate] > 0.0001:
                        cv_results_file.write('{}\t{}\t{}\t{}\n'.format(
                            lexeme, cell, candidate, this_prediction[candidate]))



### ORIGINAL VERSION: learned sublexicons again for every held-out word.

# with open('datasets/{}/{}_cv_results.txt'.format(args.language_name, 
#                                    args.language_name), 'a') as cv_results_file:
#     ## Write header now. Also useful for making sure multiple runs haven't been
#     ## appended onto each other.
#     cv_results_file.write('lexeme\tcell\tcandidate\tpredprob\n')

# prediction_log = {}
# for lexeme in cv_set:
#     for cell in cells:
#         print('To withhold: {}, {}'.format(lexeme, cell))
#         # Initialize Lexicon
#         lex = paradigms.Lexicon()

#         # Learn model
#         lex.learn('datasets/{}/{}_training.txt'.format(args.language_name, 
#                                                        args.language_name), 
#                   'datasets/{}/{}_constraints.txt'.format(args.language_name, 
#                                                        args.language_name),
#                   'datasets/{}/{}_features.txt'.format(args.language_name, 
#                                                        args.language_name),
#                   form_to_withhold=(lexeme, cell))

#         # Predict held-out word
#         this_prediction = lex.predict_single_word((lexeme, cell))
#         prediction_log[(lexeme, cell)] = this_prediction

#         # print(this_prediction)
#         # pdb.set_trace()

#         # Write this cross-validation result to file
#         with open('datasets/{}/{}_cv_results.txt'.format(args.language_name, 
#                                    args.language_name), 'a') as cv_results_file:
#             for candidate in this_prediction:
#                 if this_prediction[candidate] > 0.0001:
#                     cv_results_file.write('{}\t{}\t{}\t{}\n'.format(
#                         lexeme, cell, candidate, this_prediction[candidate]))

# print('Finished cross-validation!')
# pdb.set_trace()

