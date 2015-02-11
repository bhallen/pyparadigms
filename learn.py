#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import csv
import numpy as np
import scipy, scipy.sparse, scipy.optimize

import aligner
import hypothesize
import phoment
import gbr

PRE_REDUCTION_CUTOFF = None

# TRAINFILE = 'jp_paradigms.txt'
# CONFILE = 'jp_constraints.txt'
# FEATUREFILE = 'jp_features.txt'

# TRAINFILE = 'latin_paradigms.txt'
# CONFILE = 'latin_constraints.txt'
# FEATUREFILE = 'latin_features.txt'
# MAXALIGNMENTS = 2
# PRE_REDUCTION_CUTOFF = 8

TRAINFILE = 'kwerba_paradigms.txt'
CONFILE = 'kwerba_constraints.txt'
FEATUREFILE = 'kwerba_features.txt'


def normalize_exp(exp_probs, candidates):
    normalized = []
    lexeme = ''
    these_probs = []
    for p, c in zip(exp_probs, candidates):
        if c[0] == lexeme:
            these_probs.append(p)
        else:
            psum = sum(these_probs)
            normalized += [p/psum if psum > 0 else p for p in these_probs]
            lexeme = c[0]
            these_probs = [p]
    normalized += [p/psum if psum > 0 else p for p in these_probs]
    return np.array(normalized)

def calc_base_regularization(labeled_weights):
    result = 0
    bases = set([label[0] for weight, label in labeled_weights])
    for base in bases:
        these_weights = [weight for weight, label in labeled_weights if label[0] == base]
        result += sum([abs(w1-w2) for w1, w2 in itertools.combinations(these_weights, 2)])
    return result

def objective(weights, weight_labels, deriv_data, l1_mult=0.01, l2_mult=0.01, base_reg_mult=0.01):
    norms_by_deriv = []
    labeled_weights = list(zip(weights, weight_labels))
    for d in deriv_data:
        these_weights = [w for w, label in labeled_weights if label[1] == d and label[0] in deriv_data[d]['bases']]
        exp_probs = scipy.dot(deriv_data[d]['predicted'], these_weights)
        norms_by_deriv.append(np.linalg.norm(exp_probs-deriv_data[d]['observed']))

        l1l2_terms = l1_mult*sum(weights) + l2_mult*sum([w**2 for w in weights])
        base_reg_term = base_reg_mult*calc_base_regularization(labeled_weights)

    return sum(norms_by_deriv) + l1l2_terms + base_reg_term



## Initialize aligner and lexicon
alr = aligner.Aligner(feature_file=FEATUREFILE, sub_penalty=4.0, tolerance=1.0)

lex = gbr.Lexicon(TRAINFILE)

## Create list of *OBSERVED* MPS tuples
lex.create_cells()

## Create list of all features (bases or base-derivative mappings) to be given weights by GBR
lex.create_gbr_features()

sll_inputs = {}
for bf,df in lex.gbr_features:
    one_sll_input = []
    for base_entry in lex.select_subset(bf):
        for derivative_entry in lex.select_subset(df):
            if base_entry['lexeme'] == derivative_entry['lexeme']:
                one_sll_input.append((base_entry['lexeme'], base_entry['form'], derivative_entry['form']))
    sll_inputs[(bf,df)] = one_sll_input


## Create tableaux corresponding to each cell-to-cell mapping and organize them
full_data = {deriv_cell:{} for deriv_cell in lex.cells}
for one_mapping in sll_inputs:
    print()
    print(one_mapping)
    all_alignments = []
    for triple in sll_inputs[one_mapping]:
        lexeme = triple[0]
        probability = 1.0 # add support for reading probabilities in from the inputs (rather than assigning all observed forms 1.0)
        alignments = []
        for alignment, score in alr.align(triple[1].split(' '), triple[2].split(' ')):
            final_score = score/alr.check_cohesion(alignment)/len(alignment) # should this really divide by the length of the alignment?
            alignments.append({'alignment':alignment, 'probability': 1.0, 'lexeme': lexeme, 'score': final_score})
        alignments.sort(key=lambda x: x['score'])
        alignments.reverse()
        # print(len(alignments))

        try:
            all_alignments += alignments[:MAXALIGNMENTS] # skim off alignments with bad scores
        except NameError:

            all_alignments += alignments
    print('Number of alignments: {}'.format(str(len(all_alignments))))

    reduced_hypotheses = hypothesize.create_and_reduce_hypotheses(all_alignments, PRE_REDUCTION_CUTOFF)
    print('Hypotheses have been reduced.')

    sublexicons = hypothesize.add_zero_probability_forms(reduced_hypotheses)
    print('Zero probability forms added. # of sublexicons: {}'.format(str(len(sublexicons))))

    # for s in sublexicons:
    #     print(s)
    #     print(s.associated_forms)

    with open(CONFILE) as con_file:
        conreader = csv.reader(con_file, delimiter='\t')
        constraints = [c[0] for c in conreader if len(c) > 0]

    print('\n')
    print('Mapping: {}'.format(str(one_mapping)))
    sublexicons, megatableaux = zip(*[hypothesize.add_grammar(s, constraints) for s in sublexicons])

    mapping_tableau = hypothesize.create_mapping_tableau(sublexicons, megatableaux)

    for lexeme in mapping_tableau:
        for candidate in mapping_tableau[lexeme]:
            if lexeme in full_data[one_mapping[1]]:
                if candidate in full_data[one_mapping[1]][lexeme]:
                    full_data[one_mapping[1]][lexeme][candidate][one_mapping[0]] = mapping_tableau[lexeme][candidate]
                else:
                    full_data[one_mapping[1]][lexeme][candidate] = {one_mapping[0]: mapping_tableau[lexeme][candidate]}
            else:
                full_data[one_mapping[1]][lexeme] = {candidate: {one_mapping[0]: mapping_tableau[lexeme][candidate]}}



## Create a set of matrices containing predicted probabilities, one for each derivative cell. Also make a list of observed probabilities (or, for now, just observed/unobserved)
print('Creating prediction matrices.')
deriv_data = {}
for deriv_cell in full_data:
    this_deriv_matrix = []
    this_deriv_obs = []
    this_deriv_bases = [b for b in lex.cells if b != deriv_cell]
    this_deriv_candidates = []
    with open('{}.txt'.format(str(deriv_cell)), 'w') as outf:
        outf.write('lexeme\tform\tobserved\t'+'\t'.join([str(b) for b in this_deriv_bases])+'\n')
        for lexeme in full_data[deriv_cell]:
            for candidate in full_data[deriv_cell][lexeme]:
                this_deriv_candidates.append((lexeme, candidate))
                # get observed probability
                row = ['{}\t{}'.format(str(lexeme), candidate)]
                obs_or_not = lex.cells[deriv_cell][lexeme] == candidate
                this_deriv_obs.append(float(obs_or_not))
                row.append(str(float(obs_or_not)))
                # get predicted probabilities
                available_mappings = []
                predicted_probs = []
                for base in this_deriv_bases:
                    predicted_probs.append(full_data[deriv_cell][lexeme][candidate].get(base, 0.0))
                this_deriv_matrix.append(predicted_probs)
                row += [str(fl) for fl in predicted_probs]
                outf.write('\t'.join(row)+'\n')
    deriv_data[deriv_cell] = {'observed': np.array(this_deriv_obs), 'predicted': np.array(this_deriv_matrix), 'bases': this_deriv_bases, 'candidates': this_deriv_candidates}


## Learn weights
posReals = [(0,25) for wt in range(len(lex.gbr_features))]

print('Learning mapping weights.')
output_weights, nfeval, return_code = scipy.optimize.fmin_l_bfgs_b( 
        objective, scipy.rand(len(lex.gbr_features)), 
        args=(lex.gbr_features, deriv_data, 0.0, 0.01, 0.0),
        bounds=posReals,
        approx_grad=True)
        # maxfun=10)
        # factr=100)


# ## Print model predictions
# print('\n\nPredictions:\n')
# labeled_output_weights = list(zip(output_weights, lex.gbr_features))
# for d in deriv_data:
#     print(d)
#     these_weights = [w for w, label in labeled_output_weights if label[1] == d and label[0] in deriv_data[d]['bases']]
#     exp_probs = scipy.dot(deriv_data[d]['predicted'], these_weights)
#     normalized_exp_probs = normalize_exp(exp_probs, deriv_data[d]['candidates'])
#     this_lexeme = ''
#     for c, o, e in zip(deriv_data[d]['candidates'], deriv_data[d]['observed'], normalized_exp_probs):
#         if c[0] != this_lexeme:
#             print('')
#             this_lexeme = c[0]
#         print('{}\t{}\t\t{}\t{}'.format(c[0], c[1], str(o), str(e)))
#     print('\n')


## Print weights
print('\n\nWeights:')
labeled_weights = list(zip(output_weights, lex.gbr_features))
for d in deriv_data:
    print('\nDerivative cell: {}'.format(str(d)))
    for w, label in labeled_weights[:]:
        if label[1] == d and label[0] in deriv_data[d]['bases']:
            print('{}: {}'.format(str(label[0]), str(w)))


## L1 regularization is non-convex. Paul's term would also be non-convex, so try squaring it.

## Paul's regularization: similar to a hierarchical model? maybe equivalent?
## hierarchical is basically having weights on bases plus mapping weights: Paul's regularization is SUM(i: SUM(A: SUM(B: |w_A - wbar_A|)))))

## Derivative/gradient of obs/exp term: 2*SUM(i: w*exp-obs)*exp
# where exp is the predicted probability for a particular output candidate, and w is the b-d weight
# ~ 2a(ax-b)

# useful trick: calculate derivative numerically, see if it converges to the expected value

# try adding cross-validation!


## Convexity: given any two points, the line between them lies above the curve. This implies a unique minimum
## Strictly convex is the above for real; convex but not strict convex would be, e.g. a curve with a flat bottom
## Cf. "smooth": no corners. Orthagonal to convexness
# one solution to non-smoothness is to "smooth out the corners". Instead of |x|, minimize |x|+epsilon*x^2

## Srsly, read more of the Hastie & Tibshirani textbook on Statistical Learning. Just do it.


## old version (n^2-n features/weights) is mathematically equivalent to new version with base weights (n^2 features) according to the
# proof that Paul wrote down. But the dof of the new version are the same as the old, because the additional variables have to satisfy
# n new constraints, which is that all the deviations (new version of old constraints, mapping constraints) from the mean (new: base weights)
# have to be zero.

# So... it may be best to explain it using the Vs but actually do the optimization using Ws since they're unconstrained and easier to find.
# I would need to add one more line of code that just changes the n^2-2 variables into n^2 variables to be reported by finding their means
# and variances for each base.