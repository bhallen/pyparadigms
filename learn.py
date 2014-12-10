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


alr = aligner.Aligner(feature_file='latin_features.txt', sub_penalty=4.0, tolerance=1.0)

lex = gbr.Lexicon('example_paradigms.txt')

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
new_tab = {(lm,cl):{} for lm in lex.lexemes for cl in lex.cells}
for one_mapping in sll_inputs:
    print()
    print(one_mapping)
    all_alignments = []
    for pair in sll_inputs[one_mapping]:
        alignments = []
        for alignment in alr.align(pair[1].split(' '), pair[2].split(' ')):
            alignments.append(alignment+[1.0]) # add support for reading probabilities in from the inputs (rather than assigning all observed forms 1.0)

        scored_alignments = []
        for a in alignments:
            final_score = a[1]/alr.check_cohesion(a[0])/len(a[0]) # should this really divide by the length of the alignment?
            scored_alignments.append((a[0], final_score, a[2]))
        scored_alignments.sort(key=lambda x: x[1])
        scored_alignments.reverse()

        selected_alignments = [(a[0], a[2]) for a in scored_alignments]  # TO-DO: add ability to skim off only best scoring alignments

        all_alignments += selected_alignments # TO-DO: add ability to skim off only best scoring alignments

    
    reduced_hypotheses = hypothesize.create_and_reduce_hypotheses(all_alignments)

    sublexicons = hypothesize.add_zero_probability_forms(reduced_hypotheses)

    # for s in sublexicons:
    #     print(s)
    #     print(s.associated_forms)

    with open('example_constraints.txt') as con_file:
        conreader = csv.reader(con_file, delimiter='\t')
        constraints = [c[0] for c in conreader if len(c) > 0]

    sublexicons, megatableaux = zip(*[hypothesize.add_grammar(s, constraints) for s in sublexicons])

    mapping_tableau = hypothesize.create_mapping_tableau(sublexicons, megatableaux)
    for fm in mapping_tableau:
        lexeme = lex.retrieve_lexeme(fm, one_mapping[0])  # TEMPORARY -- final version must not look up lexeme, but rather maintain it from the beginning, to ward off homophony issues
        for candidate in mapping_tableau[fm]:
            if candidate in new_tab[(lexeme,one_mapping[1])]:
                new_tab[(lexeme,one_mapping[1])][candidate][one_mapping[0]] = mapping_tableau[fm][candidate]
            else:
                new_tab[(lexeme,one_mapping[1])][candidate] = {one_mapping[0]: mapping_tableau[fm][candidate]}



## Create a matrix of predicted probabilities to pass to the optimization function; also make a file with the table. Also make a list of observed probabilities (or, for now, just observed/unobserved)
with open('output.txt','w') as outf:
    outf.write('meaning\tform\tobserved\t'+'\t'.join(m for m in [str(f) for f in lex.gbr_features])+'\n')
    A = []
    obs = []
    for lmc in new_tab:
        print()
        print(lmc)
        for candidate in new_tab[lmc]:
            # get observed probability
            row = ['{}\t{}'.format(str(lmc),candidate)]
            obs_or_not = lex.cells[lmc[1]][lmc[0]] == candidate
            obs.append(float(obs_or_not))
            row.append(str(float(obs_or_not)))
            # get predicted probabilities
            predicted_probs = []
            for mapping in lex.gbr_features:
                if mapping[1] == lmc[1] and mapping[0] in new_tab[lmc][candidate]:
                    predicted_probs.append(new_tab[lmc][candidate][mapping[0]])
                else:
                    predicted_probs.append(0.0)
            A.append(predicted_probs)
            row += [str(fl) for fl in predicted_probs]
            outf.write('\t'.join(row)+'\n')

print(lex.cells)

## Learn weights
A = np.array(A)
posReals = [(0,25) for wt in range(len(A[0]))]

def objective(wts, cond_prob_matrix, obs_probs, l1_mult=0.0, l2_mult=0.0):
    exp_probs = scipy.dot(cond_prob_matrix, wts)
    return np.linalg.norm(exp_probs-obs_probs) + l1_mult*sum(wts) + l2_mult*sum([w**2 for w in wts])

con_weights, nfeval, return_code = scipy.optimize.fmin_l_bfgs_b( 
        objective, scipy.rand(len(A[0])), 
        args=(A,obs,0.1,0.1),
        bounds=posReals,
        approx_grad=True)

print('Weights:')
for o,w in zip(lex.gbr_features,con_weights):
    print(o)
    print(w)


