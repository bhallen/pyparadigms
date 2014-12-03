#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import csv

import aligner
import hypothesize
import phoment
import gbr


alr = aligner.Aligner(feature_file='latin_features.txt', sub_penalty=4.0, tolerance=1.0)

lex = gbr.Lexicon('latin_tiny.txt')

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

new_tab = {(lm,cl):{} for lm in lex.lexemes for cl in lex.cells}
for one_mapping in sll_inputs:
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

    with open('latin_constraints.txt') as con_file:
        conreader = csv.reader(con_file, delimiter='\t')
        constraints = [c[0] for c in conreader if len(c) > 0]

    sublexicons, megatableaux = zip(*[hypothesize.add_grammar(s, constraints) for s in sublexicons])

    mapping_tableau = hypothesize.create_mapping_tableau(sublexicons, megatableaux)
    for fm in mapping_tableau:
        lexeme = lex.retrieve_lexeme(fm, one_mapping[0])  # TEMPORARY -- final version must not look up lexeme, but rather maintain it throughout
        for candidate in mapping_tableau[fm]:
            if candidate in new_tab[(lexeme,one_mapping[0])]:
                new_tab[(lexeme,one_mapping[0])][candidate][one_mapping[1]] = mapping_tableau[fm][candidate]
            else:
                new_tab[(lexeme,one_mapping[0])][candidate] = {one_mapping[1]: mapping_tableau[fm][candidate]}

with open('abc.txt','w') as outf:
    outf.write('\t\t'+'\t'.join(m for m in [str(f) for f in lex.gbr_features])+'\n')
    A = []
    for lmc in new_tab:
        print()
        print(lmc)
        for candidate in new_tab[lmc]:
            print(candidate)
            print(new_tab[lmc][candidate])
            row = []
            outf.write('{}\t{}\t'.format(str(lmc),candidate))
            for mapping in lex.gbr_features:
                if mapping[1] == lmc[1] and mapping[0] in new_tab[lmc][candidate]:
                    row.append(new_tab[lmc][candidate][mapping[0]])
                else:
                    row.append(0.0)
            A.append(row)
            outf.write('\t'.join([str(fl) for fl in row]))
            outf.write('\n')

# print(A)


