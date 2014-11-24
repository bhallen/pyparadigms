#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import csv

import aligner
import hypothesize
import phoment
import gbr


alr = aligner.Aligner(feature_file='japanese_features.txt', sub_penalty=4.0, tolerance=1.0)

lex = gbr.Lexicon('JP_paradigms_small.txt')

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

for one_input in sll_inputs:
    all_alignments = []
    for pair in sll_inputs[one_input]:
        alignments = []
        for alignment in alr.align(pair[1].split(' '), pair[2].split(' ')):
            alignments.append(alignment+[1.0]) # add support for reading probabilities in from the inputs (rather than assigning all observed forms 1.0)

        scored_alignments = []
        for a in alignments:
            final_score = a[1]/alr.check_cohesion(a[0])/len(a[0]) # should this really divide by the length of the alignment?
            scored_alignments.append((a[0], final_score))
        scored_alignments.sort(key=lambda x: x[1])
        scored_alignments.reverse()

        for a in scored_alignments:
            alr.display_alignment(a[0])
            print(a[1])

        all_alignments.append(scored_alignments) # TO-DO: add ability to skim off only best scoring alignments

 
    # reduced_hypotheses = hypothesize.create_and_reduce_hypotheses(alignments)

    # sublexicons = hypothesize.add_zero_probability_forms(reduced_hypotheses)

    # print(sublexicons)



    # with open('en_constraints.txt') as con_file:
    #     conreader = csv.reader(con_file, delimiter='\t')
    #     constraints = [c[0] for c in conreader if len(c) > 0]

    #     sublexicons, megatableaux = zip(*[hypothesize.add_grammar(s, constraints) for s in sublexicons])