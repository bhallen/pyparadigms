#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import csv

import aligner
import hypothesize
import phoment


alr = aligner.Aligner(feature_file='en_features.txt', sub_penalty=3.0, tolerance=1.0)

alignments = []

with open('en_ex_train.txt') as training_file:
    trainingreader = csv.reader(training_file, delimiter='\t')
    training = [line for line in trainingreader if len(line) > 0]
    training = [line[:2]+[float(line[2])] if len(line) == 3 else line[:2]+[1.0] for line in training]

    for triple in training:
        for alignment in alr.align(triple[0].split(' '), triple[1].split(' ')): # To-do: trim any extra spaces off
            alignments.append([alignment]+[triple[2]])

    reduced_hypotheses = hypothesize.create_and_reduce_hypotheses(alignments)

    sublexicons = hypothesize.add_zero_probability_forms(reduced_hypotheses)

    with open('en_constraints.txt') as con_file:
        conreader = csv.reader(con_file, delimiter='\t')
        constraints = [c[0] for c in conreader if len(c) > 0]

        sublexicons, megatableaux = zip(*[hypothesize.add_grammar(s, constraints) for s in sublexicons])