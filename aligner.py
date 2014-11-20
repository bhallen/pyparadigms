#!/usr/bin/env python
# -*- coding: utf-8 -*-


## Based on aligner.js (by Michael Becker and Blake Allen),
## which in turn was based on Peter Kleiweg's Levenshtein Demo.

import random
from collections import defaultdict
import sys


class Aligner(object):

    def __init__(self, ins_penalty=1, del_penalty=1, 
                 sub_penalty=1, tolerance=0, feature_file=None):
        self.ins_penalty = ins_penalty
        self.del_penalty = del_penalty
        self.sub_penalty = sub_penalty
        self.tolerance = tolerance
        self.features_tf = False
        if feature_file:
            self.read_feature_file(feature_file)


    def read_feature_file(self, feature_file):
        with open(feature_file) as ffile:
            instring = ffile.read().rstrip().split('\n')
            colnames = instring[0].split('\t')
            segments = [line.split('\t') for line in instring[1:]]
            feature_dict = {}
            for segment in segments:
                values_dict = {}
                for i,feature_name in enumerate(colnames[1:]):
                    values_dict[feature_name] = segment[i+1]
                feature_dict[segment[0]] = values_dict

            self.features_tf = True
            self.features = feature_dict

        # Set features for the silent (empty) pseudo-segment
        try:
            self.silence_features = self.features['empty']
        except (TypeError, KeyError):
            rand_segment = random.choice(list(features.keys()))
            self.silence_features = {}
            for feature in self.features[rand_segment]:
                self.silence_features[feature] = '0'


    def align(self, seq1=None, seq2=None):
        similarity_matrix = self.make_similarity_matrix(seq1, seq2)
        alignments = self.generate_alignments(seq1, seq2, similarity_matrix)
        return alignments


    def make_similarity_matrix(self, seq1=None, seq2=None):
        if seq1 == None:
            seq1 = []
        if seq2 == None:
            seq2 = []

        d = []

        def compare(x, y):
            return x - y <= self.tolerance

        initial_vals = {'aboveleft': 0,
                        'above': 0,
                        'left': 0,
                        'trace': 0,
                        'f': None}

        d = [[initial_vals.copy() for y in seq2+[' ']] for x in seq1+[' ']]
        d[0][0]['f'] = 0

        # Fill in leftmost column and topmost row with comparisons to empty
        for x in range(1, len(seq1)+1):
            d[x][0]['f'] = d[x-1][0]['f'] + self.compare_segments(seq1[x-1], 'empty')
            d[x][0]['left'] = 1

        for y in range(1, len(seq2)+1):
            d[0][y]['f'] = d[0][y-1]['f'] + self.compare_segments('empty', seq2[y-1])
            d[0][y]['above'] = 1

        # Fill in all other cells
        for x in range(1, len(seq1)+1):
            for y in range(1, len(seq2)+1):
                aboveleft = (d[x - 1][y - 1]['f'] + self.compare_segments(seq1[x-1], seq2[y-1]))
                left = d[x - 1][y]['f'] + self.compare_segments(seq1[x-1], 'empty')
                above = d[x][y - 1]['f'] + self.compare_segments('empty', seq2[y-1])

                if compare(aboveleft,above) and compare(aboveleft,left):
                    d[x][y]['f'] = aboveleft
                    d[x][y]['aboveleft'] = 1

                if compare(above,aboveleft) and compare(above,left):
                    d[x][y]['f'] = above
                    d[x][y]['above'] = 1

                if compare(left,aboveleft) and compare(left,above):
                    d[x][y]['f'] = left
                    d[x][y]['left'] = 1

                d[x][y]['f'] = min(aboveleft, above, left)

        return d



    def compare_segments(self, segment1, segment2, underspec_cost=.25):

        def check_feature_difference(val1, val2):  
            if val1 == val2:
                return 0
            elif val1 == '0' or val2 == '0':
                return underspec_cost
            else:
                return 1

        if self.features_tf:
            if segment1 == 'empty':
                fs2 = self.features[segment2]
                return (sum(check_feature_difference('0', 
                            fs2[f]) for f in fs2) * self.ins_penalty)    # or should this be addition?
            elif segment2 == 'empty':
                fs1 = self.features[segment1]
                return (sum(check_feature_difference(fs1[f], 
                        '0') for f in fs1) * 
                        self.del_penalty)    # or should this be addition?
            else:
                fs1 = self.features[segment1]
                fs2 = self.features[segment2]
                return (sum(check_feature_difference(fs1[f], fs2[f])
                            for f in fs1) * self.sub_penalty)    # or should this be addition?
        else:
            if segment1 == 'empty':
                return self.ins_penalty * underspec_cost
            elif segment2 == 'empty':
                return self.del_penalty * underspec_cost
            else:
                return int(segment1!=segment2) * self.sub_penalty   # or should this be addition (with, in this case, a boolean condition)?


    def generate_alignments(self, seq1, seq2, d):

        def advance_alignment(so_far, x, y):
            if x > 0 or y > 0:
                if d[x][y]['aboveleft']:
                    current_element = {'elem1': seq1[x-1], 'elem2': seq2[y-1], 'dir': 'aboveleft'}
                    queued_ready_objs.append(([current_element] + so_far, x-1, y-1))
                if d[x][y]['above']:
                    current_element = {'elem1': None, 'elem2': seq2[y-1], 'dir': 'above'}
                    queued_ready_objs.append(([current_element] + so_far, x, y-1))
                if d[x][y]['left']:
                    current_element = {'elem1': seq1[x-1], 'elem2': None, 'dir': 'left'}
                    queued_ready_objs.append(([current_element] + so_far, x-1, y))
            else:
                alignments.append(so_far)

        alignments = []
        queued_ready_objs = [([], len(seq1), len(seq2))]
        while queued_ready_objs:
            advance_alignment(*queued_ready_objs[0])
            queued_ready_objs = queued_ready_objs[1:]

        return alignments



    def display_alignment(self, alignment):
        top_list, bottom_list = [], []
        for chunk in alignment:
            top_bit = chunk['elem1'] if chunk['elem1'] != None else '_'
            bottom_bit = chunk['elem2'] if chunk['elem2'] != None else '_'
            length = max(len(top_bit), len(bottom_bit))
            top_list.append(top_bit + (' ' * (length-len(top_bit))))
            bottom_list.append(bottom_bit + (' ' * (length-len(bottom_bit))))

        print(' '.join(top_list))
        print(' '.join(bottom_list))
