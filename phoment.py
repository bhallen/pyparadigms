#!/usr/bin/python
# -*- coding: utf-8 -*-

#### ####
#### Based on rdaland/PhoMEnt.git, commit 74f5c43997185aef2d29a7c7dfdac3b5acd760df, from May 2014 ####
#### ####

import sys
from collections import defaultdict
import math
import re
import scipy, scipy.optimize
import numpy as np
import hypothesize

class MegaTableau(object):

    """
    A representation of tableaux for manipulation by the maxent learner.
    Derived from a file of tab-delimited tableaux.
    Contains the following attributes:
        self.constraints -------- list of constraint names
            this is found on the first line of the input file
        self.weights ------------ a list of weights for constraints
        self.tableau ------------ a dictionary of dictionaries:
            {input: {output: [freq, violDic, maxentScore]}}
            freq = float()
            violDic = dictionary of constraint violations (integers). 
                Keys are constraint indices, based on order of constraints in self.constraints
            maxentScore = e**harmony. Initialized to zero (because harmony is undefined without weights).
    Contains the following methods:
        self.read_megt_file(megt_file) - moves the data from the .txt file to the attributes
            self.weights is not populated.
        self.read_weights_file(megt_file) - populates self.weights
    """
    
    def __init__(self, sublexicon=None, constraints=None):
        """
        sublexicon -- a sublexicon (originally a hypothesis) from hypothesize.py
        constraints -- a list of strings corresponding to the phonological constraints to be weighted
        """
        self.constraint_names = constraints
        self.gaussian_priors = {}
        self.tableau = defaultdict(dict)
        if sublexicon and constraints:
            self.constraints = self.create_re_constraints(constraints)
            self.weights = np.zeros(len(self.constraints))
            self.populate_tableau(sublexicon)


    def create_re_constraints(self, constraints):
        """To-do: add the ability to translate featural constraints
        """
        return [re.compile(c) for c in constraints]


    def populate_tableau(self, sublexicon):
        outputs = {}
        for af in sublexicon.associated_forms:
            violations = {}
            for c in range(len(self.constraints)):
                these_violations = len(self.constraints[c].findall(af['base']))
                if these_violations > 0:
                    violations[c] = these_violations
            outputs[af['base']] = [af['probability'], violations, 0]
        self.tableau = {'': outputs}

### HELPER FUNCTIONS FOR CALCULATING PROBABILITY ###

def maxent_value(weights, tableau, ur, sr):
    """ Compute maxent value P* = exp(harmony) for a particular UR/SR pair.
    """
    harmony = 0
    very_very_tiny_number = np.finfo(np.double).tiny # Approximately 2.2e-308
    for c in tableau[ur][sr][1]:
        harmony += weights[c] * tableau[ur][sr][1][c]
    return math.exp(harmony) + very_very_tiny_number # Makes positive any "0" results created by roundoff error.

def z_score(tableau, ur):
    """ Compute the Z-score for a particular UR, using current maxent values.
    """
    zScore = 0
    for j in tableau[ur]:
        zScore += tableau[ur][j][2]
    return zScore

def update_maxent_values(weights, tableau):
    """ Computes maxent value P* = exp(harmony) for all UR/SR pairs
    in a supplied tableau, and updates the tableau with these values.
    """
    for ur in tableau:
        for sr in tableau[ur]:
            tableau[ur][sr][2] = maxent_value(weights, tableau, ur, sr)


### OBJECTIVE FUNCTION(S) ###

def neg_log_probability_with_gradient(weights, tableau, l1_mult=0.0, l2_mult=1.0, gaussian_priors=None):
    """ Returns the negative log probability of the data AND a gradient vector.
    This is the objective function used in learn_weights().
    """
    update_maxent_values(weights, tableau)
    logProbDat = 0
    observed = [0 for i in range(len(weights))] # Vector of observed violations
    expected = [0 for i in range(len(weights))] # Vector of expected violations

    # Gaussian priors override L1/L2 priors
    if gaussian_priors:
        mus, sigmas = gaussian_priors[0], gaussian_priors[1]
        normalized = (weights-mus)/sigmas
        prob_prior = -(0.5*sum(normalized*normalized))
        grad_prior = -(normalized/sigmas)
    else:
        l1_prob_prior = -(l1_mult * sum(weights))
        l2_prob_prior = l2_mult * sum(weights*weights)
        l1_grad_prior = -(l1_mult * scipy.ones(len(weights)))
        l2_grad_prior = 2 * l2_mult * weights
        prob_prior = -(l1_prob_prior + l2_prob_prior)
        grad_prior = -(l1_grad_prior + l2_grad_prior)

    for ur in tableau:
        ur_count = 0 # Total observed for this UR
        z = z_score(tableau, ur)
        new_expected = [0 for i in range(len(weights))]
        
        for sr in tableau[ur]:
            ur_count += tableau[ur][sr][0]
            prob = tableau[ur][sr][2] / z
            logProbDat += math.log(prob) * tableau[ur][sr][0]
            for c in tableau[ur][sr][1]:
                observed[c] += tableau[ur][sr][1][c] * tableau[ur][sr][0]
                new_expected[c] += tableau[ur][sr][1][c] * prob
                
        for i in range(0,len(expected)):
            expected[i] += new_expected[i] * ur_count
            
    logProbDat += prob_prior
    gradient = [e-o-p for e, o, p  in zip(expected, observed, grad_prior)] # i.e. -(observed minus expected)
    return (-logProbDat, np.array(gradient))

nlpwg = neg_log_probability_with_gradient # So you don't get carpal tunnel syndrome.

def neg_log_probability(weights, tableau, l1_mult=0.0, l2_mult=1.0):
    """ Returns just the negative log probability of the data.
    """
    return (nlpwg(weights, tableau, l1_mult, l2_mult))[0]


### OPTIMIZATION FUNCTION

def learn_weights(mt, l1_mult = 0.0, l2_mult = 1.0, precision = 10000000):
    """ Given a filled-in megatableau, return the optimal weight vector.
    """
    # Set up the initial weights and weight bounds (nonpositive reals)
    w_0 = -scipy.rand(len(mt.weights)) # Random initial weights
    #w_0 = [0 for w in mt.weights]       # 0 initial weights
    nonpos_reals = [(-50,0) for wt in mt.weights]

    # Find the best weights
    learned_weights, fneval, rc = scipy.optimize.fmin_l_bfgs_b(nlpwg, w_0, \
        args = (mt.tableau,l1_mult,l2_mult, mt.gaussian_priors), bounds=nonpos_reals, factr=precision)

    # Update the mt in place with the new weights
    mt.weights = learned_weights

    # Be sociable
    print("\nBoom! Weights have been updated:")
    for i in range(0,len(learned_weights)):
        print(mt.constraint_names[i],"\t",learned_weights[i])
    print("\nLog probability of data:", -(nlpwg(learned_weights, mt.tableau))[0])
    print("")

    # Return
    return learned_weights
