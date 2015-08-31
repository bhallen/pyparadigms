#!/usr/bin/python
# -*- coding: utf-8 -*-

#### ####
#### Based on rdaland/PhoMEnt.git, commit 74f5c43997185aef2d29a7c7dfdac3b5acd760df, from May 2014 ####
#### ####

import sys
from collections import defaultdict
import math
import re
import csv
import scipy, scipy.optimize
import numpy as np
import hypothesize

import pdb



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

def new_form_probability(form, megatableau):
    viols = [len(constraint.findall(form)) for constraint in megatableau.constraints]
    eharmony = math.e ** np.dot(viols, megatableau.weights)
    z = z_score(megatableau.tableau, '') + eharmony # UR is ''

    # if form == 'n a p':
    #     print(form)
    #     print([(c, w) for c, w, v in zip(megatableau.constraints, megatableau.weights, viols)  if v > 0])
    #     print(eharmony)
    #     print(eharmony / z)

    return eharmony / z


### OBJECTIVE FUNCTION(S) ###

def neg_log_probability_with_gradient(weights, tableau, l1_mult=0.0, l2_mult=0.001, gaussian_priors=None):
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
        assert(ur == 'dummy_ur')
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

def neg_log_probability(weights, tableau, l1_mult=0.0, l2_mult=0.001):
    """ Returns just the negative log probability of the data.
    """
    return (nlpwg(weights, tableau, l1_mult, l2_mult))[0]


### OPTIMIZATION FUNCTION

def learn_weights(weights, tableau, l1_mult = 0.0, l2_mult = 0.001, precision = 1000):
    """ Given a filled-in megatableau, return the optimal weight vector.
    """
    # Set up the initial weights and weight bounds (nonpositive reals)
    # w_0 = -scipy.rand(len(weights)) # Random initial weights
    w_0 = [0 for w in weights]       # 0 initial weights
    bounds = [(-20,20) for wt in weights]

    # Find the best weights
    learned_weights, fneval, rc = scipy.optimize.fmin_l_bfgs_b(nlpwg, w_0, \
        args = (tableau, l1_mult, l2_mult), bounds=bounds, factr=precision)

    # Update the mt in place with the new weights
    weights = learned_weights


    # # Be sociable
    # print("\nBoom! Weights have been updated:")
    # for i in range(0,len(learned_weights)):
    #     print(mt.constraint_names[i],"\t",learned_weights[i])
    # print("\nLog probability of data:", -(nlpwg(learned_weights, mt.tableau))[0])
    # print("")

    # Return
    return learned_weights
