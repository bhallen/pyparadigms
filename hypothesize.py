#!/usr/bin/env python
# -*- coding: utf-8 -*-


## Based on learner.js (by Michael Becker and Blake Allen)

import itertools
import collections
from collections import defaultdict

import phoment


class Change(object):

    def __init__(self, change_type, position, input_material, output_material):
        self.change_type = change_type
        self.position = position
        self.input_material = input_material
        self.output_material = output_material

    def __repr__(self):
        # needs aesthetic improvement
        return '{0} {1} to {2} at {3}'.format(self.change_type, self.input_material, self.output_material, self.position)

    def __str__(self):
       return self.__repr__()


class Sublexicon(object):
    """Starts off as a hypothesis; will grow and compete with others, potentially becoming a sublexicon of the final grammar
    """

    def __init__(self, changes, associated_forms):
        self.changes = changes
        self.associated_forms = associated_forms
        self.constraint_names = None
        self.weights = None

    def __repr__(self):
        # needs aesthetic improvement
        example_count = min(5, len(self.associated_forms))
        return str(self.changes)

    def __str__(self):
       return self.__repr__()





def create_and_reduce_hypotheses(alignments):

    unfiltered_hypotheses = []
    all_bd_pairs = []
    for alignment in alignments:
        base = linearize_word([column['elem1'] for column in alignment[0]])
        derivative = linearize_word([column['elem2'] for column in alignment[0]])
        basic_changes = find_basic_changes(alignment[0])
        grouped_changes = group_changes(basic_changes)
        possibilities_for_all_changes = [create_change_possibilities(c, base) for c in grouped_changes]
        product = list(itertools.product(*possibilities_for_all_changes))
        for cp in product:
            unfiltered_hypotheses.append(Sublexicon(cp, [{'base':base, 'derivative':derivative, 'probability':alignment[1]}]))
        all_bd_pairs.append((base,derivative))
    
    combined_hypotheses = combine_identical_hypotheses(unfiltered_hypotheses)
    combined_hypotheses.sort(key=lambda h: len(h.associated_forms))
    combined_hypotheses.reverse()

    reduced_hypotheses = reduce_hypotheses(combined_hypotheses, all_bd_pairs)

    return reduced_hypotheses



def find_basic_changes(alignment):
    """Find the differences between the aligned base and derivative.
    Return differences as Changes with positive indices as positions.
    """
    changes = []
    surface_i = 0
    for column in alignment:
        if column['elem1'] != column['elem2']:
            if column['elem1'] == None:
                changes.append(Change('insert', surface_i*2, [column['elem1']], [column['elem2']]))
                # surface_i does not increment
            elif column['elem2'] == None:
                changes.append(Change('delete', surface_i*2+1, [column['elem1']], [column['elem2']]))
                surface_i += 1
            else:
                changes.append(Change('mutate', surface_i*2+1, [column['elem1']], [column['elem2']]))
                surface_i += 1
        else:
            surface_i += 1

    return changes


def create_change_possibilities(change, base, side='both'):
    """Given a change with segments as input and output and a positive index as position,
    return a list of changes with different positions/inputs/outputs.
    """
    change_possibilities = []
    if side in ['left', 'both']:
        change_possibilities.append(change)
    if side in ['right', 'both']:
        noned_base = add_nones(base)
        new_change = Change(change.change_type, -(len(noned_base)-change.position), change.input_material, change.output_material)
        change_possibilities.append(new_change)

    return change_possibilities


def group_changes(changes):
    """Consolidate same-position insertions and deletions into single changes.
    """
    insertions = [c for c in changes if c.change_type == 'insert']
    deletions = [c for c in changes if c.change_type == 'delete']
    mutations = [c for c in changes if c.change_type == 'mutate']
    inserted_locations = [ins.position for ins in insertions]

    grouped_insertions = []
    for i, ins in enumerate(insertions):
        if i > 0:
            if ins.position == insertions[i-1].position:
                grouped_insertions[-1].output_material += ins.output_material
                continue
        grouped_insertions.append(ins)


    grouped_deletions = []
    for i, dlt in enumerate(deletions):
        if i > 0:
            if dlt.position == deletions[i-1].position+2 and dlt.position-1 not in inserted_locations:
                grouped_deletions[-1].input_material += dlt.input_material
                continue
        grouped_deletions.append(dlt)

    return sorted(grouped_insertions + grouped_deletions + mutations, key=lambda x: x.position)


def combine_identical_hypotheses(hypotheses):
    """Combine hypotheses with the same Change objects, yielding hypotheses with associated assoc_forms
    that are the superset of component hypotheses.
    """
    temp_dict = defaultdict(list)
    for h in hypotheses:
        temp_dict[str(h.changes)].append(h)

    grouped_hypotheses = []
    for gh in temp_dict:
        assoc_forms = [h.associated_forms[0] for h in temp_dict[gh]]
        grouped_hypotheses.append(Sublexicon(temp_dict[gh][0].changes, assoc_forms))

    return grouped_hypotheses


def add_nones(word):
        """Change word into a list and add None at its beginning, end, and between every other pair of elements. Works whether the word is a str or a list.
        """
        def yield_it(word_string):
            yield None
            it = iter(word_string)
            yield next(it)
            for x in it:
                yield None
                yield x
            yield None

        if isinstance(word, str):
            return list(yield_it(word.split(' ')))
        else:
            return list(yield_it(word))


def apply_hypothesis(word, hypothesis):
    """Apply the changes in a hypothesis to a (base) word. Base word can be either
    a list of segments (no Nones) or a space-spaced string.
    """

    def apply_change(current_base, current_derivative, change):
        """Use the given set of changes to derive a new form from the base word.
        May be only one intermediate step in the application of multiple
        changes associated with a single hypothesis/sublexicon.
        """
        change_position = make_index_positive(current_base, change.position)

        changed_base = current_base[:]
        changed_derivative = current_derivative[:]

        if change.change_type == 'insert':
            changed_base[change_position] = [None for s in change.output_material]
            changed_derivative[change_position] = change.output_material
        if change.change_type == 'delete':
            for i, s in enumerate(change.input_material):
                changed_derivative[change_position+(i*2)] = None
        if change.change_type == 'mutate':
            for i, s in enumerate(change.output_material):
                changed_derivative[change_position+(i*2)] = s

        return (changed_base, changed_derivative)

    current_base = list(add_nones(word))
    current_derivative = list(add_nones(word))

    try:
        for c in hypothesis.changes:
            current_base, current_derivative = apply_change(current_base, current_derivative, c)
    except:
        return 'incompatible'

    return linearize_word(current_derivative)


def make_index_positive(word, index):
    """Return positive index based on word.
    """
    if index >= 0:
        return index
    else:
        return len(word) + index


def linearize_word(word):
    """Create a space-spaced string from a list-formatted word (even one with Nones).
    """
    def flatten(l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, str):
                for sub in flatten(el):
                    yield sub
            else:
                yield el
    flat_noneless = [s for s in list(flatten(word)) if s != None]
    return ' '.join(flat_noneless)


def account_for_all(hypotheses, all_bd_pairs):
    for base, derivative in all_bd_pairs:
        accounted_for_by_each = [apply_hypothesis(base, h) == derivative for h in hypotheses]
        if True not in accounted_for_by_each:
            return False
    return True


def reduce_hypotheses(hypotheses, all_bd_pairs):
    """Condenses the list of hypotheses about the entire dataset into the
    minimum number required to account for all base-derivative pairs.
    """
    reversed_hypotheses = hypotheses[::-1]
    # First step: check to see if any small hypotheses can be consumed by any single larger one
    for j, large in enumerate(hypotheses): # j = potential consumer, will be at least as large as consumed (i)
        for i, small in enumerate(reversed_hypotheses): # can be consumed
            if small != 'purgeable' and large != 'purgeable' and small != large:
                consumabilities = [] # could probably be refactored so as not to need to determine all consumabilities (only until failure)
                for associated_form in small.associated_forms:
                    small_base = associated_form['base']
                    small_derivative = associated_form['derivative']
                    large_predicted_derivative = apply_hypothesis(small_base, large)
                    consumabilities.append(small_derivative == large_predicted_derivative)
                if False not in consumabilities: # if there are no forms in small that large cannot account for
                    for bd in small.associated_forms:
                        if bd not in large.associated_forms:
                            large.associated_forms.append(bd)
                    hypotheses[-(i+1)] = 'purgeable'
                    reversed_hypotheses[i] = 'purgeable'


    hypotheses = [h for h in hypotheses if h != 'purgeable']

    # Second step: check for smallest number of adequate hypotheses
    combinations = itertools.chain.from_iterable([itertools.combinations(hypotheses, n) for n in range(1,len(hypotheses))])
    for combo in combinations:
        if account_for_all(combo, all_bd_pairs):
            # winner found! Add missing contexts to their respective winners
            for bd_pair in all_bd_pairs:
                for hypothesis in combo:
                    if apply_hypothesis(bd_pair[0], hypothesis) == bd_pair[1]:
                        form = {'base':bd_pair[0], 'derivative':bd_pair[1]}
                        if form not in hypothesis.associated_forms:
                            hypothesis.associated_forms.append(form) # does combo actually get modified here? Double-check! 
                            break
            return combo


    return [h for h in hypotheses if h != 'purgeable']


def add_zero_probability_forms(hypotheses):
    """Add every form from every hypothesis A to every other hypothesis B with a probability of 0 if the form is not already in hypothesis B.
    """
    all_bases = [af['base'] for hypothesis in hypotheses for af in hypothesis.associated_forms]

    for hypothesis in hypotheses:
        these_bases = [af['base'] for af in hypothesis.associated_forms]
        for base in all_bases:
            if base not in these_bases:
                hypothesis.associated_forms.append({'base':base, 'derivative':apply_hypothesis(base,hypothesis), 'probability': 0.0})

    return hypotheses


def add_grammar(sublexicon, constraints):
    mt = phoment.MegaTableau(sublexicon, constraints)
    sublexicon.weights = phoment.learn_weights(mt)
    sublexicon.constraint_names = constraints

    for ur in mt.tableau:
        for sr in ur:
            print(sr)

    return (sublexicon, mt)