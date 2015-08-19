#!/usr/bin/env python
# -*- coding: utf-8 -*-


## Based on learner.js (by Blake Allen and Michael Becker)

import itertools
import collections
from collections import defaultdict

import pdb

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
        self.megatableau = None
        self.relative_size = 0.0

    def __repr__(self):
        # needs aesthetic improvement
        example_count = min(5, len(self.associated_forms))
        return str(self.changes)

    def __str__(self):
       return self.__repr__()





def create_and_reduce_hypotheses(alignments, pre_reduction_cutoff, orientation='product'):
    unfiltered_hypotheses = []
    all_pairs = []
    for alignment in alignments:
        base = linearize_word([column['elem1'] for column in alignment['alignment']])
        derivative = linearize_word([column['elem2'] for column in alignment['alignment']])
        basic_changes = find_basic_changes(alignment['alignment'])
        grouped_changes = group_changes(basic_changes)
        possibilities_for_all_changes = [create_change_possibilities(c, base) for c in grouped_changes]
        product = list(itertools.product(*possibilities_for_all_changes))
        for cp in product:
            unfiltered_hypotheses.append(Sublexicon(cp, [{'base':base, 'derivative':derivative, 'probability':alignment['probability'], 'lexeme':alignment['lexeme']}]))
        all_pairs.append({'base':base, 'derivative':derivative, 'probability':alignment['probability'], 'lexeme':alignment['lexeme']})
    
    combined_hypotheses = combine_identical_hypotheses(unfiltered_hypotheses)
    combined_hypotheses.sort(key=lambda h: len(h.associated_forms))
    combined_hypotheses.reverse()

    if pre_reduction_cutoff:
        combined_hypotheses = [h for h in combined_hypotheses if len(h.associated_forms) >= pre_reduction_cutoff]

    print('Hypotheses ready for reduction. Pre-reduction hypothesis count: {}'.format(str(len(combined_hypotheses))))

    reduced_hypotheses = reduce_hypotheses(combined_hypotheses, all_pairs, orientation)

    sublexicon_sizes = [sum([af['probability'] for af in h.associated_forms]) for h in reduced_hypotheses]
    size_total = sum(sublexicon_sizes)
    for h, size in zip(reduced_hypotheses, sublexicon_sizes):
        h.relative_size = size / size_total

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


def apply_hypothesis(word, hypothesis, orientation='product'):
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
                if orientation == 'source' and current_base[change_position+(i*2)] != s:
                    raise Exception('Deletion incompatible with base: no {} to delete.'.format(s))
                changed_derivative[change_position+(i*2)] = None
        if change.change_type == 'mutate':
            for i, s in enumerate(change.output_material):
                if orientation == 'source' and current_base[change_position+(i*2)] != change.input_material[i]:
                    raise Exception('Mutation incompatible with base: no {} to mutate.'.format(s))
                changed_derivative[change_position+(i*2)] = s

        return (changed_base, changed_derivative)

    current_base = list(add_nones(word))
    current_derivative = list(add_nones(word))

    try:
        for change in hypothesis.changes:
            # if word == 'n e p e n' and change.change_type=='mutate' and change.input_material==['b']:
            #     pdb.set_trace()
            current_base, current_derivative = apply_change(current_base, current_derivative, change)
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


def account_for_all(hypotheses, all_pairs):
    for pair in all_pairs:
        accounted_for_by_each = [apply_hypothesis(pair['base'], h) == pair['derivative'] for h in hypotheses]
        if True not in accounted_for_by_each:
            return False
    return True


def reduce_hypotheses(hypotheses, all_pairs, orientation='product'):
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
                    large_predicted_derivative = apply_hypothesis(small_base, large, orientation)
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
        if account_for_all(combo, all_pairs):
            # winner found! Add missing contexts to their respective winners
            for pair in all_pairs:
                for hypothesis in combo:
                    if apply_hypothesis(pair['base'], hypothesis, orientation) == pair['derivative']:
                        form = pair
                        if form not in hypothesis.associated_forms:
                            hypothesis.associated_forms.append(form) # does combo actually get modified here? Double-check! 
                            break
            return combo


    return [h for h in hypotheses if h != 'purgeable']


def add_zero_probability_forms(hypotheses):
    """Add every form from every hypothesis A to every other hypothesis B with a probability of 0 if the form is not already in hypothesis B.
    """
    all_lexemes_and_bases = [(af['lexeme'], af['base']) for hypothesis in hypotheses for af in hypothesis.associated_forms]

    for hypothesis in hypotheses:
        these_lexemes = [af['lexeme'] for af in hypothesis.associated_forms]
        for lexeme, base in all_lexemes_and_bases:
            if lexeme not in these_lexemes:
                hypothesis.associated_forms.append({'base':base, 'derivative':apply_hypothesis(base,hypothesis), 'probability': 0.0, 'lexeme': lexeme})

    return hypotheses


def add_grammar(sublexicon, constraints, l1_mult = 0.0, l2_mult = 0.001):
    mt = phoment.MegaTableau(sublexicon, constraints)
    sublexicon.weights = phoment.learn_weights(mt, l1_mult, l2_mult)
    sublexicon.constraint_names = constraints
    sublexicon.megatableau = mt
    
    z = sorted(zip(sublexicon.weights, sublexicon.constraint_names), key=lambda x: abs(x[0]), reverse=True)
    print()
    print(sublexicon)
    print(str([(af['base'], af['derivative']) for af in sublexicon.associated_forms if af['probability'] > 0.0][:8]) + '...')
    for w,n in z[:8]:
        print('{}\t{}'.format(str(n),str(w)))

    return sublexicon


def predict_from_one_base_form(base_form, sublexicon):
    candidate = apply_hypothesis(base_form, sublexicon)
    # print(candidate)
    probability = phoment.new_form_probability(base_form, sublexicon.megatableau)

    return (candidate, probability)


# def create_mapping_tableau(sublexicons, megatableaux):
#     new_tableau = {}
#     for s,m in zip(sublexicons, megatableaux):
#         for af in s.associated_forms:
#             if af['lexeme'] in new_tableau:
#                 if af['derivative'] in new_tableau[af['lexeme']]:
#                     new_tableau[af['lexeme']][af['derivative']] += m.tableau[''][af['lexeme']][2]
#                 else:
#                     if af['derivative'] != 'incompatible':
#                         new_tableau[af['lexeme']][af['derivative']] = m.tableau[''][af['lexeme']][2]
#             else:
#                 new_tableau[af['lexeme']] = {}
#                 if af['derivative'] != 'incompatible':
#                     new_tableau[af['lexeme']][af['derivative']] = m.tableau[''][af['lexeme']][2]

#     for lexeme in new_tableau:
#         total = 0
#         ordered_derivatives = sorted([d for d in new_tableau[lexeme]])
#         for derivative in ordered_derivatives:
#             total += new_tableau[lexeme][derivative]
#         for derivative in new_tableau[lexeme]:
#             new_tableau[lexeme][derivative] /= total

#     return new_tableau
