"""
LEARNING

For each derivative cell:
----For each other base cell:
--------Run the SLL (no grammars) on this b-d pair to get its sublexicons
----Group the words in d into ``superlexicons'' **SEE BELOW
----For each superlexicon:
--------Learn a grammar that distinguishes that superlexicon from others for all base cells, defaulting to only single-base constraints/features
--------If the grammar doesn't meet X standard (success on training data...cross-validation?), find specific bad results, use them to get constraints
--------Add these constraints to the grammar and re-fit weights

** How to group words in d into superlexicons? Options:
1. Just look at all the lexemes and check which sublexicons they're in for different bases. Each unique combination of sublexicons is a superlexicon. But what about lexemes for which some base forms are unknown?
2. Similar to getting sublexicons: try applying combos of sublexicons and see which ones are consistent with which lexemes. This is probably really slow, and it's not clear if it's better than (1)...

NOTE: search for constraints needs to be limited to individual segment unigrams. They're discovered using something like the MGL's island detection, i.e. O/E. Although this is certainly not expressive enough, it is a necessary assumption for the model to be able to run on e.g. my laptop...
"""

"""
PRODUCTIVITY

For each unknown cell d of the target lexeme:
----For each superlexicon:
--------Give known bases B to the superlexicon, get p(B|s)
----Use Bayes' theorem to get p(s|B), then use this to get a distribution p(d|B)
"""

import aligner
import hypothesize
import phoment

import csv
import re
import numpy as np
from collections import defaultdict
import itertools
import pickle

import pdb

PRE_REDUCTION_CUTOFF = None


class Lexicon:                                                                  #TO-DO: consider renaming InflectionalSystem?
    def __init__(self):
        self.sfeatures = []
        self.wordlist = []
        self.cells = defaultdict(dict)
        self.lexemes = []
        self.mappings = defaultdict(dict) # morphological operations from b to d
        self.grammars = {c: None for c in self.cells}
        self.aligner = None
        self.psublexicons = {} # {deriv_cell: [PSublexicons]}
        self.test_data = None
        self.test_predictions = None
        self.relative_size = None

    def read_training_file(self, training_filename):
        """See documentation for detailed formatting requirements.
        In short, first line must he headers, and the order of columns
        must be form, lexeme, sfeatures..., (frequency)
        """
        with open(training_filename) as lf:
            lfstr = lf.read().rstrip().split('\n')
            self.sfeatures = lfstr[0].split('\t')[2:]
            if self.sfeatures[-1] == 'frequency':
                self.sfeatures = self.sfeatures[:-1]
                frequency_col = True
            else:
                frequency_col = False
            lfstr = lfstr[1:]
            for entry in lfstr:
                if len(entry) > 2:
                    cols = entry.split('\t')
                    word = Word()
                    word.form = cols[0]
                    word.lexeme = cols[1]
                    word.sfeatures = {}
                    for i, sfeature in enumerate(self.sfeatures):
                        word.sfeatures[sfeature] = cols[i+2]
                    if frequency_col:
                        word.frequency = cols[-1]
                    self.wordlist.append(word)
                    if word.lexeme not in self.lexemes:
                        self.lexemes.append(word.lexeme)

    def learn(self, training_filename, constraints_filename, features_filename):
        print('Initializing aligner...')
        self.aligner = aligner.Aligner(feature_file=features_filename, 
                                   sub_penalty=4.0, tolerance=1.0)              #TO-DO: Remove these magic numbers

        print('Loading training file...')
        self.read_training_file(training_filename)

        ## Create dict of *OBSERVED* MPS tuples and their forms
        print('Compiling information about available cells...')
        self.create_cells()

        for deriv_cell in self.cells:
            print('Learning grammar for the derivative cell {}...'
                  .format(deriv_cell))
            for base_cell in self.cells:
                if base_cell != deriv_cell:
                    print('Learning base sublexicons from {} to {}...'
                           .format(base_cell, deriv_cell))
                    self.mappings[deriv_cell][base_cell] = \
                           self.find_bd_sublexicons(base_cell, deriv_cell)

            print('Finding this derivative cell\'s paradigm sublexicons...')
            this_deriv_psublexicons = self.find_psublexicons(
                                                      self.mappings[deriv_cell],
                                                      deriv_cell)
            print('Learning a MaxEnt HG for each paradigm sublexicon...')
            for ps in this_deriv_psublexicons:
                ps.learn_grammar(constraints_filename)

            self.psublexicons[deriv_cell] = this_deriv_psublexicons






    def save_lexicon(self, language_name):
        with open('datasets/{}/{}.lexicon'.format(
                          language_name, language_name), 'wb') as lexfile:
            pickle.dump(self, lexfile)


    def predict(self, language_name):
        self.read_testing_file(language_name)
        self.complete_test_data()

    def read_testing_file(self, language_name):
        """(** formatting conventions **)"""
        with open('datasets/{}/{}_testing.txt'.format(
                  language_name, language_name)) as testfile:
            test_data = defaultdict(dict)
            tfstr = testfile.read().rstrip().split('\n')
            sfeatures = tfstr[0].split('\t')[2:]
            tfstr = tfstr[1:]
            for entry in tfstr:
                if len(entry) > 2:                                              #TO-DO: add support for freq specification
                    row_entries = entry.split('\t')
                    lexeme = row_entries[1]
                    form = row_entries[0]
                    fvals = []
                    for i, sfeature in enumerate(sfeatures):
                        fvals.append((sfeature, row_entries[i+2]))
                    fvals = tuple(sorted(fvals))
                    try:
                        test_data[lexeme][fvals][form] = 1.0 # freq here
                    except KeyError:
                        test_data[lexeme][fvals] = {}
                        test_data[lexeme][fvals][form] = 1.0 # freq here

            self.test_data = test_data


    def complete_test_data(self):
        predictions = defaultdict(dict)
        for lexeme in self.test_data:
            for deriv_cell in self.cells:
                if deriv_cell not in self.test_data[lexeme]:
                    predictions[lexeme][deriv_cell] = (
                        self.predict_candidate_distribution(
                            self.test_data[lexeme], deriv_cell))
        
        self.test_predictions = predictions
        # print(self.test_predictions)


    def predict_candidate_distribution(self, givens, deriv_cell):
        candidates = {base_cell: 
                          self.predict_from_base_distribution(givens[base_cell],
                                                              base_cell,
                                                              deriv_cell)
                                    for base_cell in givens}

        # below hasn't been updated...

        ## Massage dict into {candidate: pred_prob}
        arranged_dict = defaultdict(lambda:1.0)
        for base in candidates:
            for base_form in candidates[base]:
                for deriv_form in candidates[base][base_form]:
                    arranged_dict[deriv_form] *= candidates[base][base_form][deriv_form] ## TO-DO Determine how best to deal with 0s here

        # print(candidates)
        psum = sum([arranged_dict[c] for c in arranged_dict])
        for c in arranged_dict:
            arranged_dict[c] /= psum

        # print()
        # print(arranged_dict)
        return arranged_dict



    def predict_from_base_distribution(self, 
                                      base_distribution, base_cell, deriv_cell):
        base_predictions = {base: {} for base in base_distribution}
        # print()
        for base_form in base_distribution:
            # print(base_form)
            # print(self.psublexicons)
            for psl in self.psublexicons[deriv_cell]:
                # print(psl)
                candidate, probability = self.predict_from_one_base_form(
                                                                base_form,
                                                                psl,
                                                                base_cell)

                base_predictions[base_form][candidate] = (
                    ## p(base|sublexicon)p(sublexicon)
                    probability * psl.relative_size)

        # print(base_predictions)
        return base_predictions


    def predict_from_one_base_form(self, base_form, psublexicon, base_cell):
        candidate = self.apply_appropriate_operations(
                                              base_form, psublexicon, base_cell)
        probability = phoment.new_form_probability(
                                              base_form, psublexicon, base_cell)

        return (candidate, probability)


    def apply_appropriate_operations(self, base_form, psublexicon, base_cell):
        current_base = list(hypothesize.add_nones(base_form))
        current_derivative = list(hypothesize.add_nones(base_form))
        changes = psublexicon.operations[base_cell]

        try:
            for change in changes:
                ## !! hypothesize.apply_change was reeeeally slow when taken out
                ## of apply_hypothesis (while running learn.py). Figure out why
                ## to get this line below to work.
                current_base, current_derivative = hypothesize.apply_change(
                                       current_base, current_derivative, change)
        except:
            return 'incompatible'

        return hypothesize.linearize_word(current_derivative)


    def test_predictions(self):
        with open('predictions_test.txt', 'w') as testout:
            outstr = 'CELL\tLEXEME\tOUTPUT\tOBS\tPRED\n'
            for lexeme in self.test_data:
                outstr += '\t'.join([str(lexeme[0]),lexeme[1]]) + '\t'
                weighted_prob_sums = []
                obs_probs = []
                for output in testing_dict[lexeme]:
                    obs_probs.append(testing_dict[lexeme][output]['observed'])
                    weighted_prob_sum = 0
                    for feature in weight_dict:
                        if feature != (lexeme[0],):
                            weighted_prob_sum += testing_dict[lexeme][output][feature]*weight_dict[feature] # cond.prob * weight
                    weighted_prob_sums.append(weighted_prob_sum)
                Z = sum(weighted_prob_sums)
                marginal_probs = [wps/Z for wps in weighted_prob_sums]
                marginal_probs = zip([output for output in testing_dict[lexeme]],obs_probs,marginal_probs) # FINAL element is predicted prob!
                marginal_probs.sort(key=lambda tup: tup[2], reverse=True)

                outstr += '\n\t\t'.join([output+'\t'+str(obs)+'\t'+str(pred) for output,obs,pred in marginal_probs]) + '\n'

            testout.write(outstr)

    def save_predictions(self, language_name):
        pass


    def find_bd_sublexicons(self, base_cell, deriv_cell):
        all_alignments = []
        for lexeme in self.cells[deriv_cell]:
            if lexeme in self.cells[base_cell]:
                alignments = []
                for alignment, score in self.aligner.align(
                                  self.cells[base_cell][lexeme].form.split(' '),
                                self.cells[deriv_cell][lexeme].form.split(' ')):
                    final_score = score/self.aligner.check_cohesion(alignment)  #TO-DO: should this divide by the length of the alignment?
                    alignments.append({'alignment':alignment, 
                                       'probability': 1.0, 
                                       'lexeme': lexeme, 
                                       'score': final_score})
                alignments.sort(key=lambda x: x['score'])
                alignments.reverse()


                #TO-DO: only add up to MAXALIGNMENTS
                all_alignments += alignments

        print('Number of alignments: {}'.format(str(len(all_alignments))))

        print('Reducing hypotheses...')
        bsublexicons = hypothesize.create_and_reduce_hypotheses(
                     all_alignments, PRE_REDUCTION_CUTOFF, orientation='source')
        print('Hypotheses have been reduced.')

        ## Should NOT add zero frequency forms yetsince associated words are
        ## used to determine paradigm sublexicons
        print('Number of base sublexicons: {}'.format(str(len(bsublexicons))))

        return bsublexicons


    def find_psublexicons(self, bsublex_dict, deriv_cell):
        """This function first organizes lexemes by their conjunctions of
        base sublexicon operations, and then creates a PSublexicon for each
        unique combination of base sublexicon operation conjunctions.
        """

        print('Why does the 1s grammar often end up with just 1 or 2 sublexicons?'
              +' Sometimes 0, eek! Seems to be because of bad operation positions')

        lexeme_dict = defaultdict(list) # {lexeme: (base_cell, [Changes])}
        for base_cell in bsublex_dict:
            for bsublex in bsublex_dict[base_cell]:
                for af in bsublex.associated_forms:
                    lexeme_dict[af['lexeme']].append(
                      ((base_cell, bsublex.changes), 
                       Word(af['base'],
                            af['lexeme'],
                            base_cell,
                            af['probability'])))

        ####
        ## Issue: what to do when not all forms of a lexeme are known, meaning that
        ## it can't be conclusively placed into a single psublexicon? For now, this
        ## problem is ignored by only creating a psublexicon from "fully specified"
        ## lexemes, but a better solution should be found...
        ## TO-DO: add a function to infer the right psublexicon for non-fully-specified
        ## lexemes!
        ####

        psublexicons = defaultdict(dict)#{(base_cell,changes): {lexeme:[Words]}}
        for lexeme in lexeme_dict:
            basechanges = [c for c,w in lexeme_dict[lexeme]]
            if len(basechanges) == len(self.cells) - 1: # TO-DO: remove; see above; this line is the temp fix
                basechanges = tuple(sorted(basechanges, key=lambda c:c[0]))
                psublexicons[basechanges][lexeme] = [w for w in lexeme_dict[lexeme]]

        print('Number of paradigm sublexicons: {}'.format(len(psublexicons)))


        classy_psublexicons = []
        for ps in psublexicons:
            classy_psublexicons.append(PSublexicon(
                    deriv_cell=deriv_cell,
                    operations={cell: changes for cell, changes in ps},
                    lexemes=[lex for lex in psublexicons[ps]],
                    words=([word for lex in psublexicons[ps]
                           for change, word in psublexicons[ps][lex]])))
        psublexicons = classy_psublexicons

        psl_sizes = [sum([w.frequency for w in psl.words]) 
                     for psl in psublexicons]
        frequency_sum = sum(psl_sizes)
        for psl, size in zip(psublexicons, psl_sizes):
            psl.relative_size = float(size) / frequency_sum

        for ps in psublexicons:
            for other_ps in psublexicons:
                if ps != other_ps:
                    for word in other_ps.words:
                        ps.zero_frequency_words.append(word)

        return psublexicons


    def select_subset(self, sf_val_tuples):
        # print(sf_val_tuples)
        return [w for w in self.entries if w.check_values(sf_val_tuples)]

    def retrieve_lexeme(self, form, sf_val_tuples):
        with_correct_sfeatures = self.select_subset(sf_val_tuples)
        return [w.lexeme for w in with_correct_sfeatures if w.form == form][0]

    def create_cells(self):
        for w in self.wordlist:
            fval_tuple = tuple([(feature, w.sfeatures[feature]) for feature 
                         in w.sfeatures if feature != 'lexeme'])
            self.cells[fval_tuple][w.lexeme] = w



class Word:
    def __init__(self, form=None, lexeme=None, sfeatures={}, frequency=1.0):
        ## self.form is a space-spaced str
        self.form = form
        self.lexeme = lexeme
        self.cell = sfeatures
        self.frequency = frequency
        self.violations = None

    def __repr__(self):
        if type(self.cell) == dict:
            cell_repr = '\t'.join([key+': ' + self.cell[key] for key 
                  in self.cell])
        elif type(self.cell) == tuple:
            cell_repr = str(self.cell)
        return ('[{}] ({})\t'.format(self.form, self.frequency)
                + 'lexeme: {}\t'.format(self.lexeme)
                + cell_repr)

    def __str__(self):
        return repr(self)

    def check_values(self, sf_val_tuples):
        return all([self.cell[sf]==val for sf,val in sf_val_tuples])


class PSublexicon:
    """Paradigm sublexicon: a sublexicon for a particular derivative cell
    across all base cells
    """
    def __init__(self, deriv_cell=None, operations=[], 
                 lexemes=[], words=[]):
        self.deriv_cell = deriv_cell
        self.operations = operations # {base_cell: [Change, Change...]}
        self.lexemes = lexemes # Formerly: {lexeme: {base_cell: Word}}
        self.words = words
        self.zero_frequency_words = []
        self.grammar = None
        self.constraints = None
        self.violations = None
        self.tableau = {'dummy_ur': {}}
        self.weights = None

    def __repr__(self):
        # return str(self.operations)
        return str(self.lexemes)

    def __str__(self):
        return repr(self)

    def learn_grammar(self, constraints_filename):
        self.constraints = self.read_constraints_file(constraints_filename)
        self.gaussian_priors = {}
        self.weights = np.zeros(len(self.constraints))

        self.add_constraints_to_words()
        self.make_tableau_from_words()
        self.weights = phoment.learn_weights(self.weights, self.tableau)

        # self.save_tableau_to_file()                                           # TO-DO: enable, add parameters

        # temporary; for printing...
        labeled_weights = list(zip(self.constraints, self.weights))
        labeled_weights.sort(key=lambda x: x[1], reverse=True)
        print()
        print(self)
        for c,w in labeled_weights[:4]:
            print('{}\t{}'.format(c,w))

    def read_constraints_file(self, cfilename):
        constraints = [] # [(con_re, cell)...]
        with open(cfilename) as cfile:
            cfilereader = csv.reader(cfile, delimiter='\t')
            for line in cfilereader:
                if len(line) > 1:
                    con_str, cell_str = line
                    con_re = re.compile(con_str)
                    cell = tuple([tuple([feature.split(':')[0], 
                                 feature.split(':')[1]]) 
                                    for feature in cell_str.split(',')])
                    constraints.append((con_re, cell))
        return constraints

    def add_constraints_to_words(self):
        for w in self.words:
            w.violations = phoment.find_violations(w, self.constraints)

        for w in self.zero_frequency_words:
            w.violations = phoment.find_violations(w, self.constraints)

    def make_tableau_from_words(self):
        for w in self.words:
            entry = w.form
            entry_info = [w.frequency, w.violations, 0]
            self.tableau['dummy_ur'][entry] = entry_info

        for w in self.zero_frequency_words:
            entry = w.form
            entry_info = [0, w.violations, 0]
            self.tableau['dummy_ur'][entry] = entry_info

    def save_tableau_to_file(self):
        with open('tableau_{}.csv'.format(str(self)), 'w') as tabfile:
            # TO-DO: sort constraints by base cell
            tabfile.write('\t\t'+'\t'.join([str(c) for c in self.constraints])+'\n')
            tabfile.write('\t\t'+'\t'.join([str(w) for w in self.weights])+'\n')
            # TO-DO: also sort forms by base cell
            for w in self.words:
                line = '{}\t{}\t'.format(w.form, w.cell)
                line += '\t'.join([str(w.violations[i]) if i in w.violations 
                               else '0' for i,c in enumerate(self.constraints)])
                tabfile.write(line+'\n')
            tabfile.write('\n\n')
            for w in self.zero_frequency_words:
                line = '{}\t{}\t'.format(w.form, w.cell)
                line += '\t'.join([str(w.violations[i]) if i in w.violations 
                               else '0' for i,c in enumerate(self.constraints)])
                tabfile.write(line+'\n')
