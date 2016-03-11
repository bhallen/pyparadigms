import aligner
import hypothesize
import phoment

import csv
import re
import numpy as np
from collections import defaultdict
import itertools
import pickle
import random

import pdb


class Lexicon:                                                                  #TO-DO: consider renaming InflectionalSystem?
    def __init__(self):
        self.sfeatures = []
        self.wordlist = []
        self.cells = defaultdict(dict)
        self.lexemes = []
        self.mappings = defaultdict(dict) # morphological operations from b to d
        self.aligner = None
        self.psublexicons = {} # {deriv_cell: [PSublexicons]}
        self.test_data = None
        self.test_predictions = None
        self.relative_size = None
        self.change_orientation = 'source'                                      #TO-DO: add param for this

    def read_training_file(self, training_filename, form_to_withhold):
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
                    if form_to_withhold:
                        if self.should_be_withheld(word, form_to_withhold):
                            continue
                    self.wordlist.append(word)
                    if word.lexeme not in self.lexemes:
                        self.lexemes.append(word.lexeme)

    def should_be_withheld(self, word, lexeme_and_cell):
        if word.lexeme == lexeme_and_cell[0]:
            for sfeature, value in lexeme_and_cell[1]:
                if word.sfeatures[sfeature] == value:
                    continue
                else:
                    return False
        else:
            return False
        return True


    def learn(self, training_filename, constraints_filename, features_filename,
           just_return_cells=False, form_to_withhold=None, skip_grammars=False,
           pre_reduction_cutoff=None, post_reduction_cutoff=None,
           psublexicon_size_cutoff=None):
        print('Initializing aligner...')
        self.aligner = aligner.Aligner(feature_file=features_filename,
                                   sub_penalty=4.0, tolerance=1.0)              #TO-DO: Remove these magic numbers

        print('Loading training file...')
        self.read_training_file(training_filename, form_to_withhold)


        ## Create dict of *OBSERVED* MPS tuples and their forms
        print('Compiling information about available cells...')
        self.create_cells()

        if just_return_cells:
            return [c for c in self.cells]

        for i, deriv_cell in enumerate(self.cells):
            print('Learning about the derivative cell {}...'
                  .format(deriv_cell))
            for j, base_cell in enumerate(self.cells):
                if base_cell != deriv_cell:
                    print('Learning base sublexicons from {} ({} of {}) to {} ({} of {})...'
                           .format(base_cell, j+1, len(self.cells), 
                            deriv_cell, i+1, len(self.cells)))
                    self.mappings[deriv_cell][base_cell] = \
                           self.find_bd_sublexicons(base_cell, deriv_cell,
                                                    pre_reduction_cutoff,
                                                    post_reduction_cutoff)

            print('Finding this derivative cell\'s paradigm sublexicons...')
            this_deriv_psublexicons = self.find_psublexicons(
                                                      self.mappings[deriv_cell],
                                                      deriv_cell,
                                                      psublexicon_size_cutoff)
            print('Number of paradigm sublexicons: {}'.format(len(this_deriv_psublexicons)))

            if skip_grammars:
                self.psublexicons[deriv_cell] = this_deriv_psublexicons
            else:
                print('Learning a MaxEnt HG for each paradigm sublexicon...')
                for k, ps in enumerate(this_deriv_psublexicons):
                    print('Learning a grammar for paradigm sublexicon {} of {}...'.format(
                            k+1, len(this_deriv_psublexicons)))
                    ps.learn_grammar(constraints_filename)

                self.psublexicons[deriv_cell] = this_deriv_psublexicons






    def save_lexicon(self, language_name, suffix=''):
        if suffix:
            suffix = '_{}'.format(suffix)
        with open('datasets/{}/{}{}.lexicon'.format(
                          language_name, language_name, suffix
                          ), 'wb') as lexfile:
            pickle.dump(self, lexfile)    


    def predict_from_testing_data(self, language_name):
        self.read_testing_file(language_name)
        self.complete_test_data()

    def predict_single_word(self, word_tuple):
        """Tuple (lexeme, cell)
        """
        givens = {} # needs to be {cell: {form: probability}}
        for cell in self.cells:
            for lexeme in self.cells[cell]:
                if lexeme == word_tuple[0]:
                    word = self.cells[cell][lexeme]
                    givens[cell] = {word.form: word.frequency}
        return self.predict_candidate_distribution(givens, word_tuple[1])

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
        print(self.test_predictions)


    def predict_candidate_distribution(self, givens, deriv_cell):
        psublexicons = self.psublexicons[deriv_cell]
        candidates = [] # one per psublexicon
        ps_probs = [] # one per psublexicon: p(psublexicon|bases)
        for ps in psublexicons:
            ## generate a candidate
            arbitrary_base_cell = list(givens)[0]
            if arbitrary_base_cell == deriv_cell:
                arbitrary_base_cell = list(givens)[1]
            if type(givens[arbitrary_base_cell]) == str:
                base_form = givens[arbitrary_base_cell].split(' ')
            else:
                base_form = list(givens[arbitrary_base_cell])[0].split(' ') #TO-DO: remove this assumption that there's only one form with a particular meaning in the test set
            operation = ps.operations[arbitrary_base_cell]
            candidate = hypothesize.apply_operation(base_form, operation,
                                                    self.change_orientation)
            ## assign p(bases|ps)
            if candidate != 'incompatible':
                bases_prob = self.get_bases_prob(givens, ps)

                ## p(ps|bases) \propto bases_prob * ps.relative_size
                candidates.append(candidate)
                ps_probs.append(bases_prob * ps.relative_size)

        ## normalize psublexicon probabilities
        ps_probs = np.array(ps_probs)
        ps_probs = ps_probs / ps_probs.sum()

        ## add probabilities of same-candidate psublexicons
        unique_ps_probs = defaultdict(float)
        for candidate, ps_prob in zip(candidates, ps_probs):
            unique_ps_probs[candidate] += ps_prob

        return unique_ps_probs



    def get_bases_prob(self, givens, ps):
        """Calculate p(givens|ps)
        """
        return phoment.testing_bases_probability(givens, ps)


    def save_predictions(self, language_name):
        with open('datasets/{}/{}_predictions.txt'.format(language_name,
                                          language_name), 'w') as testout:
            testout.write('lexeme\t'+'\t'.join([str(cell) for cell in self.cells])+'\n')
            for lexeme in self.test_predictions:
                testout.write(lexeme)
                for cell in self.cells:
                    testout.write('\t')
                    if cell in self.test_predictions[lexeme]:
                        candidates = self.test_predictions[lexeme][cell]
                        ordered_cands = [(c,candidates[c]) for c in candidates]
                        ordered_cands.sort(key=lambda x: x[1], reverse=True)
                        ordered_cands = [c for c in ordered_cands if c[1] > 0.0001]
                        cands_str = ', '.join(['{} ({})'.format(c[0], c[1]) for c in ordered_cands])
                        testout.write(cands_str)
                testout.write('\n')




    def find_bd_sublexicons(self, base_cell, deriv_cell, pre_reduction_cutoff,
                                                        post_reduction_cutoff):
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


                all_alignments += alignments[:2]                                #TO-DO: only add up to MAXALIGNMENTS instead of 2

        print('Number of alignments: {}'.format(str(len(all_alignments))))

        print('Reducing hypotheses...')
        bsublexicons = hypothesize.create_and_reduce_hypotheses(
                     all_alignments, pre_reduction_cutoff,
                     orientation=self.change_orientation)
        print('Hypotheses have been reduced.')

        ## Should NOT add zero frequency forms yetsince associated words are
        ## used to determine paradigm sublexicons
        print('Number of base sublexicons: {}'.format(str(len(bsublexicons))))

        bsublexicons.sort(key=lambda h: h.total_probability, reverse=True)

        if post_reduction_cutoff:
            bsublexicons = ([bs for bs in bsublexicons
                            if bs.total_probability > post_reduction_cutoff])

        return bsublexicons


    def find_psublexicons(self, bsublex_dict, deriv_cell,
                          psublexicon_size_cutoff):
        """This function first organizes lexemes by their conjunctions of
        base sublexicon operations, and then creates a PSublexicon for each
        unique combination of base sublexicon operation conjunctions.
        """
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

        classy_psublexicons = []
        for ps in psublexicons:
            classy_psublexicons.append(PSublexicon(
                    deriv_cell=deriv_cell,
                    operations={cell: changes for cell, changes in ps},
                    lexemes=[lex for lex in psublexicons[ps]],
                    words=([word for lex in psublexicons[ps]
                           for change, word in psublexicons[ps][lex]])))
        psublexicons = classy_psublexicons

        ## commented out: version that uses token frequencies
        # psl_sizes = [sum([w.frequency for w in psl.words])
        #              for psl in psublexicons]
        psl_sizes = [sum([1.0 for w in psl.words if w.frequency > 0.0])
                     for psl in psublexicons]
        frequency_sum = sum(psl_sizes)
        for psl, size in zip(psublexicons, psl_sizes):
            psl.relative_size = float(size) / frequency_sum

        for ps in psublexicons:
            for other_ps in psublexicons:
                if ps != other_ps:
                    for word in other_ps.words:
                        ps.zero_frequency_words.append(word)

        if psublexicon_size_cutoff:
            psublexicons = ([ps for ps in psublexicons
                            if len(ps.lexemes) > psublexicon_size_cutoff])

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

    def evaluate_base_informativity(self):
        results = {} # best base has lowest cross-entropy sum
        # {base: (sum of cross-entropies, (.25 percentile, median, .75 prcntl))}

        for base in self.psublexicons:
            print('Evaluating base {}'.format(base))
            # find cross-entropy of observed with that cell as the only base
            cross_entropies = []
            lex_ct = 0
            for lexeme in self.lexemes:
                lex_ct += 1
                if not lex_ct % 500:
                    print('On lexeme {} out of {} (alerting every 500)'.format(
                            lex_ct, len(self.lexemes)))
                for deriv in self.psublexicons:
                    if deriv != base:
                        if base in self.cells and lexeme in self.cells[base]:
                            givens = {base: self.cells[base][lexeme].form}
                            d = self.predict_candidate_distribution(givens, deriv)
                            # print(lexeme)
                            # print('True form: {}'.format(self.cells[deriv][lexeme].form))
                            # print(d)
                            # pdb.set_trace()
                            if lexeme in self.cells[deriv] and self.cells[deriv][lexeme].form in d:
                                cross_entropies.append(d[self.cells[deriv][lexeme].form])
                            else: cross_entropies.append(0)

            mean = np.mean(cross_entropies)
            p25 = np.percentile(cross_entropies, 25)
            p50 = np.percentile(cross_entropies, 50)
            p75 = np.percentile(cross_entropies, 75)
            results[base] = (mean, (p25, p50, p75))

        return results

    def inspect_graph(self):
        pass



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
        self.constraints = None
        self.violations = None
        self.tableau = {'dummy_ur': {}}
        self.weights = None

    def __repr__(self):
        # return str(self.operations)
        return str(self.lexemes)

    def __str__(self):
        return repr(self)

    def learn_grammar(self, constraints_filename, word_to_withhold=None):
        self.tableau = {'dummy_ur': {}} # re-initialize tableau for CV
        self.constraints = self.read_constraints_file(constraints_filename)
        self.gaussian_priors = {}
        self.weights = np.zeros(len(self.constraints))

        self.add_constraints_to_words()
        self.make_tableau_from_words(word_to_withhold)
        self.weights = phoment.learn_weights(self.weights, self.tableau)

        # self.save_tableau_to_file()                                           # TO-DO: enable, add parameters

        # temporary; for printing...
        labeled_weights = list(zip(self.constraints, self.weights))
        labeled_weights.sort(key=lambda x: x[1], reverse=True)
        # print()
        # print(self)
        # for c,w in labeled_weights[:4]:
        #     print('{}\t{}'.format(c,w))

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

    def make_tableau_from_words(self, word_to_withhold=None):
        for w in self.words:
            if not word_to_withhold or not (word_to_withhold[0] == w.lexeme and
                                            word_to_withhold[1] == w.cell):
                entry = w.form
                entry_info = [w.frequency, w.violations, 0]
                self.tableau['dummy_ur'][entry] = entry_info

        for w in self.zero_frequency_words:
            if not word_to_withhold or not (word_to_withhold[0] == w.lexeme and
                    word_to_withhold[1] == w.cell):
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
