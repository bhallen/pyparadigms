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

from collections import defaultdict

PRE_REDUCTION_CUTOFF = None


class Lexicon:
    def __init__(self):
        self.sfeatures = []
        self.wordlist = []
        self.cells = defaultdict(dict)
        self.lexemes = []
        self.mappings = defaultdict(dict) # morphological operations from b to d
        self.grammars = {c: None for c in self.cells}
        self.aligner = None
        self.psublexicons = []

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
            self.psublexicons = self.find_psublexicons(
                                                      self.mappings[deriv_cell],
                                                      deriv_cell)





    def save_lexicon(self, language_name):
        pass

    def predict(self, language_name):
        pass

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

                try:
                    ## Skim off alignments with bad scores
                    all_alignments += alignments[:MAXALIGNMENTS]                #TO-DO: add param for MAXALIGNMENTS
                except NameError:
                    all_alignments += alignments

        print('Number of alignments: {}'.format(str(len(all_alignments))))

        print('Reducing hypotheses...')
        bsublexicons = hypothesize.create_and_reduce_hypotheses(
                     all_alignments, PRE_REDUCTION_CUTOFF, orientation='source')
        print('Hypotheses have been reduced.')

        ## Should NOT add zero frequency forms since associated words are used
        ## to determine paradigm sublexicons
        print('Number of base sublexicons: {}'.format(str(len(bsublexicons))))

        return bsublexicons


    def find_psublexicons(self, bsublex_dict, deriv_cell):
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

        psublexicons = defaultdict(dict)#{(base_cell,changes): {lexeme:[Words]}}
        for lexeme in lexeme_dict:
            basechanges = [c for c,w in lexeme_dict[lexeme]]
            basechanges = tuple(sorted(basechanges, key=lambda c:c[0]))
            psublexicons[basechanges][lexeme] = [w for w in lexeme_dict[lexeme]]

        print('Number of paradigm sublexicons: {}'.format(len(psublexicons)))
        output_psublexicons = []
        for ps in psublexicons:
            output_psublexicons.append(PSublexicon(
                                    deriv_cell=deriv_cell,
                                    operations=ps,
                                    lexemes=[lex for lex in psublexicons[ps]],
                                    words=[psublexicons[ps][lex] for lex in psublexicons[ps]]))


        print(output_psublexicons)


    def select_subset(self, sf_val_tuples):
        # print(sf_val_tuples)
        return [w for w in self.entries if w.check_values(sf_val_tuples)]

    def retrieve_lexeme(self, form, sf_val_tuples):
        with_correct_sfeatures = self.select_subset(sf_val_tuples)
        # print(with_correct_sfeatures)
        # print(sf_val_tuples)
        # print(form)
        # print([e['lexeme'] for e in with_correct_sfeatures if e['form'] == form])
        return [w.lexeme for w in with_correct_sfeatures if w.form == form][0]

    def create_cells(self):
        for w in self.wordlist:
            fval_tuple = tuple([(feature, w.sfeatures[feature]) for feature 
                         in w.sfeatures if feature != 'lexeme'])
            self.cells[fval_tuple][w.lexeme] = w

    def create_gbr_features(self):
        for base_c in self.cells:
            # self.gbr_features.append((base_c,))
            for deriv_c in self.cells:
                if base_c != deriv_c:
                    self.gbr_features.append((base_c,deriv_c))
        self.gbr_features.sort()


class Word:
    def __init__(self, form=None, lexeme=None, sfeatures={}, frequency=1.0):
        ## self.form is a space-spaced str
        self.form = form
        self.lexeme = lexeme
        self.sfeatures = sfeatures
        self.frequency = frequency

    def __repr__(self):
        if type(self.sfeatures) == dict:
            sfeatures_repr = '\t'.join([key+': ' + self.sfeatures[key] for key 
                  in self.sfeatures])
        elif type(self.sfeatures) == tuple:
            sfeatures_repr = str(self.sfeatures)
        return ('[{}] ({})\n'.format(self.form, self.frequency)
                + 'lexeme: {}\n'.format(self.lexeme)
                + sfeatures_repr + '\n')

    def __str__(self):
        return repr(self)

    def check_values(self, sf_val_tuples):
        return all([self.sfeatures[sf]==val for sf,val in sf_val_tuples])


class PSublexicon:
    """Paradigm sublexicon: a sublexicon for a particular derivative cell
    across all base cells
    """
    def __init__(self, deriv_cell=None, operations=[], 
                 lexemes=[], words=[]):
        self.deriv_cell = deriv_cell
        self.operations = operations # Formerly: {base_cell: [Change, Change...]}
        self.lexemes = lexemes # Formerly: {lexeme: {base_cell: Word}}
        self.words = words

    def __repr__(self):
        # return str(self.operations)
        return str(self.lexemes)

    def __str__(self):
        return repr(self)


def test_predictions(testing_dict, gbr_features, con_weights):
    weight_dict = {feature: weight for (feature,weight) in zip(gbr_features,con_weights)}
    with open('predictions_test.txt', 'w') as testout:
        outstr = 'CELL\tLEXEME\tOUTPUT\tOBS\tPRED\n'
        for item in testing_dict:
            outstr += '\t'.join([str(item[0]),item[1]]) + '\t'
            weighted_prob_sums = []
            obs_probs = []
            for output in testing_dict[item]:
                obs_probs.append(testing_dict[item][output]['observed'])
                weighted_prob_sum = 0
                for feature in weight_dict:
                    if feature != (item[0],):
                        weighted_prob_sum += testing_dict[item][output][feature]*weight_dict[feature] # cond.prob * weight
                weighted_prob_sums.append(weighted_prob_sum)
            Z = sum(weighted_prob_sums)
            marginal_probs = [wps/Z for wps in weighted_prob_sums]
            marginal_probs = zip([output for output in testing_dict[item]],obs_probs,marginal_probs) # FINAL element is predicted prob!
            marginal_probs.sort(key=lambda tup: tup[2], reverse=True)

            outstr += '\n\t\t'.join([output+'\t'+str(obs)+'\t'+str(pred) for output,obs,pred in marginal_probs]) + '\n'

        testout.write(outstr)