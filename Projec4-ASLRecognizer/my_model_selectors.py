import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ 
        select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def bic_score(self, n_components):
        """
        Calculate BIC score
        """
        model = self.base_model(n_components)
        logL = model.score(self.X, self.lengths)
        logN = np.log(len(self.X))

        # p = = n^2 + 2*d*n - 1
        d = self.max_n_components + 1
        p = n_components ** 2 + 2 * d * n_components - 1

        return -2.0 * logL + p * logN, model


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = 0
            for n in range(self.min_n_components, self.max_n_components + 1):
                if self.bic_score(n) > best_score:
                    best_score, model = self.bic_score(n)
            return model

        except Exception as e:
            return self.base_model(self.n_constant)



class SelectorDIC(ModelSelector):
    """ select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """

    def dic_score(self, n_components):
        """
        Calculate DIC score
        """
        model = self.base_model(n_components)

        logP = model.score(self.X, self.lengths)
        M = len(logP)

        sum_log_p = 0
        for word, (X, l) in self.hwords.items():
            if word != self.this_word:
                sum_log_p = sum_log_p + model.score(X, l)

        return logP - 1/(M-1)*sum_log_p 

    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = 0
            for n in range(self.min_n_components, self.max_n_components + 1):
                if self.dic_score(n) > best_score:
                    best_score, model = self.dic_score(n)
            return model

        except Exception as e:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def cv_score(self, n_components):
        scores = []
        split_method = KFold(2)

        for train_idx, test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_idx, test_idx)

            model = self.base_model(n_components)
            X, l = combine_sequences(test_idx, self.sequences)
            
            # Add the new score to the scores list
            scores.append(model.score(X, l))

        return np.mean(scores)

    def select(self):
        """
            select the best model based on CV score.
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = 0
            for n in range(self.min_n_components, self.max_n_components + 1):
                if self.cv_score(n) > best_score:
                    best_score, model = self.cv_score(n)
            return model

        except Exception as e:
            return self.base_model(self.n_constant)

