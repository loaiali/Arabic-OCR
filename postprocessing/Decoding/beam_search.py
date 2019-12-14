from collections import namedtuple
from typing import List
import scipy.sparse
import scipy.misc
import numpy as np
import itertools


Token = namedtuple(
    'token', 'id prev_id arc_number model_score lm_score creation_timestep')


class BeamSearch:
    """This class encapsulates the token lifecycle, which makes the decoder class cleaner.
    """

    def __init__(self, beam_width):
        # the tokens array holds all of the tokens we've committed to the search
        self.tokens = []

        # set beam_witdht
        self.beam_width = beam_width

    def set_intial_active_token(self, token):
        self.active_tokens = [token]
        self.last_token_id = self.active_tokens[0].id

    def token_list_to_sparse(self, num_arcs: int, token_list: List[Token]):
        """This helper function takes a list of tokens, and turns it into an equivalent set of sparse structures.
        """

        # token's score
        # the am score and lm score for an active arc is the path score from the begining of serach
        tokens_scores = np.array(
            [token.model_score + token.lm_score for token in token_list], dtype=np.float32)
        # limit scores between 0->1 this very important as pathcost will weight the next transition cost limmiting numbers betwee 0->1 will prevent overflow and maintain relative difference
        tokens_scores = np.exp(tokens_scores - np.max(tokens_scores))
        # make a column vector; row index is arc number
        rows = np.array(
            [token.arc_number for token in token_list], dtype=np.int)
        cols = np.zeros(rows.shape)
        scores = scipy.sparse.csc_matrix(
            (tokens_scores, (rows, cols)),
            shape=(num_arcs, 1),
            dtype=np.float32
        )

        # try to change to dict mappng the arc number to token directly ____!!!!!
        score_index_to_token_index = np.ones(
            num_arcs, dtype=np.int32) * -1  # bogus initialization
        for i, token in enumerate(token_list):
            score_index_to_token_index[token.arc_number] = i

        return scores, score_index_to_token_index

    def tok_backtrace(self, looking_for_token_id=None):
        """This function finds the best path described by the tokens created so far.
        """

        if looking_for_token_id is None:
            looking_for_token_id = max(
                self.active_tokens, key=lambda x: x.model_score + x.lm_score).id

        path = []
        # search backward through tokens
        for token in (self.tokens + self.active_tokens)[::-1]:
            if token.id == looking_for_token_id:
                arc_number = token.arc_number
                path.append((arc_number, token.creation_timestep))
                looking_for_token_id = token.prev_id
        # reverse backtrace so tokens are in forward-time-order
        path = path[::-1]

        # Combine sequences of identical arcs into one representative arc_number removing self loops
        return [list(x)[-1] for k, x in itertools.groupby(path, lambda e:e[0])]

    def commit_active_tokens(self):
        self.tokens += self.active_tokens

    def beam_prune(self):
        if len(self.active_tokens) > self.beam_width:
            self.active_tokens = sorted(
                self.active_tokens, key=lambda x: x.model_score + x.lm_score, reverse=True)[0:self.beam_width]

    def advance_token(self, prev_token: Token, new_token_arc, model_score, lm_score, creation_timestep):
        self.last_token_id += 1
        return Token(
            self.last_token_id,
            prev_token.id,
            new_token_arc,
            prev_token.model_score + model_score,
            prev_token.lm_score + lm_score,
            creation_timestep=creation_timestep
        )

    def do_forward(self, transition_matrix, obs_vector=None, arcs=None, all_trans=None, active_tokens=None, timestep=-1):
        """Implements the search-update algorithm using sparse matrix-vector primitives
        """
        active_tokens = active_tokens if active_tokens is not None else self.active_tokens

        #  Convert the token list into a sparse structure
        src_scores, src_tokens_index = self.token_list_to_sparse(
            len(arcs), active_tokens)

        # Project the tokens forward through the given transition matrix. Note that this is not
        # a matrix multiplication, but an application of the previous tokens to the columns of the
        # transition matrix. The (i,j) element in the resulting two-dimensional structure represents
        # the score assocaited with creating a new token on arc i, from an old token on arc j.
        trans = transition_matrix.multiply(src_scores.T)

        # Convert the sparse trans matrix into two obects: row_to_column, which for every row of
        # trans, indicates which column had the best score, and active_rows, which is the set
        # of rows in trans with non-zero entries. This tells us which tokens will be created
        # (active_rows), and who their best predecessor is (row_to_column).
        row_to_column = np.array(trans.argmax(axis=1)).squeeze()
        # nonzero return the rows indices that have maximums but not zero (paired with useless array so we use [0] to discard that array)
        active_rows = trans.max(axis=1).nonzero()[0]

        # Create a complete set of new tokens and return it.
        return [
            self.advance_token(
                active_tokens[src_tokens_index[row_to_column[active_row]]],
                active_row,
                obs_vector[arcs[active_row].input_label_indx] if obs_vector is not None else 0,
                all_trans[active_row, row_to_column[active_row]],
                timestep
            )
            for active_row in active_rows]
