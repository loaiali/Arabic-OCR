from collections import namedtuple
import itertools
import scipy.sparse
import scipy.misc
import numpy as np

epsSym = '٭'
startSym = 'ـسـ'
endSym = 'ـأـ'
backOffSym = 'ـجـ'


Arc = namedtuple(
    'arc', 'index source_state target_state input_label_indx output_label cost')
Token = namedtuple(
    'token', 'id prev_id arc_number model_score lm_score creation_timestep')


class FST:
    def __init__(self, fst_file: str, label_mapping: str = None):
        """
        This class encapsulates the loading and processing of WFST decoding graphs.

        :param fst_file: The text-format WFST representing the decoding graph.
        :param label_mapping: A mapping from acoustic model label strings to label indices, identical to one
        that was used when training the acoustic model.
        """
        self._arcs = []
        self._final = {}
        self._index2label = []
        self._label2index = {}

        self._load_indx_label_map(label_mapping)
        self._load_fst(fst_file)

    def _preprocess_activations(self, activations):
        """This function renormalizes acoustic model scores to be in a reasonable range.
            from 0 to 1 (probabilty) subraction logs is same as dividing
        """
        max_per_sample = np.max(activations, axis=1).reshape(
            (activations.shape[0], 1))
        return (activations - max_per_sample)

    def _load_indx_label_map(self, filename):
        """Read the index to label mapping file from disc.
        """
        with open(filename, encoding="utf8") as f:
            self._index2label = [x.rstrip() for x in f]

        self._label2index = {epsSym: -2, startSym: -1}
        for i, x in enumerate(self._index2label):
            self._label2index[x] = i
        self._index2label += [epsSym, startSym]

    def _load_fst(self, filename):
        """Read a text-format WFST decoding graph into memory.
        """
        self._final = {}  # final states
        self._arcs = []  # every arc in fst

        # this start-arc is where every token lives before the first frame of data
        # self._arcs.append(
        #     Arc(0, -1, 0, self._label2index[epsSym], epsSym, float(0)))

        # Read the FST into our self_.arcs list.
        # Specialized functions parse "final state" and "normal arc"
        # lines of the input file, keyed on the number of space-separated fields in the line.
        def process_final_state(fields):
            assert (len(fields) in (1, 2))
            # missing field means 0 cost
            self._final[int(fields[0])] = -float(fields[1]
                                                 if len(fields) > 1 else 0)

        def process_normal_arc(fields):
            assert (len(fields) in (4, 5))
            # state numbers -> integers
            src_state = int(fields[0])
            dst_state = int(fields[1])
            input_label_indx = self._label2index[fields[2]]
            output_label = fields[3]
            # missing field means 0 cost
            cost = float(fields[4] if len(fields) > 4 else 0)
            # save this arc
            self._arcs.append(Arc(len(self._arcs), src_state,
                                  dst_state, input_label_indx, output_label, cost))

        with open(filename, encoding="utf8") as f:
            for line in f:
                fields = line.rstrip().split()
                if len(fields) <= 2:
                    process_final_state(fields)
                else:
                    process_normal_arc(fields)

        # Pre-index all arcs coming out of a state to speed up transition-matrix creation.
        # each src state has a list of outgoing arcs idicies in _arcs list
        state_outgoing_arcs = [() for _ in range(
            1 + max(arc.source_state for arc in self._arcs))]
        for source_state, arcs in itertools.groupby(sorted(self._arcs, key=lambda arc: arc.source_state), key=lambda arc: arc.source_state):
            state_outgoing_arcs[source_state] = [arc.index for arc in arcs]

        # We encode the graph transition structure as three sparse matrices. Each i,j element represents a
        # cost for a path through the graph to transition from arc number i to arc number j. The log_score
        # matrix faithfully represents these scores, whereas the emit_trans and eps_trans matrices store the
        # scores processed through exp(). he emit_trans contains only nonzero rows that don't have epsilons on the corresponding arcs
        # and the eps_trans contains the rows that do.

        emit_row, emit_col, emit_val = [], [], []
        eps_row, eps_col, eps_val = [], [], []

        for arc in self._arcs:
            # if arc.input_label_indx >= 0:
            #     # non-epsilon implies zero-cost self-loop as it corresponed to a state in HMM
            #     emit_val.append(float(0))
            #     emit_col.append(arc.index)
            #     emit_row.append(arc.index)
            next_state = arc.target_state
            for next_arc_index in state_outgoing_arcs[next_state]:
                next_arc = self._arcs[next_arc_index]
                score = -next_arc[-1]  # to be replaced with .cost
                if next_arc.input_label_indx >= 0:
                    # non-epsilon row
                    emit_val.append(score)
                    emit_col.append(arc.index)
                    emit_row.append(next_arc_index)
                else:
                    # epsilon row
                    eps_val.append(score)
                    eps_col.append(arc.index)
                    eps_row.append(next_arc_index)

        # The linear transition score for arcs with emitting symbols
        self.emit_trans = scipy.sparse.csr_matrix(
            (np.exp(emit_val), (emit_row, emit_col)),
            shape=(len(self._arcs), len(self._arcs)),
            dtype=np.float32
        )
        # The linear transition score for arcs with non-emitting symbols
        self.eps_trans = scipy.sparse.csr_matrix(
            (np.exp(eps_val), (eps_row, eps_col)),
            shape=(len(self._arcs), len(self._arcs)),
            dtype=np.float32
        )
        # The log-score for all arcs
        self.log_score = scipy.sparse.csr_matrix(
            (emit_val + eps_val, (emit_row + eps_row, emit_col + eps_col)),
            shape=(len(self._arcs), len(self._arcs)),
            dtype=np.float32
        )

    def decode(self, decoder, activations, lmweight, emit_trans=None, eps_trans=None, all_trans=None):
        """
        Find the best path through the decoding graph, using a given set of acoustic model scores.
        """

        decoder.active_tokens = []
        decoder.set_intial_active_token(Token(
            id=0, prev_id=-1, arc_number=0, model_score=0., lm_score=self._arcs[0].cost, creation_timestep=-1))

        #  If the caller doesn't give us transition matrices, then use our own.
        if emit_trans is None:
            emit_trans = self.emit_trans
        if eps_trans is None:
            eps_trans = self.eps_trans
        if all_trans is None:
            all_trans = self.log_score

        #  Turn the given activations of the acoustic model scores into something useful
        model_scores = self._preprocess_activations(activations) / lmweight

        # Here is the core of the search algorithm. It loops over time, using the acoustic model scores
        for t, obs_vector in enumerate(model_scores):
            # print(t)
            arcs = [self._arcs[token.arc_number]
                    for token in decoder.active_tokens]

            lines = [
                f'{arc.index} {arc.source_state} {arc.target_state} {self._index2label[arc.input_label_indx]} {arc.output_label} {arc.cost}\n'
                for arc in arcs]
            # with open("active_arcs", "a") as aa:
            #     aa.writelines(lines)

            # if(t > 0):  # ignore first state
            decoder.commit_active_tokens()

            # replace old active tokens with new ones (expantion of frontier)
            decoder.active_tokens = decoder.do_forward(
                emit_trans, np.array(obs_vector).squeeze(), self._arcs, all_trans, timestep=t+1)

            decoder.beam_prune()  # prone new active tokens

          # Advance the tokens we've just created onto any arcs with epsilon input symbols they can reach.
            epsilon_tokens = []
            curr_active_tokens = decoder.active_tokens

            while len(curr_active_tokens) > 0:
                curr_active_tokens = decoder.do_forward(transition_matrix=eps_trans, active_tokens=curr_active_tokens,
                                                        arcs=self._arcs, all_trans=all_trans, timestep=t+1)
                epsilon_tokens += curr_active_tokens

            # Among the epsilon tokens we've just created, only keep the best for each arc.
            # this necessary as there is tokens each may take different socres and not handeled by the sparse matrix as that happens in different forwarding steps unlike the non epsilon tokens
            epsilon_tokens = [
                max(tokens, key=lambda token: token.model_score + token.lm_score)
                for arc_num, tokens in itertools.groupby(
                    sorted(epsilon_tokens, key=lambda token: token.arc_number),
                    key=lambda token: token.arc_number
                )
            ]

            # Ensure the tokens are sorted by ID. This invariant is used by the tok_backtrace member function.
            epsilon_tokens.sort(key=lambda token: token.id)
            decoder.active_tokens += epsilon_tokens

        decoder.commit_active_tokens()

        # apply final state scores
        dest_state = [
            self._arcs[x.arc_number].target_state for x in decoder.active_tokens]
        decoder.active_tokens = [
            Token(x.id, x.prev_id, x.arc_number,
                  x.model_score, x.lm_score + self._final[s], -1)
            for x, s in zip(decoder.active_tokens, dest_state) if s in self._final
        ]

        best_tok = max(decoder.active_tokens,
                       key=lambda x: x.model_score + x.lm_score)
        # print(
        #     "best cost: AM={} LM={} JOINT={}".format(
        #         best_tok.model_score, best_tok.lm_score, best_tok.model_score + best_tok.lm_score
        #     )
        # )

        # return best path
        hypothesis = map(lambda arc_with_frame_num: (self._index2label[arc_with_frame_num[0].input_label_indx], arc_with_frame_num[0].output_label, arc_with_frame_num[1]),
                         map(lambda arc_number_frame_num: (self._arcs[arc_number_frame_num[0]], arc_number_frame_num[1]), decoder.tok_backtrace()))
        words = [outlabel for _, outlabel, _ in hypothesis if outlabel not in [
            epsSym, startSym, endSym]]
        return words
