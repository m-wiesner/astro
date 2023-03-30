import k2
from astro.tokenizers.LexiconTokenizer import LexiconTokenizer
from collections import defaultdict


class Lexicon(object):
    @classmethod
    def from_file(cls, filename):
        lexicon = Lexicon.load_from_file(filename)
        return cls(lexicon)

    @classmethod
    def from_tokenizer(cls, tokenizer):
        tokenizer = LexiconTokenizer.load(tokenizer)
        lexicon = []
        for w in tokenizer.lexicon:
            for pron in tokenizer.lexicon[w]:
                lexicon.append((w, list(pron)))
        return cls(lexicon)

    @staticmethod
    def load_from_file(filename):
        lexicon = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                word, pron = l.strip().split(None, 1)
                lexicon.append((word, pron.split()))
        #lexicon.append((oov, [oov]))
        return lexicon
 
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.token2id = {p: i for i, p in enumerate(self.phoneset)}
        vocab = sorted(set([w for w, p in lexicon]))
        word_syms = '\n'.join(f'{w} {i}' for i, w in enumerate(vocab, 1))
        word_syms = k2.SymbolTable.from_str(word_syms)
        word_syms.add("#0")
        word_syms.add("<s>")
        word_syms.add("</s>")
        self.word2id = word_syms

    @property
    def phoneset(self):
        phoneset = set()
        for w, pron in self.lexicon:
            for p in pron:
                phoneset.add(p)
        return ['<eps>'] + sorted(phoneset) + ['#0'] 

    def to_fst_no_sil(self, need_self_loops=False):
        """Convert a lexicon to an FST (in k2 format).
        Args:
          need_self_loops:
            If True, add self-loop to states with non-epsilon output symbols
            on at least one arc out of the state. The input label for this
            self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
        Returns:
          Return an instance of `k2.Fsa` representing the given lexicon.
        """
        loop_state = 0  # words enter and leave from here
        next_state = 1  # the next un-allocated state, will be incremented as we go

        arcs = []

        # The blank symbol <blk> is defined in local/train_bpe_model.py
        assert self.word2id["<eps>"] == 0

        eps = 0

        for word_idx, (word, pieces) in enumerate(self.lexicon):
            assert len(pieces) > 0, f"{word} has no pronunciations"
            cur_state = loop_state

            word = self.word2id[word]
            pieces = [self.token2id[i] for i in pieces]
        
            for i in range(len(pieces) - 1):
                w = word if i == 0 else eps
                score = 0
                arcs.append([cur_state, next_state, pieces[i], w, score])
                cur_state = next_state
                next_state += 1

            # now for the last piece of this word
            i = len(pieces) - 1
            w = word if i == 0 else eps
            score = 0
            arcs.append([cur_state, loop_state, pieces[i], w, score])
        
        if need_self_loops:
            disambig_token = self.token2id["#0"]
            disambig_word = self.word2id["#0"]
            arcs = self.add_self_loops(
                arcs,
                disambig_token=disambig_token,
                disambig_word=disambig_word,
            )

        final_state = next_state
        arcs.append([loop_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)
        fsa = k2.Fsa.from_str(arcs, acceptor=False)
        return fsa

    def add_self_loops(self, arcs, disambig_token, disambig_word):
        """Adds self-loops to states of an FST to propagate disambiguation symbols
        through it. They are added on each state with non-epsilon output symbols
        on at least one arc out of the state.
        See also fstaddselfloops.pl from Kaldi. One difference is that
        Kaldi uses OpenFst style FSTs and it has multiple final states.
        This function uses k2 style FSTs and it does not need to add self-loops
        to the final state.
        The input label of a self-loop is `disambig_token`, while the output
        label is `disambig_word`.
        Args:
          arcs:
            A list-of-list. The sublist contains
            `[src_state, dest_state, label, aux_label, score]`
          disambig_token:
            It is the token ID of the symbol `#0`.
          disambig_word:
            It is the word ID of the symbol `#0`.
        Return:
          Return new `arcs` containing self-loops.
        """
        states_needs_self_loops = set()
        for arc in arcs:
            src, dst, ilabel, olabel, score = arc
            if olabel != 0:
                states_needs_self_loops.add(src)

        ans = []
        for s in states_needs_self_loops:
            ans.append([s, s, disambig_token, disambig_word, 0])

        return arcs + ans

    def add_disambig_symbols(self):
        """It adds pseudo-token disambiguation symbols #1, #2 and so on
        at the ends of tokens to ensure that all pronunciations are different,
        and that none is a prefix of another.
        See also add_lex_disambig.pl from kaldi.
        """

        # (1) Work out the count of each token-sequence in the
        # lexicon.
        count = defaultdict(int)
        for _, tokens in self.lexicon:
            count[" ".join(tokens)] += 1

        # (2) For each left sub-sequence of each token-sequence, note down
        # that it exists (for identifying prefixes of longer strings). We don't
        # need to look at subsequences of the bpe symbols we are adding as words
        # since they are each only 1 token long.
        issubseq = defaultdict(int)
        for _, tokens in self.lexicon:
            tokens = tokens.copy()
            tokens.pop()
            while tokens:
                issubseq[" ".join(tokens)] = 1
                tokens.pop()

        # (3) For each entry in the lexicon:
        # if the token sequence is unique and is not a
        # prefix of another word, no disambig symbol.
        # Else output #1, or #2, #3, ... if the same token-seq
        # has already been assigned a disambig symbol.
        ans = {'lexicon': []}

        # We start with #1 since #0 has its own purpose
        first_allowed_disambig = 1
        max_disambig = first_allowed_disambig - 1
        last_used_disambig_symbol_of = defaultdict(int)

        for word, tokens in self.lexicon:
            tokenseq = " ".join(tokens)
            assert tokenseq != ""
            if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
                ans['lexicon'].append((word, tokens))
                continue

            cur_disambig = last_used_disambig_symbol_of[tokenseq]
            if cur_disambig == 0:
                cur_disambig = first_allowed_disambig
            else:
                cur_disambig += 1

            if cur_disambig > max_disambig:
                max_disambig = cur_disambig
            last_used_disambig_symbol_of[tokenseq] = cur_disambig
            tokenseq += f" #{cur_disambig}"
            ans['lexicon'].append((word, tokenseq.split()))
        
        self.lexicon = ans['lexicon']
        next_token_id = max(self.token2id.values()) + 1
        for i in range(1, max_disambig + 1):
            disambig = f"#{i}"
            assert disambig not in self.token2id
            self.token2id[disambig] = next_token_id
            next_token_id += 1
 
