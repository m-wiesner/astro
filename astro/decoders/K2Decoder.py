import torch
import k2
from pathlib import Path
from .AbstractDecoder import AbstractDecoder
from astro.tokenizers.BaseTokenizer import BaseTokenizer
import torch.nn.functional as F


###############################################################################
#                       Graph init methods
#
# A collection of methods to initialize the decoding graph for K2
###############################################################################
class GraphInit(object):
    @staticmethod
    def T(vocab_size, **kwargs):
        '''
            :param lm: None in this case
            :param vocab_size: the number of bpe tokens including ctc_blank
            :return: decoding graph 
        '''
        graph = k2.ctc_topo(vocab_size - 1)
        return graph

    @staticmethod
    def TL(vocab_size, **kwargs):
        L_file = kwargs.get('L', None)
        word2id = kwargs.get('word2id', None)
        if None in (L_file, word2id):
            raise ValueError("Expected a lexicon FST, and symbol table when "
                "creating the TL graph."
            )

        disambig_id = word2id["#0"]
        uses_oovs = False
        for w in word2id:
            if w.startswith('@@') or w.startswith('\u2581'):
                uses_oovs = True
                break
        
        if kwargs.get('min_topo', False):
            T = GraphInit.T_min(vocab_size, **kwargs)
        else:
            T = k2.ctc_topo(vocab_size - 1)
        
        L = k2.Fsa.from_dict(torch.load(L_file))
        L = k2.arc_sort(L)
        L = k2.determinize(L)
        L = k2.connect(L)
        
        L.labels[L.labels >= vocab_size] = 0
        L.aux_labels.values[L.aux_labels.values >= disambig_id] = 0
        L.__dict__["_properties"] = None
        L = k2.remove_epsilon(L)
        L = k2.connect(L)
        L.aux_labels = L.aux_labels.remove_values_eq(0)
        L = k2.arc_sort(L)
        graph = k2.compose(T, L)
        graph = k2.connect(graph)
        graph = k2.arc_sort(graph)
        return graph

    
    @staticmethod
    def TLG(vocab_size, **kwargs):
        '''
            :param lm: Word LM as WFST 
            :param vocab_size: the number of bpe tokens including ctc_blasp2,    :return: decoding graph 
        '''
        L_file = kwargs.get('L', None)
        word2id = kwargs.get('word2id', None)
        G_file = kwargs.get('G', None)
        if None in (L_file, G_file, word2id):
            raise ValueError("Expected a lexicon Fst, Grammar FST and words"
                " symbol table when creating the TLG graph."
            )
      
        disambig_id = word2id["#0"]
        uses_oovs = False
        for w in word2id:
            if w.startswith('@@') or w.startswith('\u2581'):
                uses_oovs = True
                break

        # Create the CTC topology
        if kwargs.get('min_topo', False):
            T = GraphInit.T_min(vocab_size, **kwargs)  
        else:
            T = k2.ctc_topo(vocab_size - 1) # -1 since vocab_size include blank

        L = k2.Fsa.from_dict(torch.load(L_file)) 
        with open(G_file) as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        
        L = k2.arc_sort(L)
        G = k2.arc_sort(G)
        LG = k2.compose(L, G)
        LG = k2.connect(LG)
        
        LG = k2.determinize(LG)
        LG = k2.connect(LG)
       
        LG.labels[LG.labels >= vocab_size] = 0
        LG.__dict__["_properties"] = None
        LG.aux_labels.values[LG.aux_labels.values >= disambig_id] = 0
        LG = k2.remove_epsilon(LG)
        LG = k2.connect(LG)
        LG.aux_labels = LG.aux_labels.remove_values_eq(0)
        LG = k2.arc_sort(LG)
        graph = k2.compose(T, LG)
        graph = k2.connect(graph)
        graph = k2.arc_sort(graph)
        return graph  

###############################################################################
#                         K2 Decoder
#
# This class defines the K2 decoder. For now it supports decoding with 
# the ctc topology T, the composition of T with a BPE language model P,
# and finally, the composition with word-level language model G, via bpe,
# lexicon L. In all cases, we project onto bpe units so that each decoder type
# has the same output units.
###############################################################################
class K2Decoder(AbstractDecoder):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--k2-method', type=str, default='TLG', choices=['T', 'TL', 'TLG'])
        parser.add_argument('--k2-lm', type=str, default=None, help="The arpa LM file.")
        parser.add_argument('--k2-lexicon', type=str, default=None, help="L_disambig.pt")
        parser.add_argument('--k2-words', type=str, default=None, help="words.txt")  
        parser.add_argument('--k2-max-active', type=int, default=10000)
        parser.add_argument('--k2-min-active', type=int, default=20)
        parser.add_argument('--k2-search-beam', type=int, default=20)
        parser.add_argument('--k2-output-beam', type=int, default=8)
        parser.add_argument('--k2-acoustic-weight', type=float, default=3.0)
        parser.add_argument('--k2-blank-weight', type=float, default=1.0)
        parser.add_argument('--k2-min-topo', action='store_true')
        parser.add_argument('--k2-graph', type=str, default=None, help="path to graph. TLG.pt")

    @classmethod
    def from_args(cls, args):
        tokenizer = BaseTokenizer.load(args.tokenizer)  
        device = torch.device('cuda')
        return cls(
            tokenizer,
            lm=args.k2_lm, lexicon=args.k2_lexicon, words=args.k2_words,
            graph_method=getattr(GraphInit, args.k2_method),
            device=device,
            acoustic_weight=args.k2_acoustic_weight,
            blank_weight=args.k2_blank_weight,
            max_active=args.k2_max_active,
            min_active=args.k2_min_active,
            search_beam=args.k2_search_beam,
            output_beam=args.k2_output_beam,
            graph=args.k2_graph,
        )

    def __init__(self,
        tokenizer,
        lm=None, lexicon=None, words=None,
        graph_method=GraphInit.TLG,
        min_topo=False,
        device=torch.device('cpu'),
        acoustic_weight=1.0,
        blank_weight=0.0,
        max_active=200,
        min_active=30,
        search_beam=50,
        output_beam=8,
        graph=None,
    ):
        word2id = {}
        with open(words, encoding='utf-8') as f:
            for l in f:
                word, word_id = l.strip().split(None, 1)
                word2id[word] = int(word_id)
        options = {
            'G': lm,
            'L': lexicon,
            'word2id': word2id,
            'min_topo': min_topo,
        }

        self.graph_method = graph_method
        self.tokenizer = tokenizer
        if graph is not None and Path(graph).is_file():
            self.graph = k2.Fsa.from_dict(torch.load(graph, map_location=device))  
        else:
            self.graph = graph_method(tokenizer.vocab_size, **options).to(device)
        self.acoustic_weight = acoustic_weight
        self.blank_weight = blank_weight
        self.max_active = max_active
        self.min_active = min_active
        self.search_beam = search_beam
        self.output_beam = output_beam
        self.token_type = 'phone' if self.graph_method == GraphInit.T else 'word' 
         
    @torch.no_grad()
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def __call__(self, nnet_outputs, lengths=None):
        batch_size = nnet_outputs.size(0)
        T = nnet_outputs.size(1)
        sequence_idx = torch.arange(
            0, batch_size,
        ).unsqueeze(0).t().to(torch.int32)
   
        start_frame = torch.zeros(
            [batch_size], dtype=torch.int32,
        ).unsqueeze(0).t()

        if lengths is not None:
            num_frames = lengths.unsqueeze(-1).cpu().to(torch.int32)
        else:
            num_frames = T * torch.ones(
                [batch_size], dtype=torch.int32,
            ).unsqueeze(0).t().cpu().to(torch.int32)

        supervision_segments = torch.cat(
            [sequence_idx, start_frame, num_frames],
            dim=1,
        )
        supervision_segments = supervision_segments.to(torch.int32)

        nnet_outputs[..., 0] -= self.blank_weight # / self.acoustic_weight
        
        dense_fsa_vec = k2.DenseFsaVec(
            self.acoustic_weight * nnet_outputs, supervision_segments
        )
        
        lattices = k2.intersect_dense_pruned(
            self.graph,
            dense_fsa_vec,
            search_beam=self.search_beam,
            output_beam=self.output_beam,
            min_active_states=self.min_active,
            max_active_states=self.max_active,
            frame_idx_name='frame_idx',
        )
        
        lattices = k2.connect(lattices)
        lattices = k2.arc_sort(lattices)
        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        if isinstance(best_paths.aux_labels, torch.Tensor):
            aux_shape = best_paths.arcs.shape().remove_axis(1)
            aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels) 
            labels = aux_labels.remove_values_leq(0).tolist()
        else:
            aux_labels = best_paths.aux_labels.remove_values_leq(0) 
            aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)
            aux_shape = aux_shape.remove_axis(1)
            aux_shape = aux_shape.remove_axis(1)
            aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values) 
            labels = aux_labels.remove_values_leq(0).tolist()
        hypotheses = self.tokenizer.tokens_to_syms(
            labels, token_type=self.token_type
        )  
        
        return hypotheses
