import torch
import re


###############################################################################
#                              BPE tokenizer
#
# This defines the tokenizer + collation used for the ASRDataset class above.
# We use the sentencepiece module to handle tokenization.
###############################################################################
class BPEEnglishMandarinTokenizer(object):
    '''
       Defines the collating and tokenization for ASR training using a BPE
       model using lhotse cuts 
    '''
    def __init__(
        self,
        bpe_english,
        bpe_mandarin,
        use_lang_tags=False,
    ):
        '''
            Inputs:
                :bpe: The sentencepiece model (bpe tokenizer) to use
                :return: BPE Collater object
        '''
        self.bpe_english = bpe_english # The BPE (sentencepiece model)
        self.bpe_mandarin = bpe_mandarin
        self.lang_tags = use_lang_tags

    def __call__(self, cuts: CutSet):
        # Encode the text
        token_sequences = [
            self.prepare_transcript(sup.text.replace("<unk>", "[unk]").replace("<noise>", "<noise> ").replace("  ", " "))
            for cut in cuts for sup in cut.supervisions
        ]
        
        # Get the max length in the batch (needed for padding).
        max_len = len(max([t[0] for t in token_sequences], key=len))
        
        # Get the padded sequences, all of the same length
        seqs_eng = [
            seq[0] + [self.bpe_english.pad_id()] * (max_len - len(seq[0]))
            for seq in token_sequences
        ]
        
        seqs_man = [
            seq[1] + [self.bpe_mandarin.pad_id()] * (max_len - len(seq[1]))
            for seq in token_sequences
        ]

        seqs_bil = [
            seq[2] + [self.bpe_english.pad_id()] * (max_len - len(seq[2]))
            for seq in token_sequences
        ]

        # Create a tensor and return it along with the sequence lengths
        tokens_batch_eng = torch.LongTensor(seqs_eng)
        tokens_batch_man = torch.LongTensor(seqs_man)
        tokens_batch_bil = torch.LongTensor(seqs_bil) 
        tokens_lens = torch.IntTensor([len(seq[0]) for seq in token_sequences])
        return [tokens_batch_eng, tokens_batch_man, tokens_batch_bil], tokens_lens


    def prepare_transcript(self, text):
        '''
            This is a special case for Mandarin and English. I first need
            to split on all script changes. I then need to apply the
            appropriate BPE tokenizer to each part of the script, adding
            language tags before each group of words in a specific language,
            and possibly a mask tag.

            Mandarin: <eid> <em> <mid> SOME_MANDARIN_TEXT <eid> <em>
            English: <eid> hello <mid> <mm><mm><mm><mm><mm><mm> <eid> Matthew
        '''

        english_transcript, mandarin_transcript, bilingual_transcript = [], [], []
        curr_pos = 0
        # Find all segments with exclusively Mandarin unicode ranges
        for group in re.finditer(r'([\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC\uF900-\uFAAD]+\s*)+', text):
            # Assume we start in English
            group_start, group_end = group.start(), group.end()
            if group_start > curr_pos:
                eng_encode = self.bpe_english.encode(text[curr_pos:group_start])
                eng_lang_tag = [self.bpe_english.PieceToId("<s_eng>")] if self.lang_tags else [] 
                man_lang_tag = [self.bpe_mandarin.PieceToId("<s_eng>")] if self.lang_tags else [] 
                eng_group_trns = (eng_lang_tag + eng_encode)
                man_group_trns = (
                    man_lang_tag + [self.bpe_mandarin.PieceToId("<mask>") for i in eng_encode]
                )
                english_transcript.extend(eng_group_trns)
                mandarin_transcript.extend(man_group_trns)
                bilingual_transcript.extend(eng_group_trns)
                curr_pos = group_start
            # Now we do Mandarin
            group_text = text[group_start:group_end]
            group_text = re.sub(r' ', '', group_text)
            man_encode = self.bpe_mandarin.encode(group_text)
            eng_lang_tag = [self.bpe_english.PieceToId("<s_man>")] if self.lang_tags else [] 
            man_lang_tag = [self.bpe_mandarin.PieceToId("<s_man>")] if self.lang_tags else [] 
            man_group_trns = (
                man_lang_tag + man_encode
            )
            eng_group_trns = (
                eng_lang_tag + [self.bpe_english.PieceToId("<mask>") for i in man_encode]
            )
            bil_offset = self.bpe_english.vocab_size()
            man_mask_id = self.bpe_mandarin.PieceToId("<mask>")
            man_unk_id = self.bpe_mandarin.PieceToId("[unk]")
            s_man_id = self.bpe_mandarin.PieceToId("<s_man>")
            bil_group_trns = [bil_offset + i if i not in (man_mask_id, man_unk_id, s_man_id) else i for i in man_group_trns]
            english_transcript.extend(eng_group_trns)
            mandarin_transcript.extend(man_group_trns)
            bilingual_transcript.extend(bil_group_trns)
            curr_pos = group_end

        # Once we've reach the end of the last mandarin group, the rest is
        # English
        if curr_pos < len(text):
            eng_encode = self.bpe_english.encode(text[curr_pos:])
            eng_lang_tag = [self.bpe_english.PieceToId("<s_eng>")] if self.lang_tags else [] 
            man_lang_tag = [self.bpe_mandarin.PieceToId("<s_eng>")] if self.lang_tags else [] 
            eng_group_trns = (eng_lang_tag + eng_encode)
            man_group_trns = (
                man_lang_tag + [self.bpe_mandarin.PieceToId("<mask>") for i in eng_encode]
            )
            english_transcript.extend(eng_group_trns)
            mandarin_transcript.extend(man_group_trns)
            bilingual_transcript.extend(eng_group_trns)
        
        return english_transcript, mandarin_transcript, bilingual_transcript 

