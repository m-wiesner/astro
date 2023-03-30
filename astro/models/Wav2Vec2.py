from transformers import Wav2Vec2ForCTC, Wav2Vec2ForPreTraining
import torch
import torch.nn as nn


class Wav2Vec2(nn.Module):
    '''
        The pretrained Wav2Vec2 models from hugging face, but packaged in a way to
        be useable for ASR using lhotse.
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--wav2vec2-model-path', type=str, default='facebook/wav2vec2-large-xlsr-53')
        parser.add_argument('--wav2vec2-num-freeze-epochs', type=int, default=1)
    def __init__(self, modelpath='facebook/wav2vec2-large-xlsr-53'):
        '''
            Inputs:
                :param num_classes: the number of bpe output classes
                :param modelpath: the huggingface model path to use.
                :return: Wav2Vec2 model
        '''
        super(Wav2Vec2, self).__init__()
  
        # The pretrained wav2vec2 encoder. Depending on how a specific Wav2Vec2
        # model was packaged, it either uses the Wav2Vec2ForCTC or 
        # Wav2VecForPretraining classes for model loading
        try:
            self.encoder = Wav2Vec2ForCTC.from_pretrained(modelpath).wav2vec2
        except:
            self.encoder = Wav2Vec2ForPreTraining.from_pretrained(modelpath).wav2vec2

        # In fine-tuning, it is common practice to freeze the parameters of the
        # feature extractor, i.e., https://arxiv.org/pdf/2006.11477.pdf 
        self.encoder.feature_extractor._freeze_parameters()
        self.odim = self._get_output_dim()

    def forward(self, x, input_lens):
        # We pass the input tensor through the wav2vec2 model, i.e., feature
        # extractor and subsequent transformer, and then extract a specific
        # hidden layer from the transformer. We then pass these embeddings
        # through a linear layer that reduces the dimensionality of the
        # embeddings from 1024 to the desired number of bpe output classes.
        x = self.encoder(
            x.squeeze(-1), output_hidden_states=True
        ).hidden_states[-1]

        # We also must keep track of the input lengths, T_i, associated with
        # target sequence, i, so that we compute the CTC, or other loss,
        # for each minibatch example (i.e., minibatch[i, 0:T_i, :]). The way
        # the input length changes at each layer depends on the amount of
        # downsampling in the network. In Wav2Vec2, all of the downsampling
        # occurs in the convolutional layers of the feature extractor. We know
        # the kernel widths and strides of each of these layers so for now we
        # have just hardcoded them as a list of tuples (kernel, stride).
        for width, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            input_lens = torch.floor((input_lens - width) / stride + 1)
        return  x, input_lens.to(torch.int32)

    # This function is used in fine-tuning only the output linear layer
    def freeze_encoder(self):
        for p in self.encoder.encoder.parameters():
            if p.requires_grad:
                p.requires_grad = False

    # This function undoes the previous one
    def unfreeze_encoder(self):
        for i, p in enumerate(self.encoder.encoder.parameters()):
            p.requires_grad = True

    def _get_output_dim(self):
        x = torch.rand(1, 400)
        return self.encoder(x).last_hidden_state.size(-1)

