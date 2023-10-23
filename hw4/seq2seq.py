import argparse
import time
import logging
import torch
import torch.nn as nn
from io import open
from nltk.translate.bleu_score import corpus_bleu
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from torch import optim

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_SYMBOL = "<START>"
END_SYMBOL = "<END>"
START_INDEX = 0
END_INDEX = 1
SEQ_MAX_LEN = 15

class Vocabulary:
    def __init__(self, language):
        self.language = language
        self.word_to_index = {START_SYMBOL: START_INDEX, END_SYMBOL: END_INDEX}
        self.word_freq = {}
        self.index_to_word = {START_INDEX: START_SYMBOL, END_INDEX: END_SYMBOL}
        self.total_words = 2

    def add_line(self, line):
        for word in line.split():
            self._add_word(word)

    def _add_word(self, word):
        if word in self.word_to_index:
            self.word_freq[word] += 1
        else:
            self.word_to_index[word] = self.total_words
            self.word_freq[word] = 1
            self.index_to_word[self.total_words] = word
            self.total_words += 1

def read_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    sentence_pairs = [line.split('|||') for line in lines]
    return sentence_pairs

def generate_vocabs(src_lang, tgt_lang, data_file):
    src_vocab = Vocabulary(src_lang)
    tgt_vocab = Vocabulary(tgt_lang)
    sentence_pairs = read_data(data_file)
    for src, tgt in sentence_pairs:
        src_vocab.add_line(src)
        tgt_vocab.add_line(tgt)

    logging.info('Vocab size for %s: %d', src_vocab.language, src_vocab.total_words)
    logging.info('Vocab size for %s: %d', tgt_vocab.language, tgt_vocab.total_words)

    return src_vocab, tgt_vocab

def convert_to_tensor(vocab, line):
    indices = [vocab.word_to_index.get(word, END_INDEX) for word in line.split()]
    indices.append(END_INDEX)
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

def pair_to_tensors(src_vocab, tgt_vocab, pair):
    src_tensor = convert_to_tensor(src_vocab, pair[0])
    tgt_tensor = convert_to_tensor(tgt_vocab, pair[1])
    return src_tensor, tgt_tensor

class SimpleLSTM(nn.Module):
    # ... [rest of the SimpleLSTM code]
    def __init__(self, input_size, hidden_size, bias=True):
        super(SimpleLSTM, self).__init__()
        
        # input gate
        self.wx_input = nn.Linear(input_size, hidden_size, bias=bias)
        self.wh_input = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # forget gate
        self.wx_forget = nn.Linear(input_size, hidden_size, bias=bias)
        self.wh_forget = nn.Linear(hidden_size, hidden_size, bias=bias)

        # output gate 
        self.wx_output = nn.Linear(input_size, hidden_size, bias=bias)
        self.wh_output = nn.Linear(hidden_size, hidden_size, bias=bias)

        # context gate 
        self.wx_context = nn.Linear(input_size, hidden_size, bias=bias)
        self.wh_context = nn.Linear(hidden_size, hidden_size, bias=bias)

        #utilities
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        
        h_t, c_prev = hidden

        i_t = self.sigmoid(self.wx_input(input) + self.wh_input(h_t))
        f_t = self.sigmoid(self.wx_forget(input) + self.wh_forget(h_t))
        o_t = self.sigmoid(self.wx_output(input) + self.wh_output(h_t))
        
        c_t = self.tanh(self.wx_context(input) + self.wh_context(h_t)) #tilde c_t 
        c_t = f_t * c_prev + i_t * c_t
        
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

    

class SeqEncoder(nn.Module):
    # ... [rest of the SeqEncoder code]
    def __init__(self, input_size, hidden_size):
        super(SeqEncoder, self).__init__()
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should NOT use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "* YOUR CODE HERE *"
        self.emb = nn.Embedding(input_size, hidden_size) 
        self.lstm = SimpleLSTM(hidden_size, hidden_size)       
        


    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "* YOUR CODE HERE *"
        
        
        x = self.emb(input)
        h_t, c_t = self.lstm(x, hidden)
        return h_t, (h_t, c_t)

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnSeqDecoder(nn.Module):
    # ... [rest of the AttnSeqDecoder code]
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=SEQ_MAX_LEN):
        super(AttnSeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.emb = nn.Embedding(output_size, hidden_size)
        self.lstm = SimpleLSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.w_query = nn.Linear(hidden_size, hidden_size)
        self.w_key = nn.Linear(hidden_size, hidden_size)
        self.w_value = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        """
        Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        
        "* YOUR CODE HERE *"
        x = self.emb(input)
        #x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        h_t, _ = hidden #h_t is the decoder hidden state
        
        # (scaled) dot product attention
        if len(encoder_outputs.shape) == 2:
            encoder_outputs = torch.unsqueeze(encoder_outputs, 0)

        query = self.w_query(h_t) #b, 1, n
        key, value = self.w_key(encoder_outputs), self.w_value(encoder_outputs) #b, length, n
        #print(f"key.shape = {key.shape}")
        
        attn_weights = torch.matmul(query, torch.transpose(key, 1, 2))
        # q^{\top}k should be b, 1, length
        # attn_weights = torch.div(attn_weights, math.sqrt(self.hidden_size))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # context vector c_t = sum of dot product of attn_weights and encoder_outputs
        # attn_weight shape (1, length), encoder_outputs shape (length, hidden_size)
        for i in range(attn_weights.shape[2]):
            if i == 0:
                c_t = attn_weights[:, :, i]*encoder_outputs[:, i, :]
            else:
                c_t = c_t + attn_weights[:, :, i]*encoder_outputs[:, i, :]
        
        hidden = (h_t, c_t)
        h_t, c_t = self.lstm(x, hidden)
        output = h_t
        hidden = (h_t, c_t)
        output = self.out(output)
        
        log_softmax = torch.log(torch.softmax(output, dim=-1))
        return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def run_training(input_tensor, target_tensor, encoder, decoder, opt, loss_func, max_len=SEQ_MAX_LEN):
    # ... [rest of the run_training code]
    encoder_hidden = (encoder.get_initial_hidden_state(), encoder.get_initial_hidden_state())

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    opt.zero_grad()
    "* YOUR CODE HERE *"
    loss = 0.0
    for i in range(input_tensor.shape[0]):
        output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        # shape of output = batch, 1, hidden_size
        if i == 0:
            encoder_outputs = output
        else:
            encoder_outputs = torch.cat([encoder_outputs, output], dim=1)
    
    
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[0]], device=device)
    for i in range(target_tensor.shape[0]):
        logits, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
        logits = torch.squeeze(logits, 0) 
        #B, 1, vocab_size
        loss += loss_func(logits, target_tensor[i])
        decoder_input = target_tensor[i]
    
    loss.backward()
    opt.step()
    
    return loss.item() 

# Rest of the code...

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=SEQ_MAX_LEN):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = convert_to_tensor(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = (encoder.get_initial_hidden_state(), encoder.get_initial_hidden_state())

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[START_INDEX]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == END_INDEX:
                decoded_words.append(END_SYMBOL)
                break
            else:
                decoded_words.append(tgt_vocab.index_to_word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=SEQ_MAX_LEN):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    #TODO
    att=torch.transpose(attentions,0,1)
    input_words = input_sentence.strip().split()
    fig, ax = plt.subplots()
    ax.pcolor(att, cmap=plt.cm.Greys)
    ax.set_xticks(np.arange(attentions.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(input_words)) + 0.5, minor=False)
    ax.set_xlim(0, int(attentions.shape[0]))
    ax.set_ylim(0, int(len(input_words)))
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(output_words, minor=False)
    ax.set_yticklabels(input_words, minor=False)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Saving the plot to a file
    index = len(input_sentence) 
    plt.savefig(f'heatmap_{index}.png')


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(END_SYMBOL, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=200000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=500, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = generate_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    print(f"tgt_vocab.total_words = {tgt_vocab.total_words}")
    encoder = SeqEncoder(src_vocab.total_words, args.hidden_size).to(device)
    decoder = AttnSeqDecoder(args.hidden_size, tgt_vocab.total_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = read_data(args.train_file)
    dev_pairs = read_data(args.dev_file)
    test_pairs = read_data(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        iter_num += 1
        if iter_num % 100 == 0:
            print(f"now at iter {iter_num}")
        training_pair = pair_to_tensors(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = run_training(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)
    
    print("Translating the test set")
    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')
    
    print("Visualizing the attention")
    # Visualizing Attention
    # translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    # translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    # translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    # translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)
    

if __name__ == '__main__':
    main()
