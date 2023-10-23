# Recurrent Neural Network with Attention mechanism
# Overview
This code implements a Long Short-Term Memory RNN encoder and decoder with attention. It does not make use of the pytorch nn.LSTM function and instead implements the formulas. In addition to the implementation, it contains a write up of the visualization for the attention mechanism and discusses selected plots.

# Table of Contents
1. Main Components 
2. Installation
3. Usage
4. Algorithms
5. Results
6. Contributors


# Main Components
## Vocab Class
A class that handles the mapping between words and their indices. It provides methods to convert sentences into numerical representations suitable for modeling.
Data Preparation
The provided functions allow users to:
Split lines from a file into pairs of sentences.
Create vocabularies based on the training corpus.
Convert sentences into tensors that can be fed into the models.

## MyLSTM Class
An implementation of the LSTM cell. The class encapsulates the inner workings of the LSTM unit, defining the input, forget, output, and context gates.

## EncoderRNN Class
The encoder takes in a sequence of embeddings and returns the final hidden state which is then used by the decoder. The encoder is implemented using the aforementioned LSTM cell.

## AttnDecoderRNN Class
The decoder, augmented with an attention mechanism. It computes attention weights for each time step based on the encoder's output and the decoder's previous state, enabling the model to focus on different parts of the input sequence at each decoding step.
Training Function
This function accepts an input tensor (source sentence) and a target tensor (target sentence), and updates the model's weights using backpropagation.



# Installation
1. Make sure you have python and github on your system

2. Follow the proper steps in the INSTALL_NOTES.md file

3. Clone the repository to your local machine:  
   **git clone https://github.com/nicoledeprey/MachineTranslation_hw4.git**

4. Navigate to the project directory:  
**cd hw4**


# Usage
Running seq2seq.py


1. Run the file with the following command:
**python seq2seq.py**

2. The arguments can be viewed by running:
**python seq2seq.py -h**


# Algorithms
## Long Short-Term Memory (LSTM)  
LSTM can be used to solve problems faced by the RNN model, such as, long term dependency problems and the vanishing and exploding gradient. LSTM makes use of three gates: forget gate, f, input gate, i, and output gate, o. LSTM also makes use of a cell state and candidate cell state to find the final output. A description of the LSTM Algorithm can be found in the MathDescription.pdf.


## Attention Visualization  
Attention is used to focus on different parts of the input at different steps. The attention mechanism computes attention scores for each element of the input sequence, indicating its relevance to the current decoding step. The attention scores are then normalized to create a probability distribution. A description of the attention decoder can be found in the MathDescription.pdf.

# Results
The code results should produce a BLEU Score of 0.4617316003241794.
This sequence-to-sequence with attention model offers a starting point for machine translation tasks and other applications where both the input and output are sequences. Users are encouraged to adapt and expand upon the base code to fit specific needs and challenges.


# Contributors
This code was developed by Janvi Prasad, Nicole Deprey, and Hirtika Mirghani.
