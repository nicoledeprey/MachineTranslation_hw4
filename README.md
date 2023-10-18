# Recurrent Neural Network with Attention mechanism
# Overview
This code implements a Long Short-Term Memory RNN encoder and decoder with attention. It does not make use of the pytorch nn.LSTM function and instead implements the formulas. In addition to the implementation, it contains a write up of the visualization for the attention mechanism and discusses selected plots.

# Table of Contents
1. Installation
2. Usage
3. Algorithms
4. Results
5. Contributors


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

1. Long Short-Term Memory (LSTM)  
LSTM can be used to solve problems faced by the RNN model, such as, long term dependency problems and the vanishing and exploding gradient. LSTM makes use of three gates: forget gate, f, input gate, i, and output gate, o. LSTM also makes use of a cell state and candidate cell state to find the final output. A description of the LSTM Algorithm can be found in the MathDescription.pdf.

2. Attention Visualization  
Attention is used to focus on different parts of the input at different steps. The attention mechanism computes attention scores for each element of the input sequence, indicating its relevance to the current decoding step. The attention scores are then normalized to create a probability distribution. A description of the attention decoder can be found in the MathDescription.pdf.

# Results
The code results should produce a BLEU Score of XXX.

# Contributors
This code was developed by Janvi Prasad, Nicole Deprey, and Hirtika Mirghani.
