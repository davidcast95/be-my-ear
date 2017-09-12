# Fourth generation
###### Assume : we set a same weights, but different RNN, with 5 different features

5 features :
- MFCC (26 coef, 0 past and future context) [26]
- Spectrogram [129]
- MFCC (26 coef, 3 past and future context) [182]
- MFCC (26 coef, 5 past and future context) [286]
- MFCC (26 coef, 9 past and future context) [494]

### Model Specification

| Input					    | Model         			      | Output      | Batch |
| ----------------------------------------- | --------------------------------------- |-------------| ----- |
| MFCC (26 coef, 0 past and future context) | 3 NN (128 hidden neuron), 1 BiRNN (512) | 28 char     | 1     |
| Spectrogram				    | 3 NN (128 hidden neuron), 1 BiRNN (512) | 28 char     | 1     |
| MFCC (26 coef, 3 past and future context) | 3 NN (128 hidden neuron), 1 BiRNN (512) | 28 char     | 1     |
| MFCC (26 coef, 5 past and future context) | 3 NN (128 hidden neuron), 1 BiRNN (512) | 28 char     | 1     |
| MFCC (26 coef, 9 past and future context) | 3 NN (128 hidden neuron), 1 BiRNN (512) | 28 char     | 1     |

NB: Initial value of all model's weight are equally similar, using Normal Distribution. The problem is, every input has its vector representation ex: MFCC (26 coef, 0 past and future context) has 26 depth, spectrogram has 129 depth. If the model want to have similar weights, we generate 129 weights from previous model (we choose spectrogram because in previous model, the weights can solve the minimum error), thus when we need 26 depth, we resize the array of weights become 26 (we use numpy array). If we need a larger depth, numpy will copy the occurance value of its array.

### Data Representation
Because we running in 1 batch, it means this model has same behaviour with Stocastic Gradient Descent widely known as online learning. We have 104 dataset of training. So each iteration will result 104 batches.

### Result
###### We recap all the result file in csv files in Log directory
#### Directions
Each report.csv contains 4 columns (batch, learning_rate, ctc_loss, decode_text)
You may open this file using MS. Excel (Windows) or Numbers (MacOS)
You can filter by first column to see the changes each batch. Each rows represents amount of iterations.

##### Log-12-09-2017
We run all the models for 200 iterations. 
We surprised by the result. MFCC has advantages over Spectrogram. 
MFCC with 0 past-future context start discover word in around 100th ireation
MFCC with 3 and 5 past-future context start discover word in around 50th iteration.
Spectrogram cannot discover any word.


