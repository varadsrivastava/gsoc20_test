# TEST
## Easy :

###	Can you explain what is the convolutional neural network?

#### Answer : 
Neural networks have a series of hidden layers, through which they transform the input, which is essentially in the form of a single vector. Each neuron in each hidden layer is fully connected to all the neurons in the previous layer, and neurons in the same layer do not share any connections.

When, the input is in the form of an image, however, these regular neural networks do not scale well to images of respectable sizes. For example, an image of dimensions 224x224x3 would mean, a neuron in the first hidden layer, having 224*224*3 i.e. 150,528 weights, thus increasing the number of parameters, which may lead to overfitting. 

However, in a convolutional neural network, network architecture allows the neurons to be arranged in the three dimensions i.e. width, height and depth. Also, instead of being fully connected, neurons in a layer are only connected to a small region of the neurons in the layer before it.

The hidden layers in these neural networks seem more biologically plausible to our visual system in the brain, in the way, that each of these hidden layers recognise a different set of features in the image.

###	Can you explain what is the transfer learning?
#### Answer : 
Essentially, when we come to use of neural networks in real life problems usage, we come across a big problem, which is lack of sufficient data.

Transfer learning comes to the rescue here, as it allows us to use a pretrained model as a feature extractor for a problem, where our problem dataset is similar to the data on which it was pre-trained. Fine-tuning of the weights through the whole network is also useful in cases, where new dataset is large as well as similar to the original one.

###	Can you install TensorFlow in your machine and implement a simple CNN on MNIST by TF estimator API following the official document?
#### Answer : 
Yes, I can install Tensorflow in my machine. However, I'm not able to properly implement the aforementioned following the official document. Official document is not updated for TensorFlow 2.0. feature_columns have been moved to tfdatasets recently, and I tried to go around that, as well as also use 'feature_specs' in tfdatasets. Currently, I'm stuck with the error "got an unexpected keyword argument 'input_layer_partitioner'". Input layer partitioners are supposed to be defined explicitely, and otherwise default to min_max_variable_partitioner, according to the official documentation here (https://tensorflow.rstudio.com/reference/tfestimators/dnn_estimators/). They have however been removed in the latest TensorFlow according to the Python version. I'm not sure that the documentation is updated.

However, I have attached the code here as """"" Test.R """""
And I'll keep trying to run it successfully.


## Medium :

###	What are overfitting?
#### Answer :
Overfitting happens, when a model learns the data instead of learning the function underlying the data. So, we achieve a really good accuracy on the training data, while a significantly poor accuracy on the test data.

###	What are L1 and L2 regularization?
#### Answer :
There are several ways of preventing overfitting. Two of them are L1 and L2 regularization. 

In L2 regularization, we encourage the network to use a little of all of its inputs rather than using some of the inputs a lot. Intuitively, we penalize extreme weight vectors and prefer diffuse ones. This is done by adding a term (1/2)*A*(w^2) to the objective, where A is the regularization strength.

In L1 regularization, we add A*|w| for each weight. It leads the weight vectors to become sparse. We use L1 for sparsity training.

###	What should we do if the loss doesn’t converge?
#### Answer :
If the loss doesn’t converge, we should do one of the following : 
1.	We should try lowering the learning rate. It’s possible, that a higher learning rate may have overshot the global minima.
2.	We should check the magnitudes of gradients to see if they are exploding or vanishing. If they are, we should use an adaptive optimizer.

###	Can you implement a simple CNN without estimator API?
####Answer : 
Yes, kindly refer to """" Test2.R """"

