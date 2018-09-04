
# Introduction to Keras Lab

## Objective

In this lab, we'll learn how to use Keras-- an industry standard tool to build for quickly prototyping and training Deep Neural Networks.

## Introduction

In the last lab, we learned how to build a small neural network from scratch, with only numpy to help.  While this was illuminating to help us learn how Artificial Neural Networks actually work, the construction of it was a bit too tedious for real-world data scientists to do on the job. Because of this, many great frameworks and libraries have been created to make it a simple task to build a neural network--Keras, and TensorFlow!
<table>
<tr>
    <td><a href="https://www.tensorflow.org"><img src='tensorflow-logo.png' height=100% width=100%></a></td>
    <td><a href="https://keras.io/"><img src='keras-logo.png' height=75% width=75%></a></td>
</tr>
</table>

## A Bit of Background

If we're going to talk about Keras, we'll first need to talk about TensorFlow.  Keras and TensorFlow are both created and maintained by Google.  TensorFlow is a framework for building **_Static Computational Graphs_**, with a special emphasis on Deep Learning.  It provides much of the same functionality of numpy, but with a ton of common functionality needed for neural networks already created. Since working with TensorFlow can be a bit involved, Keras was invented as a high-level API for building Neural Networks with TensorFlow as a backend.  When you're using Keras, you're still using TensorFlow--you just don't have to write any TensorFlow code directly.

The biggest benefit of using TensorFlow or Keras is that they provide [Auto Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) by default, which means that we don't actually have to handle Back Propagation, or caching any of the intermediate values during our forward propagation step so that we can use them later in back prop.  This is just one of the many reasons you'll love working with these tools!

## Getting Started

We'll start by building a small MLP with Keras.  Run the cell below to import everything we'll need for this lab.


```python
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
%matplotlib inline
```

    Using TensorFlow backend.


## The Data Set

In this lab, we'll be working with the MNIST dataset.  This dataset is a collection of handwritten digits, 0 through 9.  Each image is black and white, 28 pixel high, 28 pixels wide, and comes with a label.  From a historical perspective, this dataset has incredible significance to the Deep Learning community.  These days, it is treated as the "Hello, World!" of Deep Learning.  

We'll have to do some basic data preparation in order to load in our data and get it in a shape usable by a Neural Network.

In the cell below, call `mnist.load_data()` to store our data in the tuples we've laid out. You'll note that it's already been split into training and testing sets.

**_NOTE:_** If this is your time ever running this on your computer, Keras will have to download the images from the internet.  This is a large dataset, and the download could take a while.  Be patient!


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

Let's inspect a sample image to see what one of the images looks like.  

In the cell below, call `plt.imshow()` and pass in the first item from `X_train`.


```python
plt.imshow(X_train[0])
print("Label: {}".format(y_train[0]))
```

    Label: 5



![png](output_5_1.png)


a 5! A somewhat sloppy 5, but still a 5.  After a few short lines of code, we'll have built and trained a neural network that is able to recognize this as a 5.  Let's take a look at the way raw data for the first sample.

In the cell below, access `X_train[0]` to see what it looks like.


```python
X_train[0]
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
             18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
            253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
            253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
            253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
            205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
             90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
            190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
            253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
            241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
            148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
            253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
            253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
            195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
             11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0]], dtype=uint8)



Our image is a 28x28 matrix.  In plain English, that means that it's an array containing 28 arrays, with each inner array containing 28 integer values between 0 and 255.  

In order for us to feed this into a neural network, we'll need to reshape this from a matrix into a vector.  This is a basic step needed for most image preparation (except when we're building a Convolutional Neural Network--more on that in a later lab!).  

In the cell below, we'll reshape `X_train` and `X_test` using their built in `.reshape()` methods.

`X_train` contains 60,000 samples of shape `(28,28)`--so we'll reshape it to `(60000, 784)`.  We'll also method chain to cast the data inside of it from type `int8` to `float32`.  

`X_test` contains only 10,000 samples, so we'll reshape that to `(10000, 784)`.  We'll also cast this to type `float32`.  


```python
X_train = X_train.reshape(60000, 784).astype("float32")
X_test = X_test.reshape(10000, 784).astype("float32")
```

While we're preprocessing our data, it's also a good idea to normalize it.  Normally, this would involve subtracting the mean and dividing by the standard deviation for each column, or using a `StandardScaler()` object from sklean.  However, since we're working with image data, we can use a quick hack and just divide every value by 255.

In the cell below, normalize the data in `X_train` and `X_test` by dividing each by 255.


```python
X_train /= 255.
X_test /= 255.
```

We've successfully pre-processed our data, but we still need to deal with our labels, which are inside of `y_train` and `y_test`.  This is much easier--we need to change the data from its current categorical form to a one-hot encoded format.  Keras makes this easy by providing a `to_categorical()` function inside of its `utils` module.

In the cell below, call `keras.utils.to_categorical()` and pass in the variable containing the labels to be reshaped as the first parameter.  For the second parameter, pass in the number of categories the data contains--in this case, `10`.


```python
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

Great! We've now successfully preprocessed our data so that it's ready for use in a neural network!

## Building a Neural Network with Keras

Let's start by building our neural network. This means we need to create a `Sequential()` object.  This is an empty neural network that hasn't had any layers added to it yet.  

Our workflow in Keras will consist of of the following:

* Create a `Sequential` object
* Add the layers we want to our `Sequential` object. When defining the first hidden layer, we'll also define our input layer.  
* Compile our model
* Fit our model
* Examine the results!

Let's get started! In the cell below, create a `Sequential()` object.


```python
model = Sequential()
```

Now, we can add layers to our model by passing them in to the model's `.add()` method.  There are a ton of different kinds of layers we can add to our model, but for this lab we'll keep it simple and only use a single `Dense()` layer.



There are several parameters we'll need to pass in to the `Dense()` layer we're creating at each step, to define exactly how we want our Dense layer to be shaped.  We'll need to tell the first layer the following things:

* The number of neurons in the layer
* The activation function of the layer (although this can also be added using `model.add()`, it's easier to just pass it in as a keyword parameter when creating the `Dense` layer)
* If the layer we're creating is the first hidden layer, we'll also need to tell it the shape of the input layer.  In this case, we'll do that by passing in the keyword parameter `input_shape=(784,)`.

In the cell below:

* Call `model.add()` and pass in a `Dense` layer with `25` neurons.  Also set the activation to `"sigmoid"`, and the input shape to `(784,)`
* Call `model.add()` again to create out output layer.  This time, pass in a `Dense` layer with `10` neurons (one for each possible class that it can predict).  This time, set the activation to `"softmax"`.


```python
model.add(Dense(25, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
```

Great! With 2 lines of code, we've now created a complex neural network capable of reading human handwriting!

Let's look at the structure of the model we've just created.

In the cell below, call `model.summary()`.


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 25)                19625     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                260       
    =================================================================
    Total params: 19,885
    Trainable params: 19,885
    Non-trainable params: 0
    _________________________________________________________________


Note that the summary of our model doesn't show the input layer by default. This is a common convention in Deep Learning, since the input layer doesn't contain any **_trainable parameters_**, meaning that it doesn't really do anything.  

Also note the number of trainable parameters in our network--19,885! Although this may seem like a lot, its still quite small compared to typical Artificial Neural Networks used in Deep Learning.  

Now, we need to compile our model.  This is a step unique to Keras, where the neural network we've created is actually created as a static computational graph in TensorFlow.  During this step, we'll need to pass in certain some information. Specifically:

* We'll need to specify the loss.  Since this is a multi-categorical problem, we'll set the `loss` to `"categorical_crossentropy"`.
* We'll also need to tell the model what type of optimizer we'll use.  In this case, set the `optimzer` to `"SGD"`, which is short for **_Stochastic Gradient Descent_**.
* Finally, we'll need to tell the model what metrics we want back.  In this case, we'll pass in `["accuracy"]` to the `metrics` keyword argument.


```python
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
```

Now, the fun part--we can actually fit our model, and see how well it does!

Just like the last step, there are multiple parameters we'll need to pass in, such as:

* `X_train` and `y_train`
* a `batch_size` of `32`
* The number of `epochs`, which we'll set to `10`.
* Set `verbose` to `1`, so that we'll get messages as the model trains.
* We'll also pass in `validation_data` to the tuple`(X_test, y_test)`.  With this, the model will compute the accuracy on our validation set at the end of each epoch.  


```python
model = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/10
    60000/60000 [==============================] - 3s 45us/step - loss: 1.6858 - acc: 0.6077 - val_loss: 1.1757 - val_acc: 0.7675
    Epoch 2/10
    60000/60000 [==============================] - 2s 40us/step - loss: 0.9520 - acc: 0.8040 - val_loss: 0.7624 - val_acc: 0.8444
    Epoch 3/10
    60000/60000 [==============================] - 2s 40us/step - loss: 0.6855 - acc: 0.8499 - val_loss: 0.5910 - val_acc: 0.8708
    Epoch 4/10
    60000/60000 [==============================] - 2s 40us/step - loss: 0.5597 - acc: 0.8710 - val_loss: 0.4983 - val_acc: 0.8873
    Epoch 5/10
    60000/60000 [==============================] - 2s 41us/step - loss: 0.4867 - acc: 0.8821 - val_loss: 0.4425 - val_acc: 0.8958
    Epoch 6/10
    60000/60000 [==============================] - 2s 41us/step - loss: 0.4396 - acc: 0.8898 - val_loss: 0.4045 - val_acc: 0.9010
    Epoch 7/10
    60000/60000 [==============================] - 2s 41us/step - loss: 0.4067 - acc: 0.8952 - val_loss: 0.3775 - val_acc: 0.9053
    Epoch 8/10
    60000/60000 [==============================] - 2s 41us/step - loss: 0.3823 - acc: 0.8994 - val_loss: 0.3577 - val_acc: 0.9081
    Epoch 9/10
    60000/60000 [==============================] - 2s 41us/step - loss: 0.3635 - acc: 0.9027 - val_loss: 0.3412 - val_acc: 0.9099
    Epoch 10/10
    60000/60000 [==============================] - 2s 41us/step - loss: 0.3483 - acc: 0.9059 - val_loss: 0.3289 - val_acc: 0.9123


Validation accuracy of 91%--that's pretty good!


## Visualizing our Results

Calling `model.fit()` in Keras has the added benefit of returning a `History` object that contains all the statistics we saw above, that were printed out above live during the training.  It's a good habit to visualize our loss and accuracy, as well as our validation loss and accuracy.  

In the cell below, examine `model.history` to see what it contains.


```python
model.history
```




    {'val_loss': [1.1756522094726563,
      0.76244125356674197,
      0.59095669641494752,
      0.4983105968475342,
      0.44249384846687317,
      0.40451340870857239,
      0.377485573554039,
      0.35771549837589262,
      0.34124959199428556,
      0.3289165615439415],
     'val_acc': [0.76749999999999996,
      0.84440000000000004,
      0.87080000000000002,
      0.88729999999999998,
      0.89580000000000004,
      0.90100000000000002,
      0.90529999999999999,
      0.90810000000000002,
      0.90990000000000004,
      0.9123],
     'loss': [1.6858388323783875,
      0.95204407924016321,
      0.68553595670064293,
      0.55968093349138892,
      0.48669601863225304,
      0.43961626720428465,
      0.40667204344272612,
      0.38229380979537964,
      0.36350644185940423,
      0.34825242487192154],
     'acc': [0.60766666666666669,
      0.80396666666666672,
      0.84991666666666665,
      0.87103333333333333,
      0.88213333333333332,
      0.88975000000000004,
      0.8952,
      0.89943333333333331,
      0.90273333333333339,
      0.90588333333333337]}



Great! It's just basic python dictionary with the following keys:

* `val_loss`
* `val_acc`
* `loss`
* `acc`

Let's plot our Accuracy and Validation Accuracy on the same plot, so that we check if the model is overfitting at all.

In the cell below, create a visualization using Matplotlib, and plot `val_acc` and `acc`.


```python
epochs = [i for i in range(1, 11)]
plt.figure()
plt.plot(epochs, model.history['val_acc'])
plt.plot(epochs, model.history['acc'])
plt.title("Model Accuracy by Epoch")
plt.ylabel("Accuracy (%)")
plt.xlabel("Epoch")
plt.legend(["val_acc", "acc"], loc=4)
plt.show()
```


![png](output_27_0.png)


**_Question:_** Do you detect any overfitting or underfitting in the model? How can you tell? Explain your answer.

Write your answer below this line:
______________________________________________________________________________________________________________________

The model seems to have a high overall validation accuracy score, which means the model is not underfit.  The training accuracy and testing accuracy are also very close, suggesting that the model has not yet begun to overfit the data.  However, this would likely happen if we drastically increased the size of the network, or continued training on the data for many more epochs.

## Building a Bigger Model

Now that you've cut your teeth in Keras, let's build a bigger model! For the remainder of this notebook, your task it build your own neural network in Keras from start to finish.  You do not need to repeat the data preprocessing steps done at the beginning of this notebook, since that data is still available to us.

In the cells below, your task is to:
* Build and train a bigger model for classification on the MNIST dataset.
* Visualize your results to check for overfitting and underfitting.

When building your MLP below, consider the following choices you'll need to make:

* How many layers will your model have?
* How many neurons will each layer have?
* What activation function will your model use?

We'll explore these topics in depth in a later lab, but it never hurts to learn through experimentation.

**_NOTE:_**  The larger your network, the longer the training will take.  Don't build something too big if you don't have access to GPU-backed training!


```python
# Example of a bigger model

model2 = Sequential()
model2.add(Dense(100, activation='tanh', input_shape=(784,)))
model2.add(Dense(75, activation='tanh'))
model2.add(Dense(25, activation='tanh'))
model2.add(Dense(10, activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_2 = model2.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 5s 81us/step - loss: 0.2836 - acc: 0.9214 - val_loss: 0.1544 - val_acc: 0.9541
    Epoch 2/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.1256 - acc: 0.9626 - val_loss: 0.1128 - val_acc: 0.9658
    Epoch 3/20
    60000/60000 [==============================] - 5s 85us/step - loss: 0.0890 - acc: 0.9734 - val_loss: 0.0919 - val_acc: 0.9716
    Epoch 4/20
    60000/60000 [==============================] - 5s 86us/step - loss: 0.0685 - acc: 0.9791 - val_loss: 0.0964 - val_acc: 0.9689
    Epoch 5/20
    60000/60000 [==============================] - 5s 87us/step - loss: 0.0539 - acc: 0.9835 - val_loss: 0.0939 - val_acc: 0.9734
    Epoch 6/20
    60000/60000 [==============================] - 6s 93us/step - loss: 0.0454 - acc: 0.9859 - val_loss: 0.0799 - val_acc: 0.9775
    Epoch 7/20
    60000/60000 [==============================] - 5s 85us/step - loss: 0.0384 - acc: 0.9878 - val_loss: 0.0853 - val_acc: 0.9754
    Epoch 8/20
    60000/60000 [==============================] - 5s 84us/step - loss: 0.0313 - acc: 0.9900 - val_loss: 0.0889 - val_acc: 0.9755
    Epoch 9/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.0284 - acc: 0.9908 - val_loss: 0.0821 - val_acc: 0.9766
    Epoch 10/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.0249 - acc: 0.9919 - val_loss: 0.0834 - val_acc: 0.9755
    Epoch 11/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.0221 - acc: 0.9930 - val_loss: 0.0867 - val_acc: 0.9768
    Epoch 12/20
    60000/60000 [==============================] - 5s 86us/step - loss: 0.0210 - acc: 0.9931 - val_loss: 0.1009 - val_acc: 0.9730
    Epoch 13/20
    60000/60000 [==============================] - 5s 81us/step - loss: 0.0179 - acc: 0.9941 - val_loss: 0.1047 - val_acc: 0.9722
    Epoch 14/20
    60000/60000 [==============================] - 5s 81us/step - loss: 0.0149 - acc: 0.9950 - val_loss: 0.1025 - val_acc: 0.9757
    Epoch 15/20
    60000/60000 [==============================] - 5s 81us/step - loss: 0.0150 - acc: 0.9950 - val_loss: 0.0963 - val_acc: 0.9749
    Epoch 16/20
    60000/60000 [==============================] - 5s 81us/step - loss: 0.0135 - acc: 0.9955 - val_loss: 0.0919 - val_acc: 0.9777
    Epoch 17/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.0137 - acc: 0.9955 - val_loss: 0.1325 - val_acc: 0.9666
    Epoch 18/20
    60000/60000 [==============================] - 5s 90us/step - loss: 0.0120 - acc: 0.9960 - val_loss: 0.0960 - val_acc: 0.9779
    Epoch 19/20
    60000/60000 [==============================] - 5s 84us/step - loss: 0.0151 - acc: 0.9951 - val_loss: 0.0992 - val_acc: 0.9751
    Epoch 20/20
    60000/60000 [==============================] - 5s 84us/step - loss: 0.0103 - acc: 0.9966 - val_loss: 0.1003 - val_acc: 0.9750



```python
epochs_2 = [i for i in range(1, 21)]
plt.figure()
plt.plot(epochs_2, history_2.history['val_acc'])
plt.plot(epochs_2, history_2.history['acc'])
plt.title("Model Accuracy by Epoch")
plt.ylabel("Accuracy (%)")
plt.xlabel("Epoch")
plt.legend(["val_acc", "acc"], loc=4)
plt.show()
```


![png](output_31_0.png)


## Conclusion

In this lab, we learned how to build a basic model in Keras, and gained some valuable experience with the general Deep Learning workflow.
