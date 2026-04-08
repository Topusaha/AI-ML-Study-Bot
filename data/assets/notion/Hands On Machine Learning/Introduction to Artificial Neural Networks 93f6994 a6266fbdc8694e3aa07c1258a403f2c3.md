# Introduction to Artificial Neural Networks 93f6994a1fed4aa98aca57b2096299e4

# Introduction to Artificial Neural Networks

### Main Ideas

---

- What is Artificial Neural Networks (ANN)?
    - A machine learning model inspired by the networks of biological neurons found in our brains
    - They have differed from them with time though
- Define the following terms
    - Input Layer
        - The input layer is the first layer of an artificial neural network that receives input data and passes it on to the next layer for processing.
    - Output Layer
        - The output layer is the final layer of an artificial neural network that produces the model’s predictions or outputs based on the input data and the processing done by the previous layers.
    - Bias Neuron
        - A bias neuron is a neuron in an artificial neural network that always outputs 1. It is used to adjust the output of the previous layer and make it more adaptable to the desired output.
- What is a fully connected layer or dense layer?
    - When all the neurons in a layer are connected to every neuron in the previous layer
- Describe Hebb’s Rule?
    - Cells that fire together, wire together
    - The weight between two neuron’s tends to increase when they fire at the same time
- What is a Perceptron?
    - A single layer of Artificial neurons that produce output values and used for binary classification problems.
    - Hard to solve complex problems with
- What is the difference between Logistic Regression and Perceptrons?
    - Logistic Regression outputs class probability while Perceptrons outputs predictions based on hard thresholds.
- Explain the Reverse-mode autodiff algorithm?
    - A back-propagation algorithm to optimize the neural network
    - Steps
        - Forward Pass
            - Makes a prediction for each training instance
        - Calculates Error
        - Reverse Pass
            - Goes through each layer in reverse to measure error contribution from each connection
        - Tweaks connection weights to reduce error using Gradient Descent
- Explain Common Activation Functions?
    - Sigmoid Function: maps any input value to a value between 0 and 1
    - Rectified Linear Unit (ReLU): outputs the input value if it is positive, and outputs 0 if it is negative
    - Hyperbolic Tangent (tanh): maps any input value to a value between -1 and 1
- When would you use the different activation functions for the output layer?
    - ReLU or Softplus - When you want the output will always be positive
    - Logistic Function or Hyperbolic tangent - When you want the output to fall between a certain range
- Explain the architecture for a regression MLP below
    - Input Neurons
        - one per input feature
    - Hidden Layers
        - Depends
    - Neurons per Hidden Layer
        - Depends
    - Output Neurons
        - one per prediction dimension
    - Hidden Activation
        - ReLU or SELU
    - Output Activation
        - None
        - ReLU or soft-plus if you need positive outputs
        - Logistic or tanh if you need bounded outputs
    - Loss Function
        - MSE
        - MAE/Huber if they’re outliers
- Explain the wide and deep architecture for neural networks
    - It is able to connect all or part of the inputs to the outer layer
    - This way the network can learn deep patterns and simple patterns
    
    ![](Introduction%20to%20Artificial%20Neural%20Networks%2093f6994/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaIntroduction_to_Artificial_Neural_Networks_93f6994a1fed4aa98aca57b2096299e4Untitled.png)
    
    Untitled
    
    ![](Introduction%20to%20Artificial%20Neural%20Networks%2093f6994/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaIntroduction_to_Artificial_Neural_Networks_93f6994a1fed4aa98aca57b2096299e4Untitled_1.png)
    
    Untitled
    
- In what situations would you want multiple inputs
    - When the tasks requires two different kinds of algorithms such as classification and regression
        - Finding faces within an image
        - Recognizes faces is a classification task
        - Needs regression to outputs x and y values for the image
    - Trying to find multiple independent values within the same dataset
    - Regularization
- What is the difference and similarities between Sequential API and Functional API and Subclassing API
    - Sequential API has all the neurons in the network go through each other in order
    - Functional API have more flexibility in the order of the network
    - Both Sequential API and Functional API are declarative meaning you start by declaring what layers you are going to use and how they are connected.
- What is the advantages and disadvantages in using a declarative API like Sequential or Functional other than the Subclassing API
    - Declarative API makes the model easy to save, clone, share, debug, and see the structure
    - Subclassing API are more dynamic since you can add if statements, for-loops and other structures within it

### Questions

---

- Why is it generally preferable to use a Logistic Regression classifier rather than a classical Perceptron (i.e., a single layer of threshold logic units trained using the Perceptron training algorithm)? How can you tweak a Perceptron to make it equivalent to a Logistic Regression classifier?
    - Logistic Regression is able to give probabilities as well rather then just a clear binary answer like the Perceptron does
    - To make Perceptron equivalent to the Logistic Regression Classifier we can have two output values instead of one to show how likely the input is to be in those values
- Why was the logistic activation function a key ingredient in training the first MLPs?
    - Because MLPs gives binary outputs which is what a logistic activation function is good at doing. It will provide ether 2 values to represent a binary output
- Name three popular activation functions?
    - Relu
    - Softplus
    - SoftMax
    - tanh
    - sigmod
- 409 Questions for ML chapter 10