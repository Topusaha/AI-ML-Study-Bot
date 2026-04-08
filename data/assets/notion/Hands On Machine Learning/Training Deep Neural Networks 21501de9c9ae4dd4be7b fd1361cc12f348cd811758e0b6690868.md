# Training Deep Neural Networks 21501de9c9ae4dd4be7b9389b6a7b1a0

# Training Deep Neural Networks

### Main Ideas

---

- What are some problems you can face when training a deep neural network?
    - faced with a tricky vanishing gradient or exploding gradient problem
    - not having enough data
    - slow training speed
    - risk overfitting
- Explain vanishing gradient and exploding gradient problems?
    - vanishing gradient - gradients get smaller and smaller as the back propagation algorithm progresses down to the lower layers. Because of this, the lower layers connection weights are unchanged
    - exploding gradient - the gradients get bigger and bigger until the lower layers get an insane connection weight updates
- Explain how to solve the vanishing gradient and exploding gradient problems
    - The problem is caused by the variance not being equal within the activation function
    - As a result, we use Glorot initialization or Xavier initialization
    - fan(avg) = (fan(in) + fan(out) / 2
    - This equation randomly sets the connection weights within a specific distribution to promote efficient learning by keeping the variance at an appropriate level
- Explain the concept of dying ReLU
    - the neurons die and all it is able to output is zero
    - neuron dies when the weights are tweaked in a way that the weighed sum of its inputs are negative for all instances returning zeros
- Explain the concept of leaky ReLU
    - leaky ReLU is when the function leaks so it can output some value other than zero when dealing with negative values
    - leaky ReLU will not die because of this however they can go into a long comma but have a chance of waking back up again
    - hyperparamter a defines how leaky it is and the more leaky the function is the better the function does
- Explain the benefits of ELU Function
    - Exponential linear unit Function
    - can output negative values for Z < 0
    - avoids dying neuron problem
    - avoids vanishing gradients
- Explain the benefits of SELU Function
    - Scaled ELU
    - when using a stack of dense layers with all layers using the function SELU the DNN will self normalize if the following conditions are met
        - input features are standardize (mean = 0, standard dev = 1)
        - hidden layers weight must be initialized with LeCun normal initialization
        - Architecture must be sequential
        - layers are densed
- Explain Batch Normalization
    - A technique to address the concept of exploding or vanishing gradients during training by adding an operation in the model before or after the activation function of each hidden layer to zero-center and normalize the input and scale and shift the output
- Explain Gradient Clipping
    - helps solve the exploding gradient problem by clipping gradients during the back propagation so that they will never exceed a defined threshold
- Explain Transfer Learning
    - Finding a DNN that already exists and solves a similar problem to the one you are solving and reusing some of its layers
    - The output layer would always be replaces as well as the input layer to match the specific problem
    - The lower layers are most useful and the higher layers would need to be replaced to tweaked
    - Try freezing all the reused layers to get a base model, and then unfreeze two or one of the top layers to let back propaganda tweak them to see if the performance increases
    - If training data is low, try deleting some of the top layers and repeating the process until you get a good performance
-