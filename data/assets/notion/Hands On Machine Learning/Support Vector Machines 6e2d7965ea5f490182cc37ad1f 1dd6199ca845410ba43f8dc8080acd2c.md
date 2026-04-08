# Support Vector Machines 6e2d7965ea5f490182cc37ad1fb7cea8

# Support Vector Machines

---

### Main Ideas

---

- What is a Support Vector Machine?
    - A powerful and versatile Machine Learning Model able to preform linear and nonlinear classification, regression, and even outlier detection
- What is Hard Margin Classification and how is it different than Soft Margin Classification?
    - Hard Margin Classification
        - When all instances must be off the streets (lines)
        - Only works if data is linearly separable
        - sensitive to outliers
    - Soft Margin Classification
        - good balance between keeping the street as large as possible and limiting margin violations
        - Some instances can be misclassified
        - Better at generalizing
- Explain Nonlinear SVM Classification?
    - Not many datasets are linearly separable so one way to handle this is to add more features
    - By adding Polynomial features we can achieve a linearly separable dataset
    
    ![](Support%20Vector%20Machines%206e2d7965ea5f490182cc37ad1f/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaSupport_Vector_Machines_6e2d7965ea5f490182cc37ad1fb7cea8Untitled.png)
    
    Untitled
    
- Explain the kernel trick?
    - A mathematical technique to get the same results as if you had added many polynomial features without making the model slow
- Explain how you can use a similarity function to tackle nonlinear problems?
    - By creating bell shape functions for every landmark you transform the data into new features which gets the data to be linearly separable
    - You can select landmarks for each instance in the dataset but this approach the data from mxn to mxm and can increase the size of the dataset
- In the code block below what are the Hyper parameters and what do they do?
    
    ```jsx
    SVC(kernel='rbf', gamma=5, C=0.001)
    ```
    
    - gamma - the influence of the instance’s range
        - Higher gamma = narrower curve
        - Lower gamma = wider curve
        
        ![](Support%20Vector%20Machines%206e2d7965ea5f490182cc37ad1f/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaSupport_Vector_Machines_6e2d7965ea5f490182cc37ad1fb7cea8Untitled_1.png)
        
        Untitled
        
    
    ![](Support%20Vector%20Machines%206e2d7965ea5f490182cc37ad1f/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaSupport_Vector_Machines_6e2d7965ea5f490182cc37ad1fb7cea8Untitled_2.png)
    
    Untitled
    
    - C - penalty for each misclassified point
        - Low C = the more hard margin the model is
        - High C = the more soft margin the model is
- How does the polynomial degree, C, and gamma affects overfitting and under fitting?
    - If your model is overfitting you should decrease the polynomial degree, gamma, and C
    - If your model is under fitting you should increase the polynomial degree, gamma, and C
- How do you choose which kernel to use?
    - Start with linear kernel especially if the training set is large
    - LinearSVC is faster then SVC(kernel=’linear’)
    - Gaussian RBF kernel works most of the time
    - Experiment with other kernels
- Explain the Computational Complexity for Support Vector Machines
    - m = data points, n = features
    - LinearSVC = O(m x n)
    - SVC = O(m^2 * n) or O(m^3 * n)
- Explain how you can use Support Vector Machines for Regressions
    - SVM Regression tries to fit as many instances as possible on the street while limiting margin violations which are instances off the street
    - The width of the street is controlled by the hyper parameter epsilon
        - Larger the epsilon the wider the street
        - Smaller the epsilon the narrower the street

### Exercises

---

- What is the fundamental idea behind Support Vector Machines?
    - The fundamental idea behind Support Vector Machines is to create a decision line between two classes so it can classify new instances within those classes
- Why is it important to scale the inputs when using SVMs?
    - You have to scale the inputs when using SVMs because the mean will mess up the calculations of the decision line
- Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?
    - No it cannot output a confidence score nor a probability because it is only able to output binary results based on its calculations
- Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?
    - This question only refers to Linear SVM since kernels can only be done with the dual form
    - The dual form makes kernels available to use which slows down the process and with an instances of hundreds of features you would want to want a faster computation which would be the primal form.
    - The computational complexity for the dual form is m^2 or m^3
- Say you’ve trained an SVM classifier with an RBF kernel, but it seems to under fit the training set. Should you increase or decrease γ (gamma)? What about C?
    - We should increase the penalty for each instance the model gets wrong which we can do by increasing the hyper parameter C
    - We can also increase the gamma so the curves for the landmarks can be narrower