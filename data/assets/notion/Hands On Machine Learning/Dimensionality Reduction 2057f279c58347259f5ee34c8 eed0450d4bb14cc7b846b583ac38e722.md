# Dimensionality Reduction 2057f279c58347259f5ee34c839bf308

# Dimensionality Reduction

Dimensionality reduction is a technique used to reduce the number of features in a dataset while preserving as much relevant information as possible. This is often done to improve the efficiency of machine learning algorithms, reduce computational costs, and avoid overfitting.

There are several methods for dimensionality reduction, including **Principal Component Analysis (PCA)**, **t-distributed Stochastic Neighbor Embedding (t-SNE)**, and **Linear Discriminant Analysis (LDA)**. PCA is a popular method that identifies the most important features in a dataset by projecting them onto a lower-dimensional space, while t-SNE is often used for visualizing high-dimensional data in a two- or three-dimensional space. LDA, on the other hand, is a supervised learning algorithm that finds a linear combination of features that separates the classes in a dataset.

When applying dimensionality reduction, it is important to choose the right method for the task at hand and to carefully tune the parameters of the algorithm. It is also important to evaluate the performance of the machine learning model on the reduced dataset, as the reduction process may result in a loss of information.

Overall, dimensionality reduction is a useful tool for working with high-dimensional datasets and can improve the efficiency and accuracy of machine learning algorithms.

### Main Ideas

---

- Explain the curse of dimensionality
    - Machine learning models are trained on datasets to many features which makes the training process slow
    - The more features they are the more dimensions they are so data that are graphically close to each other will actually be really far from other.
- What is Dimensional Reduction?
    - technique used to reduce the number of features in a dataset while preserving as much relevant information as possible
- What is projection vs manifold learning?
    - projecting is when you take a lower-dimensional subspace of a high-dimensional space
    - Manifold is when you can bent and twist a shape in a higher dimensional space
- What are the two main approaches to reducing dimensionality?
    - Projection - Most of the data lies within or close to a lower-dimensional subspace of the higher dimensional space. So we project every training instance perpendicularly onto this subspace to reduce dimensionality
    
    ![](Dimensionality%20Reduction%202057f279c58347259f5ee34c8/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaDimensionality_Reduction_2057f279c58347259f5ee34c839bf308Untitled.png)
    
    Untitled
    
    - Manifold Learning - given high-dimensional embedded data, it seeks a low-dimensional representation of the data that preserves certain relationships within the data
    
    ![](Dimensionality%20Reduction%202057f279c58347259f5ee34c8/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaDimensionality_Reduction_2057f279c58347259f5ee34c839bf308Untitled_1.png)
    
    Untitled
    
- Explain the steps of the PCA dimension reduction algorithm
    - PCA dimension reduction algorithm tries to reduce the dimensions while keeping as much information as it can
    - Steps
        - Find principal components
            - Projects the data onto a lower-dimensional hyperplane by picking the hyperplane that has the highest variance
            - Finds a second axis that accounts for the largest remaining amount of variance and continue to do so for as many dimensions in the dataset
        - Projects the first x dimensions onto a hyperplane to keep as much variance as possible and reduce dimensions
            - x = any amount
- How do you choose the right number of dimensions for PCA?
    - choose the number of dimensions that add up to a sufficiently large portion of the variance like 95%
    - Three ways to do this
        - run PCA without reducing dimensionality and then computing the minimum number of dimensions required
        
        ```python
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(X_train)
        cumsum = np.cumsum(
            pca.explained_variance_ratio_
        )
        d = np.argmax(cumsum >= 0.95) + 1
        ```
        
        - set n_components to a float value that is equal to the amount of variance you want to preserve
        
        ```python
        from sklearn.decomposition import PCA
        pca = PCA(n_componenets=0.95)
        ```
        
        - Plot to explained variance as a function of number of dimensions
- How is randomized PCA faster than PCA?
    - Randomized PCA is a stochastic algorithm that quickly finds an approximation of the first d principal components
    - The computational complexity is O(m x d^2) + O(d^3)
    - Full SVD computational complexity is O(m x n^2) + O(n^3)
- Explain Incremental PCA and its advantage over PCA?
    - PCA requires the whole training set to be in memory
    - Incremental PCA allows to split the training set into mini-batches which is helpful for large datasets and for applying PCA online
- Explain Kernel PCA
    - When the Kernel trick is applied to PCA allowing it to preform complex nonlinear projections for dimensionality reduction
- Explain Locally Linear Embedding (LLE)
    - powerful nonlinear dimensionality reduction technique
    - works by measuring how each training instance linearly relates to its closest neighbors and then looks for a low dimensional representation where the local relationships are best preserved

### Exercises

---

- What are the main motivations for reducing a dataset’s dimensionality? What are the main drawbacks?
    - Motivations are to increase training speeds and to be able to generate good visuals representations of the data since we can only visualize up to 3 dimensions at a time
    - The main drawbacks is that by decreasing the dimensions we are exposing ourselves to loosing data and decreasing the quality of the dataset
- What is the curse of dimensionality?
    - Machine learning models are trained on datasets to many features which makes the training process slow
    - The more features they are the more dimensions they are so data that are graphically close to each other will actually be really far from other.
- Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?
    - No because we are removing information and depending on the algorithm it might be impossible to get it back
    - Some algorithms have a reverse the operation but you will still loose some information in the data.
- Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?
    - Yes PCA can be used to reduce the dimension of any dataset
- Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?
    - There are three ways to choosing the right number of dimensions to achieve a variance of 95%
        - Using an equation
        - Setting the n_components to a float value of 0.95 to represent we want to achieve a variance of 95%
        - Plotting the graph and seeing where the elbow is
- In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?
    - Incremental PCA - When using out of core memory
    - Randomized PCA - When you want to speed up training time and you are just looking for a good enough solution. Usually the first few principal components.
    - Kernel PCA - When you have a nonlinear dataset
    - Vanilla PCA - When you have a linear dataset that fits into core memory and you do not have to worry about training time
- How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?
    - You can run a ML model on it and evaluate that to see how well it is working
- Does it make any sense to chain two different dimensionality reduction algorithms?
    - Only in the case if one dimensionality reduction algorithm achieves a manifold
    - Then a manifold learning dimensionality reduction algorithm will need to be used