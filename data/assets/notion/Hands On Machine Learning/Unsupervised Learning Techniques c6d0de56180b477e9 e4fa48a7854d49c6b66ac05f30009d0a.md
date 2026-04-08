# Unsupervised Learning Techniques c6d0de56180b477e938d78ebcf8482d9

# Unsupervised Learning Techniques

### Main Ideas

---

- What are the most common unsupervised learning task?
    - Dimensionality Reduction
    - Clustering
    - Anomaly Detection
    - Density Estimation
- What is clustering and how can it help with the following applications?
    - Clustering is identifying similar instances and grouping them
    - Customer Segmentation
        - cluster based on purchases and activity to understand who your customers are and their needs
    - Data Analysis
        - When analyzing a new dataset, you can cluster the data and analyze each cluster separately
    - Anomaly Detection or Outlier Detection
        - By clustering and finding any instances that has a low affinity to all the clusters would be an outlier
    - Semi-Supervised Learning
        - By reforming clustering you can share the labels to all the instances without labels in the same cluster
    - Search Engines
        - By letting you search using a reference object, it will simply return all the objects in the same cluster as the reference
- What is the difference between hard clustering and soft clustering?
    - Hard Clustering: Assign an instance to a single cluster
    - Soft Clustering: Assign Instances a score per cluster
- Explain the steps of K-Means Algorithm
    - The algorithm plots the instances on a graph
    - Places k centroids on the graph and clusters the instances
    - Re-Clusters the instances by the mean of the previous cluster
    - Repeats until assignments does not change
- What are the different centroid initialization methods?
    - The centroids can be initialized randomly
        - This can cause the algorithm to be stuck in a local minimum when trying to converge providing a sub-optimal solution
        - When using this method the algorithm has to be run multiple times and the best one should be kept
        - The best one is determined by the inertia which is the mean squared distance between each instance and its closest cluster
    - If you have an idea of where the centroids should be you can set the coordinates for them to be initialized on the plot
        - Because the centroids are being initialized at a given location the model only has to run once
- How can you find the optimal number of clusters?
    - Elbow Method
        - Plot the inertia for different Ks
        - Find the elbow or the point were the inertia starts to decrease less and that might be a good K value to use
        - inertia will naturally decrease as K increases because of how it is calculated
    - Silhouette Score Method
        - Compute the Silhouette Coefficient for multiple Ks and go with the highest value.
- What are some limits of K-Means?
    - Needs to run multiple times to find optimal solution
    - does not work well when clusters have different sizes, different densities, and non-spherical shapes
- Explain Active Learning and its steps?
    - When an human expert interacts with the learning algorithm
    - Steps of Active Learning
        1. The model is trained on the labeled instances gathered so far
        2. Make predictions on all the instances that are not labeled
        3. The instances in which the probability is the lowest is given to an expert to be labeled
        4. Repeat until performance improvements stop being worth the labeling effort
- Explain how the DBSCAN clustering algorithm works?
    - For each instance, the algorithm will count its neighbors within a small distance
    - If the neighbors are equal to or greater than a certain number, then the instance becomes a core instance, and all the neighbors and itself form a cluster
        - Core instances are those that are located in dense regions
    - Any instances that does not have a core instance in its neighborhood is considered an anomaly
- What can Gaussian Mixture Models be used for?
    - Density Estimation
    - Clustering
    - Anomaly Detection
- What are Gaussian Mixtures Models?
    - A probabilistic model that assumes every instances belongs to a Gaussian Distribution whose parameters are unknown
    - All instances from the same Gaussian Distribution will form a cluster
- How can Gaussian Mixture Models be used for Anomaly Detection?
    - Any instance located in a low-density region can be considered an anomaly
- What is the difference between novelty detection and anomaly detection?
    - Anomaly Detection is often used to clean up a dataset
    - Novelty Detection is trained on a clean dataset to find outliers or anomalies in new arriving data
- How can Gaussian Mixture Models be affected by outliers?
    - If they’re many outliers in the dataset, the model might wrongly consider them as normal
    - In this case you can fit the model once to detect and remove the most extreme outliers
    - Then fit the model again on a cleaned up dataset
- Explain Bayesian Gaussian Mixture Models?
    - Works similar to Gaussian Mixture Models where it understands the instances are generated from Gaussian Distributions
    - It gives weights to clusters which helps it automatically find the best number of clusters needed
    - Clusters with a weight close to or at zero means that the cluster is unnecessary

### Questions

---

- How would you define clustering? Can you name a few clustering algorithms?
    - Clustering - grouping of objects based on similar features that are not labeled
    - K-Means and DBSCAN
- What are some of the main applications of clustering algorithms?
    - Dimension reduction
    - Anomaly Detection
    - Clustering
    - Density Estimation
- Describe two techniques to select the right number of clusters when using K-Means
    - Elbow Method
        - Trying different K number of clusters and plotting the inertia to spot where the elbow is. The elbow would represent a good number of clusters to have.
    - Silhouette Coefficient
        - Getting the Silhouette score for each number of clusters used. The highest score is a good number of clusters to have for the dataset.
- What is label propagation? Why would you implement it, and how?
    - Label propagation is when you cluster a dataset where most of the features are not labeled. After clustering you assign the label to all of the members in each cluster
    - Implementing label propagation helps us create more labeled instances for us to train a dataset and it helps decrease expert intervention to label the data them selfs saving time.
- Can you name two clustering algorithms that can scale to large datasets and two that look for regions of high density?
    - Clustering
        - K-Means
        - DBSCAN
    - Looks for high density
        - Gaussian Mixture Models
        - Bayesian Gaussian Mixture Models
- Can you think of a use case where active learning would be useful? How would you implement it?
    - Active learning will be useful when we are training a model to preform really well on data that is usually not labeled. After every iteration we have an expert intervene correcting the model and feeding it more labeled data to learn from until it any more expert intervention would be considered a waste of time.
- What is the difference between anomaly detection and novelty detection?
    - Anomaly Detection - Usually used to clean up a dataset and find outliers to drop. Focused on understanding what abnormal data looks like.
    - Novelty Detection - used on a cleaned up version of the dataset to inspect new data coming in that does not seem normal. Focused on understanding what normal data looks like.
- What is Gaussian Mixture and what can you use it for?
    - Gaussian Mixture models thinks of all the instances in the dataset to be apart of a Gaussian Distribution. When thinking of instances in this manner, we can use Gaussian Mixtures for anomaly detection and clustering.