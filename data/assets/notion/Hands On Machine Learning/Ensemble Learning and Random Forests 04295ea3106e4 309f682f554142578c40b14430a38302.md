# Ensemble Learning and Random Forests 04295ea3106e425e8215ad919e123ee4

# Ensemble Learning and Random Forests

# Ensemble Learning and Random Forests

### Overview

---

Ensemble learning is a machine learning technique that combines several base models to improve the overall performance of the model. Random forests, which are a type of ensemble learning, are a popular algorithm used for classification and regression tasks.

Random forests work by creating multiple decision trees and combining their predictions. Each decision tree is trained on a random subset of the training data and a random subset of the features. This randomness helps to prevent overfitting and improves the generalization performance of the model.

In a classification task, the final prediction of the random forest is the majority vote of all the decision trees. In a regression task, the final prediction is the average of all the decision tree predictions.

Random forests have many advantages over other machine learning algorithms. They can handle both categorical and continuous data, they are resistant to overfitting, and they can handle missing data. They are also easy to use and interpret, making them a popular choice for many machine learning applications.

In conclusion, ensemble learning and random forests are powerful machine learning techniques that can improve the performance of models in various tasks. Random forests, in particular, are a popular and effective algorithm that can handle complex data and prevent overfitting.

### Main Ideas

---

- Explain the wisdom of the crowd concept and how it applies to machine learning
    - Asking a complex question to thousands of people and aggregating their answers if often better than an experts answer
    - Aggregating the answers of multiple models is often better than an individual model
    - Technique called Ensemble Learning
- Explain how a Random Forrest works for Regression and Classification?
    - A Random Forest Algorithm trains many Decision Trees on different subsets of the data
    - In a classification task, the final prediction of the random forest is the majority vote of all the decision trees.
    - In a regression task, the final prediction is the average of all the decision tree predictions.
- What is a majority-vote classifier?
    - When you aggregate the predictions of each classifier and pick the class that gets the most vote
- When do ensemble methods work best?
    - When the predictors are as independent from one another
    - Hence: Use different algorithms
    - each algorithm will make different types of errors
- Explain the difference between Hard Voting and Soft Voting when using voting classifiers?
    - Hard voting is majority rule voting
    - Soft voting can only be done if the models are able to estimate class probabilities and it will predict the class with the highest class probability averaged over all the individual classifiers.
- What is bagging and pasting?
    - Bagging = sampling with replacement = bootstrap
    - Pasting = sampling without replacement
- What is the out of bag evaluation and why can it serve as a validation set?
    - When bagging some of the data from the train set may never be seen, as a result we can use the remaining data as a validation set to get an idea on how the model will preform on the test set
- Explain Random Patches and Random Subspaces?
    - Random Patches - Samples both training instances and features
    - Random Subspaces - Samples only features
- Explain Extremely Randomized Trees or Extra-Trees?
    - An Extremely Random Forest that is using a random subset of features and a random thresholds for each features
    - Regular Decision Trees searches for the best thresholds so in this way, the training process is sped up
- What is the general idea of boosting?
    - Any Ensemble method that combines several weak learners into a strong learner
    - Train predictors sequentially, each trying to correct its predecessor
- Explain AdaBoost
    - AdaBoost sequentially trains models that corrects its predecessor by focusing on the instances that the predecessor under fitted
    - To make predictions, AdaBoost computes the predictions of all the models and weights them using the predictor weights. Whichever class receives the majority of weighted votes is the result
- Explain Gradient Boosting
    - Gradient Boosting sequentially trains models that corrects its predecessors by focusing on the residual
- Explain Stacking
    - Stacking is a ensemble method that creates several baseline models which predictions are fed to another model (Blending) to determine the final result
    
    ![](Ensemble%20Learning%20and%20Random%20Forests%2004295ea3106e4/Hands_on_Machine_Learning_7651e7a86bdf4776a31b1679829bcbaaEnsemble_Learning_and_Random_Forests_04295ea3106e425e8215ad919e123ee4Untitled.png)
    
    Untitled
    

### Exercises

---

- If you have trained five different models on the exact same training data, and they all achieve 95% precision, is there any chance that you can combine these models to get better results? If so, how? If not, why?
    - Yes, only if the models are independent of each other
    - This method is called ensemble and it is where you combine different models together in hopes to get a better result
    - You would use sklearn ensemble bagging method to combine all the models together and let it run
- What is the difference between hard and soft voting classifiers?
    - Hard Voting Classifiers are when the end prediction or classification is determined by whichever class got the highest number of votes
    - Soft Voting Classifiers are when the end prediction or classification is determined by taking the class with the highest probability and returning the average probability for that class among the different models used
- Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers? What about pasting ensembles, boosting ensembles, Random Forests, or stacking ensembles?
    - Distributing the work for ensembles across multiple servers will speed up the training process
- What is the benefit of out-of-bag evaluation?
    - Out-of-bag is a subset of the training set when using bagging or sampling without replacement. The idea is that some instances will not be sampled so those instances can serve as a validation set to predict how well the model will preform on the test set.
    - When running the model on the out-of-bag set we get to see how the model is working on data it has never seen before.
- What makes Extra-Trees more random than regular Random Forests? How can this extra randomness help? Are Extra-Trees slower or faster than regular Random Forests?
    - Extra-Trees are more random than regular Random Forests Trees because in Random Forest, the algorithm tries to find the best thresholds for each feature that is being used, however, for Extra-Trees the thresholds is determined at random speeding up the training process faster then regular Random Forests.
- If your AdaBoost ensemble under fits the training data, which hyper parameters should you tweak and how?
    - The hyper parameters we can tweak are the learning rate. By increasing the learning rate the model will fit the data more tightly with fewer iterations. Another hyper parameter we can increase is the number of trees since more trees will allow more complicated patterns to be fitted.
- If your Gradient Boosting ensemble under fit the training set, should you increase or decrease the learning rate?
    - If the Gradient Boosting ensemble is under fitting the training data the hyper parameter to tweak is the learning rate by decreasing it.
    - Gradient Boosting works by training multiple models that is trained by the data by the residual from the previous model. So if the Gradient Boosting ensemble is under fitting the training data that must mean the learning rate is too high.