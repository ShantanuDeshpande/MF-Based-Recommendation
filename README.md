# MF-Based-Recommendation
Recommendation System based on Matrix Factorization

Recommendation System for rating movies based on Matrix Factorization.
Recommender.java can be used to predict rating for given user-movie pair. Command-line usage -
java Recommender ratings.csv toRate.csv latent-factor learning-rate regularization
e.g - java Recommender ratings.csv toBeRated.csv 5 0.001 0.1

Recommender_CV.java can be used for K-fold cross-validation to check the performance of algorithm on dataset. Command-line usage -
java Recommender_CV ratings.csv latent-factor learning-rate regularization K
e.g - java Recommender ratings.csv 5 0.001 0.1 10
