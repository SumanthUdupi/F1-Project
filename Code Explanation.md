# F1 Data Science Project
---
## Data Science Learning Path
---
### 1. Mathematics for Data Science
   - Algebra: Linear equations, matrix operations.
   - Calculus: Derivatives, optimization.
   - Probability fundamentals: Basic probability, conditional probability.
---
### 2. Statistics Basics
   - Descriptive statistics: Mean, median, mode, standard deviation, variance.
   - Probability distributions: Normal, binomial, Poisson.
   - Inferential statistics: Sampling, estimation.
---
### 3. Python Programming
   - Data types and structures: Lists, tuples, dictionaries, sets.
   - Control structures: Loops, conditionals.
   - Functions and modules.
   - Data science libraries: Pandas, Numpy, Matplotlib.
---
### 4. Data Wrangling and Cleaning
   - Handling missing values: Imputation, removal.
   - Handling outliers: Detection and treatment.
   - Data transformation: Scaling, normalization, encoding categorical variables.
---
### 5. Exploratory Data Analysis (EDA)
   - Summary statistics: Mean, median, skewness, kurtosis.
   - Data visualization: Histogram, boxplot, pairplot.
   - Feature relationships: Correlation analysis, scatter plots.
---
### 6. Probability and Probability Distributions
   - Basic probability: Rules of probability, Bayes' theorem.
   - Probability distributions: Normal, binomial, Poisson, uniform distributions.
   - Sampling methods: Random, stratified, cluster sampling.
---
### 7. Hypothesis Testing
   - Basics: Null and alternative hypotheses.
   - p-values and confidence intervals.
   - Types of tests: t-tests, chi-square tests, ANOVA.
---
### 8. Data Visualization
   - Basic plots: Histogram, scatter plot, line plot.
   - Advanced visualizations: Heatmap, pairplot, violin plot.
   - Interactive visualizations: Plotly, Dash, Tableau basics.
---
### 9. Linear Regression
   - Simple linear regression.
   - Multiple linear regression.
   - Evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared.
---
### 10. Logistic Regression
- Binary classification.
- Sigmoid function and decision boundary.
- Model interpretation and performance metrics.
---
### 11. Decision Trees and Random Forests
- Basics of decision trees: Splitting, pruning, information gain.
- Random forests: Ensemble learning, bagging.
- Hyperparameters for tuning: Max depth, min samples split.
---
### 12. k-Nearest Neighbors (kNN)
- Distance metrics: Euclidean, Manhattan.
- Choosing k and model performance.
- Applications: Classification, regression.
---
### 13. Model Evaluation Metrics
- Classification metrics: Accuracy, precision, recall, F1-score, ROC-AUC.
- Regression metrics: MAE, MSE, RMSE.
- Cross-validation techniques.
---
### 14. Clustering Algorithms
- K-means clustering: Choosing k, cluster evaluation.
- Hierarchical clustering: Dendrograms, agglomerative and divisive methods.
- Evaluation metrics: Silhouette score, Davies-Bouldin index.
---
### 15. Dimensionality Reduction
- Principal Component Analysis (PCA): Eigenvalues, eigenvectors.
- t-SNE: Visualization of high-dimensional data.
- Application of dimensionality reduction in preprocessing.
---
### 16. Hyperparameter Tuning
- Grid Search and Random Search.
- Cross-validation: k-Fold, Leave-One-Out.
- Tuning with libraries: Scikit-Learn’s GridSearchCV.
---
### 17. Neural Networks
- Basics of neural networks: Perceptron, activation functions.
- Backpropagation and gradient descent.
- Types of layers: Input, hidden, output.
---
### 18. Deep Learning with CNNs and RNNs
- Convolutional Neural Networks (CNNs): Convolutional layers, pooling.
- Recurrent Neural Networks (RNNs): Sequence data, LSTM, GRU.
- Applications: Image classification, natural language processing.
---
### 19. Natural Language Processing (NLP)
- Text preprocessing: Tokenization, stemming, lemmatization.
- Vectorization methods: Bag-of-Words, TF-IDF.
- Advanced NLP: Word embeddings, language models (BERT, GPT).
---
### 20. Model Deployment and Monitoring
- Model deployment: Flask, FastAPI, Docker.
- Cloud platforms: AWS, GCP, Azure for model deployment.
- Monitoring models: Performance tracking, retraining triggers.
---
### Loading of Data, Libraries and Table.

#### List of tables and columns.

| Table                     | Columns                                                                                                                                                                        |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **circuits_df**           | `circuitId`, `circuitRef`, `name`, `location`, `country`, `lat`, `lng`, `alt`, `url`                                                                                           |
| **constructor_results_df** | `constructorResultsId`, `raceId`, `constructorId`, `points`, `status`                                                                                                          |
| **constructor_standings_df** | `constructorStandingsId`, `raceId`, `constructorId`, `points`, `position`, `positionText`, `wins`                                                                        |
| **lap_times_df**          | `raceId`, `driverId`, `lap`, `position`, `time`, `milliseconds`                                                                                                                |
| **pit_stops_df**          | `raceId`, `driverId`, `stop`, `lap`, `time`, `duration`, `milliseconds`                                                                                                        |
| **qualifying_df**         | `qualifyId`, `raceId`, `driverId`, `constructorId`, `number`, `position`, `q1`, `q2`, `q3`                                                                                     |
| **results_df**            | `resultId`, `raceId`, `driverId`, `constructorId`, `number`, `grid`, `position`, `positionText`, `positionOrder`, `points`, `laps`, `time`, `milliseconds`, `fastestLap`, `rank`, `fastestLapTime`, `fastestLapSpeed`, `statusId` |
| **seasons_df**            | `year`, `url`                                                                                                                                                                  |
| **sprint_results_df**     | `resultId`, `raceId`, `driverId`, `constructorId`, `number`, `grid`, `position`, `positionText`, `positionOrder`, `points`, `laps`, `time`, `milliseconds`, `fastestLap`, `fastestLapTime`, `statusId` |
| **status_df**             | `statusId`, `status`                                                                                                                                                           |
| **drivers_df**            | `driverId`, `driverRef`, `number`, `code`, `forename`, `surname`, `dob`, `nationality`, `url`                                                                                  |
| **races_df**              | `raceId`, `year`, `round`, `circuitId`, `name`, `date`, `time`, `url`, `fp1_date`, `fp1_time`, `fp2_date`, `fp2_time`, `fp3_date`, `fp3_time`, `quali_date`, `quali_time`, `sprint_date`, `sprint_time` |
| **constructors_df**       | `constructorId`, `constructorRef`, `name`, `nationality`, `url`                                                                                                                |
| **driver_standings_df**   | `driverStandingsId`, `raceId`, `driverId`, `points`, `position`, `positionText`, `wins`                                                                                        |

#### Loading of Libraries.

1. **Importing Libraries:**

  -   **Pandas** (`import pandas as pd`):  
  Pandas is like an advanced spreadsheet tool that allows us to load, manipulate, and analyze large sets of data quickly.

  -   **Seaborn** (`import seaborn as sns`):  
  Seaborn is a tool for making nice-looking charts and graphs. It builds on top of another tool (Matplotlib) to make visualizations prettier and easier to create.

  -   **Matplotlib** (`import matplotlib.pyplot as plt` and `import matplotlib`):  
  This is a library for creating plots and charts in Python. Think of it like drawing tools that help us visualize data.

  -   **NumPy** (`import numpy as np`):  
  NumPy is used for handling numbers and calculations in a more efficient way. It’s great for working with large groups of numbers, especially in math-heavy tasks.

  -   **Scikit-Learn** (`from sklearn.model_selection import train_test_split`):  
  This is a popular library for machine learning. It helps split data into training and testing parts, which is a key step in training predictive models.

2. **Setting Up Warnings:**

   ```python
   import warnings
   warnings.simplefilter("ignore")


3. **Printing Version Information:**

   ```python
   print("Pandas version:", pd.__version__)
   print("Seaborn version:", sns.__version__)
   print("Matplotlib version:", matplotlib.__version__)
   print("NumPy version:", np.__version__)

These lines display the versions of each library in use, which helps in keeping track of the exact setup, since different versions might have small differences in functionality.

### Loading of Data.

1. **Creating a Dictionary of DataFrames**:
   - A dictionary called `dataframes` is created, where each key-value pair represents a table name and its corresponding DataFrame. This setup makes it easy to iterate over multiple tables.

2. **Looping to Display Shapes and Sample Rows**:
   - The first loop iterates over each DataFrame in the dictionary, printing:
     - The name of the DataFrame.
     - The shape of the DataFrame, which shows the number of rows and columns.
     - The first few rows of data using `head()`, giving a sample preview of the data.

3. **Looping to Display Column Names**:
   - The second loop iterates over each DataFrame again to print:
     - The name of each table.
     - A list of the column names in each DataFrame.
     - This part is helpful for understanding the structure of each table and identifying available fields for analysis or further processing.