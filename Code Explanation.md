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
### Loading of Table, Libraries and Data.

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

### Loading of Libraries.

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

---

## 1. Mathematics for Data Science

### Algebra Exercises
---
1. **Filter and Analyze Circuit Locations.**
   - **Goal**: Identify all unique `location` and `country` pairs in the `circuits_df` table and create a table where each `country` has a count of `circuits` it contains.
   - **Hint**: Use grouping to count occurrences and display results.
   
   **_Solution :_** Analyzing Unique Location-Country Pairs and Circuit Counts

      This code looks at the `circuits_df` DataFrame to find unique locations and countries, as well as how many circuits each country has.
      
   **_Steps :_**

   1. **Find Unique Location-Country Pairs**:
      - The code gets unique combinations of `location` and `country` from the `circuits_df` DataFrame. This helps us see where circuits are located without duplicates.

      ```python
      unique_location_country_pairs = circuits_df[['location', 'country']].drop_duplicates()
      
   2. **Print Unique Pairs**:
      - It then prints these unique location-country pairs to show the different circuits.

      ```python
      print("Unique location-country pairs:")
      print(unique_location_country_pairs)
      
   3. **Count Circuits per Country**:
      - The code groups the data by country and counts the number of unique circuits in each country. This tells us how many circuits each country has.

      ```python
      country_circuit_count = circuits_df.groupby('country')['circuitId'].nunique().reset_index(name='circuit_count')
      
   4. **Print Circuit Count**:
      - Finally, it prints the count of circuits for each country.

      ```python
      print("Circuit count per country:")
      print(country_circuit_count)
---      
2. **Matrix of Constructor Standings**
   - **Goal**: Create a matrix showing the `position` of constructors across multiple `races`. Each row represents a `raceId` and each column represents a `constructorId` from the `constructor_standings_df` table.
   - **Hint**: Use pivoting or matrix transformation functions to reshape data.

   **_Solution:_** Creating a Constructor Position Matrix

      This code builds a matrix showing the positions of constructors across multiple races. Each row represents a race (`raceId`), and each column represents a constructor (`name`).

   **_Steps :_**

   1. **Merge Constructor Data**:
      - The `constructor_standings_df` is merged with the `constructors_df` to replace `constructorId` with the constructor's name.
      - This makes the data more understandable by using names instead of numeric IDs.

      ```python
      merged_df = constructor_standings_df.merge(
         constructors_df[['constructorId', 'name']],
         on='constructorId',
         how='left'
      )
      
   2. **Create the Matrix**:

   - The data is pivoted into a matrix where:
      - Rows (index) represent raceId (the race).
      - Columns (columns) represent name (constructor name).
      - Values (values) represent the position of each constructor in the respective race.
      
      ```python
      constructor_position_matrix = merged_df.pivot(
         index='raceId',  
         columns='name',  
         values='position'
      )

   3. **Handle Missing Values**:

   - Missing positions (where a constructor did not participate in a race) are filled with `"N/A"` for clarity.

      ```python
      constructor_position_matrix.fillna("N/A", inplace=True)

   4. **Print the Matrix**:
   The final matrix is printed to display the constructor positions across races.

      ```python
      print("Constructor Position Matrix (Rows: raceId, Columns: Constructor Name):")
      print(constructor_position_matrix)
---

3. **Sum of Points by Driver**
   - **Goal**: Calculate the total `points` each driver has scored across all races using the `results_df` table.
   - **Hint**: Group by `driverId` and use aggregation to sum the `points`.

   **_Solution:_** Summing Points Scored by Each Driver

      **_Steps :_**

   1. **Merge Driver Details with Results**
      ```python
      merged_df = results_df.merge(drivers_df[['driverId', 'forename', 'surname']], on='driverId', how='left')
      ```
      - **Purpose:** Combine race results (`results_df`) with driver details (`drivers_df`) to include the driver's first and last names alongside race data.
      - **Key Points:**
      - `on='driverId'`: The `driverId` column is used as the key for merging the two datasets.
      - `how='left'`: Ensures all rows from `results_df` are retained, even if a matching driver isn't found in `drivers_df`. Missing names will appear as `NaN`.

   2. **Create a Full Driver Name**
      ```python
      merged_df['driver_name'] = merged_df['forename'] + ' ' + merged_df['surname']
      ```
      - **Purpose:** Combine the `forename` and `surname` columns into a single column, `driver_name`, for easier readability and grouping.
      - **Result:** A new column, `driver_name`, is added to `merged_df`, containing the full name of each driver.

   3. **Calculate Total Points by Driver**
      ```python
      total_driver_points = merged_df.groupby('driver_name')['points'].sum().reset_index()
      ```
      - **Purpose:** Group the dataset by `driver_name` and calculate the total points scored by each driver across all races.
      - **Steps:**
      - `groupby('driver_name')`: Groups rows by each driver's name.
      - `['points'].sum()`: Sums the `points` column for each group (driver).
      - `reset_index()`: Converts the grouped result back into a DataFrame for easier manipulation.

   4. **Rename the Points Column**
      ```python
      total_driver_points.rename(columns={'points': 'total_points'}, inplace=True)
      ```
      - **Purpose:** Rename the `points` column in `total_driver_points` to `total_points` for clarity.
      - **Key Argument:**
      - `inplace=True`: Ensures the renaming happens directly on the `total_driver_points` DataFrame.

   5. **Sort Drivers by Total Points**
      ```python
      total_driver_points = total_driver_points.sort_values(by='total_points', ascending=False)
      ```
      - **Purpose:** Sort the drivers in descending order of their total points.
      - **Key Points:**
      - `by='total_points'`: Specifies the column used for sorting.
      - `ascending=False`: Ensures the drivers with the highest points appear first.

   6. **Display the Results**
      ```python
      print("Total Points Scored by Each Driver:")
      print(total_driver_points)
      ```
      - **Purpose:** Print the final `total_driver_points` DataFrame, showing each driver's name and their total points in descending order.

   ---

4. **Eigenvalues of a Points Matrix**
   - **Goal**: Construct a 2x2 matrix of points scored by two constructors in two races from `constructor_results_df` and compute its eigenvalues.
   - **Hint**: Choose two specific `constructorId`s and `raceId`s for simplicity.

**_Solution:_** Analyzing Constructor Points and Eigenvalues

This code calculates a matrix of points scored by two constructors in two races and computes the eigenvalues of that matrix. It uses the `constructor_results_df` DataFrame to derive the points.

**_Steps:_**

   1. **Group Data by Constructor and Race**:
   - The data is grouped by `constructorId` and `raceId`, and the total points scored by each constructor in each race are calculated.

      ```python
      selected_data = constructor_results_df.groupby(['constructorId', 'raceId'])['points'].sum().reset_index()

   2. **Select Two Constructors and Two Races**:
   - Two constructors and two races are chosen for simplicity and analysis.

      ```python
      constructors = selected_data['constructorId'].unique()[:2]
      races = selected_data['raceId'].unique()[:2]
   
   3. **Construct the Points Matrix**:

   - A 2x2 matrix is created where each cell contains the points scored by a specific constructor in a specific race. If no points are available for a combination, it is set to 0.

      ```python
         points_matrix = np.zeros((2, 2))
         for i, constructor in enumerate(constructors):
            for j, race in enumerate(races):
               points = selected_data[
                     (selected_data['constructorId'] == constructor) & 
                     (selected_data['raceId'] == race)
               ]['points']
               points_matrix[i, j] = points.values[0] if not points.empty else 0

   4. **Compute Eigenvalues**:

   - The eigenvalues of the points matrix are calculated using NumPy's eigvals function. These eigenvalues provide mathematical insights into the matrix.
         
      ```python
         eigenvalues = np.linalg.eigvals(points_matrix)

   5. **Print the Results**:

   - The constructed points matrix and its eigenvalues are displayed.

      ```python
         print("Points Matrix:")
         print(points_matrix)
         print("\nEigenvalues:")
         print(eigenvalues)

**_Explanation of the Output:_**
   
   **Points Matrix**:
   - The matrix represents the points scored by two constructors in two races.
   - Each row is a race, and each column is a constructor.
   - Example:
   - Constructor 1 scored **0 points** in Race 1 and **1 point** in Race 2.
   - Constructor 2 scored **0 points** in Race 1 and **4 points** in Race 2.

   **Eigenvalues**:
   - Eigenvalues summarize the matrix's characteristics.
   - For this matrix:
   - **0** means there’s no contribution from Constructor 1 in Race 1.
   - **4** reflects Constructor 2’s dominant score in Race 2.

   It helps quickly see which constructor performed better overall.
---
## Calculus Exercises

5. **Rate of Change in Lap Time**

   - **Goal**: Determine the rate of change in lap times for a given `driverId` by analyzing successive lap times in the `lap_times_df` table.

   - **Hint**
      - Use `sort_values()` to order the data by `driverId` and `lap` for proper calculation.
      - Use `diff()` to compute the time difference between consecutive laps within each driver's lap records.

**_Solution:_** Calculating Lap Time Change for Each Driver

**_Steps:_**

1. **Sort Lap Time Data**
   ```python
   lap_times_df = lap_times_df.sort_values(by=['driverId', 'lap'])
   ```
   - **Purpose:** Ensure the data is ordered by `driverId` and `lap` so that successive laps are properly aligned for computation.
   - **Key Points:**
     - Sorting is crucial for the `diff()` function to work as intended.
     - `by=['driverId', 'lap']`: Sorts data first by `driverId` and then by `lap` number.

2. **Calculate Rate of Change in Lap Times**
   ```python
   lap_times_df['lap_time_change'] = lap_times_df.groupby('driverId')['milliseconds'].diff()
   ```
   - **Purpose:** Compute the difference in lap times (`milliseconds`) for consecutive laps within each `driverId`.
   - **Steps:**
     - `groupby('driverId')`: Groups data by `driverId` so each driver's laps are processed independently.
     - `['milliseconds'].diff()`: Calculates the difference in lap times between consecutive laps.
   - **Result:** A new column, `lap_time_change`, is added to the DataFrame, indicating the change in lap time for each driver.

3. **Display Relevant Data**
   ```python
   print("Rate of change of lap times for each driver:")
   print(lap_times_df[['driverId', 'lap', 'milliseconds', 'lap_time_change']].tail())
   ```
   - **Purpose:** Display the relevant columns to understand the lap time changes for each driver.
   - **Key Points:**
     - `[['driverId', 'lap', 'milliseconds', 'lap_time_change']]`: Selects only the necessary columns for display.
     - `.tail()`: Shows the last few rows of the DataFrame to verify results.

---
