# Movie Analysis and Prediction Project

## Overview
This project analyzes the TMDB 5000 Movie Dataset to explore relationships between movie features and ratings, and builds predictive models for both movie ratings (regression) and rating classifications (classification). The analysis provides insights into what factors contribute to successful movies and demonstrates various data science techniques from data cleaning to model deployment.

## Dataset
The project uses two datasets from TMDB (The Movie Database):
- `tmdb_5000_credits.csv`: Contains information about movie credits including cast and crew
- `tmdb_5000_movies.csv`: Contains movie metadata such as budget, genres, popularity, and ratings

## Features
- Data cleaning and preprocessing
- Feature engineering including weighted rating calculation
- Exploratory data analysis with visualizations
- Correlation analysis between movie features
- Genre-based popularity analysis
- Machine learning models for regression and classification

## Techniques Used

### Data Preprocessing
- Merging datasets
- Handling missing values
- JSON parsing using the `ast` module
- Feature extraction from nested data structures
- Data transformation for modeling

### Exploratory Data Analysis
- Visualization using Matplotlib and Seaborn
- Pair plots for feature relationships
- Bar charts for genre popularity
- Correlation heatmaps
- Distribution analysis of movie features

### Feature Engineering
- Creation of weighted rating system considering both vote count and average
- Classification of movies into rating categories (High, Average, Low)
- Genre extraction and one-hot encoding

### Machine Learning
- Regression models to predict movie ratings
- Classification models to predict rating categories
- Pipeline construction with preprocessing steps
- Model evaluation and comparison
- Hyperparameter tuning with GridSearchCV

## Models Implemented

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Neighbors Regressor
- Support Vector Regression (SVR)

### Classification Models
- Logistic Regression
- K-Nearest Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## Results
The project evaluates each model based on appropriate metrics:
- Regression models: Mean Squared Error (MSE)
- Classification models: Accuracy Score

Visualizations of actual vs. predicted values provide insights into model performance. The best performing models and their optimal parameters are documented for future reference.

## Installation and Usage
```python
# Clone the repository
git clone https://github.com/yourusername/movie-analysis-project.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook movie_analysis.ipynb
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The Movie Database (TMDB) for providing the dataset
- All open-source libraries used in this project
