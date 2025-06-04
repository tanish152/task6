# Iris KNN Classification Project

## Overview
This project implements a K-Nearest Neighbors (KNN) classification model to predict Iris flower species using the classic Iris dataset. As a data analyst, I designed this project to demonstrate advanced machine learning techniques, including feature normalization, model evaluation with multiple metrics, cross-validation, and interactive visualizations of decision boundaries. The project experiments with various K values to optimize performance and provides insights into the dataset's structure and model behavior.

### Key Objectives
- **Data Preprocessing**: Clean and normalize the Iris dataset to ensure robust model performance.
- **Model Training**: Use KNN with different K values to classify Iris species (Setosa, Versicolor, Virginica).
- **Evaluation**: Assess model performance using accuracy, precision, recall, F1-score, and confusion matrices, with cross-validation for robustness.
- **Visualization**: Create interactive decision boundary plots for all feature pairs using Plotly to explore model behavior.
- **Insights**: Analyze model performance and dataset characteristics to identify optimal K values and feature separability.

## Dataset
The Iris dataset contains 150 samples with four features:
- **SepalLengthCm**: Sepal length in centimeters
- **SepalWidthCm**: Sepal width in centimeters
- **PetalLengthCm**: Petal length in centimeters
- **PetalWidthCm**: Petal width in centimeters
- **Species**: Target variable (Iris-setosa, Iris-versicolor, Iris-virginica)

The dataset is stored in `Iris.csv` and is cleaned to remove outliers and missing values.

## Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas**: Data loading and preprocessing
- **NumPy**: Numerical computations
- **Scikit-learn**: KNN classifier, feature scaling, and evaluation metrics
- **Plotly**: Interactive visualizations of decision boundaries
- **Seaborn/Matplotlib**: (Optional, for static confusion matrix plots in earlier versions)
- **Jupyter Notebook** (optional): For interactive development and testing

## Prerequisites
Ensure the following are installed:
- Python 3.8 or higher
- Required Python libraries:
  ```bash
  pip install pandas numpy scikit-learn plotly
  ```
- The `Iris.csv` dataset file in the project directory

## Steps to Run the Project
1. **Clone or Download the Repository**:
   - Clone the project or download the `iris_knn_classification_enhanced.py` and `Iris.csv` files.
   ```bash
   git clone <repository-url>
   ```

2. **Set Up the Environment**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install pandas numpy scikit-learn plotly
     ```

3. **Prepare the Dataset**:
   - Ensure `Iris.csv` is in the same directory as `iris_knn_classification_enhanced.py`.
   - The script handles data cleaning (removing missing values and outliers).

4. **Run the Script**:
   - Execute the Python script:
     ```bash
     python iris_knn_classification_enhanced.py
     ```
   - Alternatively, run in a Jupyter Notebook by copying the code into a `.ipynb` file.

5. **View Outputs**:
   - **Console Output**: Accuracy, precision, recall, F1-score, 5-fold cross-validation scores, and confusion matrices for K values [1, 3, 5, 7, 9, 11, 15].
   - **Interactive Plots**: Plotly visualizations of decision boundaries for all feature pairs (e.g., PetalLengthCm vs. PetalWidthCm) for selected K values.
   - **Best K**: The script identifies the K value with the highest test accuracy.

## Project Workflow
1. **Data Loading and Cleaning**:
   - Load `Iris.csv` using Pandas.
   - Drop the `Id` column, remove missing values, and filter outliers using the Interquartile Range (IQR) method (1.5 * IQR rule).
   - Validate species entries to ensure only valid classes are included.

2. **Feature Normalization**:
   - Standardize features using `StandardScaler` to ensure zero mean and unit variance, critical for KNN's distance-based algorithm.

3. **Model Training and Evaluation**:
   - Split data into 70% training and 30% testing sets.
   - Train KNN models with K values [1, 3, 5, 7, 9, 11, 15].
   - Evaluate using:
     - **Accuracy**: Proportion of correct predictions.
     - **Precision, Recall, F1-Score**: Weighted averages for multi-class performance.
     - **Confusion Matrix**: Shows true vs. predicted classes.
     - **5-Fold Cross-Validation**: Assesses model robustness with mean accuracy and standard deviation.

4. **Visualization**:
   - Generate interactive Plotly contour plots for decision boundaries across all feature pairs (6 combinations).
   - Each plot shows normalized feature values, training points, and decision regions for K values [3, 5, 7, 9].

5. **Analysis**:
   - Identify the best K value based on test accuracy.
   - Analyze confusion matrices to detect misclassifications, particularly between Iris-versicolor and Iris-virginica due to feature overlap.

## Key Insights
- **Dataset Characteristics**: Iris-setosa is highly separable, while Iris-versicolor and Iris-virginica have overlapping petal measurements, leading to potential misclassifications.
- **Model Performance**: K=5 or K=7 typically yields high accuracy (0.95–0.98) and balanced precision/recall, as confirmed by cross-validation.
- **Decision Boundaries**: Smaller K values (e.g., 1, 3) create complex boundaries (risking overfitting), while larger K values (e.g., 15) produce smoother boundaries (risking underfitting).
- **Feature Importance**: PetalLengthCm and PetalWidthCm are often the most discriminative features, as seen in decision boundary plots.
- **Ethical Considerations**: Proper data cleaning and robust evaluation ensure reliable predictions, critical for applications like botanical classification or medical diagnostics.

## Example Output
```plaintext
K=5 Results:
Accuracy: 0.9778
Precision: 0.9780
Recall: 0.9778
F1-Score: 0.9777
5-Fold CV Accuracy: 0.9667 (±0.0213)
Confusion Matrix:
[[15  0  0]
 [ 0 17  1]
 [ 0  0 12]]

Best K based on test accuracy: 5 (Accuracy: 0.9778)
```

## Interactive Visualizations
- **Decision Boundaries**: Hover over Plotly plots to inspect feature values, zoom to analyze boundaries, or pan to explore regions.
- **Feature Pairs**: All combinations (e.g., SepalLengthCm vs. PetalWidthCm) are visualized to provide a comprehensive view of model behavior.

## Future Improvements
- Add hyperparameter tuning (e.g., distance metrics like Manhattan or Euclidean).
- Incorporate feature selection techniques (e.g., PCA) to reduce dimensionality.
- Extend to other classifiers (e.g., SVM, Random Forest) for comparison.
- Add real-time prediction functionality for user-inputted Iris measurements.

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, reach out via GitHub or email.