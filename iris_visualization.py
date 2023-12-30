import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from a local CSV file
file_path = "Iris_data.csv"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_df = pd.read_csv(file_path, header=None, names=column_names)

# Display basic information about the dataset
print("Basic Information about the Iris Dataset:")
print(iris_df.info())
print()

# Display the first few rows of the dataset
print("First Few Rows of the Iris Dataset:")
print(iris_df.head())
print()

# Descriptive statistics
print("Descriptive Statistics of the Iris Dataset:")
print(iris_df.describe())
print()

# Visualization :- 
# Scatter plot of sepal length vs sepal width
plt.figure(figsize=(8, 6))
plt.scatter(iris_df["sepal_length"], iris_df["sepal_width"], c=iris_df["class"].astype("category").cat.codes, cmap="viridis")
plt.title("Scatter Plot of Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Analysis of Scatter Plot
print("Analysis of Scatter Plot:")
print("The scatter plot shows the relationship between sepal length and sepal width.")
print("It appears that there are distinct clusters for each class (setosa, versicolor, virginica).")
print("Setosa generally has smaller sepal length and width, while virginica tends to have larger dimensions.")
print()

# Histogram for each feature
iris_df.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms for Iris Dataset Features", y=0.95)
plt.show()

# Analysis of Histograms
print("Analysis of Histograms:")
print("The histograms provide insights into the distribution of each feature.")
print("Petal length and petal width show distinct peaks for each class, indicating potential differences.")
print("Sepal length and sepal width also have variations across the classes.")
print()

# Pair plot to visualize relationships between features
sns.pairplot(iris_df, hue="class", palette="viridis")
plt.suptitle("Pair Plot of Iris Dataset Features", y=1.02)
plt.show()

# Analysis of Pair Plot
print("Analysis of Pair Plot:")
print("The pair plot allows us to visualize relationships between all pairs of features.")
print("Diagonal plots show histograms, and scatter plots show relationships between different features.")
print("Classes are color-coded, and we can observe patterns and differences.")
print()

# Box plot to visualize the distribution of each feature by class
plt.figure(figsize=(12, 8))
sns.boxplot(x="class", y="sepal_length", data=iris_df)
plt.title("Box Plot of Sepal Length by Class")
plt.show()

# Analysis of Box Plot
print("Analysis of Box Plot:")
print("The box plot illustrates the distribution of sepal length for each class.")
print("We can observe differences in the median and spread across different iris species.")
print("This plot helps us identify potential outliers and variations in sepal length.")
print()


# Correlation Matrix (excluding non-numeric column)
numeric_columns = iris_df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = iris_df[numeric_columns].corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Heatmap of the Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Value Counts of Classes
class_counts = iris_df["class"].value_counts()
print("Class Distribution:")
print(class_counts)

# Groupby and Aggregation
groupby_class = iris_df.groupby("class")
class_statistics = groupby_class.agg({"sepal_length": "mean", "petal_length": "median"})
print("Class-wise Statistics:")
print(class_statistics)

# Crosstabulation
cross_tab = pd.crosstab(iris_df["class"], iris_df["sepal_width"])
print("Cross-Tabulation:")
print(cross_tab)
