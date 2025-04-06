import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("cleaned/restoration_ready_data.csv")
df.columns = df.columns.str.strip() 
print("Columns found in dataset:")
print(df.columns.tolist())

#Check if the 'NeedsRestoration' column exists in the DF
if "NeedsRestoration" not in df.columns:
    raise ValueError("'NeedsRestoration' column not found in dataset.")


# X removes the target variable and y is the target variable
X = df.drop("NeedsRestoration", axis=1)
y = df["NeedsRestoration"]
X = pd.get_dummies(X)

# Use the Discrete Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)


#plot the decision tree with some basic rules
plt.figure(figsize=(24, 14))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=12,
    precision=2
)
plt.title("Decision Tree - NeedsRestoration", fontsize=18, pad=20)
plt.tight_layout(pad=4.0)
plt.savefig("decision_tree_visualization_clear.png")
plt.show()

#export the decision tree as an image
print("Decision Tree Rules:\n")
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
