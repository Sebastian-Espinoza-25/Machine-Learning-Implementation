import numpy as np
import pandas as pd
from collections import Counter

#Class for ID3 Decision Tree. 
class ID3DecisionTree:
    #Initialize the decision tree with hyperparameters.
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-12, random_state=None):
        #How deep the tree can go
        self.max_depth = max_depth
        #Minimum samples required to split a node
        self.min_samples_split = min_samples_split
        #Minimum information gain that is required to make a split
        self.min_gain = min_gain
        # Random seed for reproducibility
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        #Placeholder for the tree structure
        self.tree_ = None
        self.feature_types_ = None
        self.classes_ = None

    #Function to fit the decision tree to the data.
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Copy inputs to avoid modifying the original data
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)
        # Ensure X and y have the same index
        X.index = y.index
        # Validate and normalize all feature columns
        for col in X.columns:
            #Ensure all of the features in the dataset are categorical
            if pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Column '{col}' is numeric. This pure ID3 implementation accepts only categorical features.")
            mask = X[col].notna()
            # Convert to string and strip whitespace
            X.loc[mask, col] = X.loc[mask, col].astype(str).str.strip()
        self.classes_ = np.array(sorted(y.unique()))
        self.feature_types_ = {col: "categorical" for col in X.columns}
        #Build the decision tree recursively
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    #Function to predict the class labels for new data points.
    def predict(self, X: pd.DataFrame):
        # Copy input to avoid modifying the original data
        X = X.copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                # Ensure categorical features are strings and stripped of whitespace
                mask = X[col].notna()
                X.loc[mask, col] = X.loc[mask, col].astype(str).str.strip()
        #For each row in the dataset, traverse the tree to make a prediction
        preds = [self._predict_row(row, self.tree_) for _, row in X.iterrows()]
        #Return the predictions as a numpy array
        return np.array(preds)
    
    #Function to print the structure of the decision tree.
    def print_tree(self):
        #Recursively print the tree structure
        self._print_node(self.tree_)

    #Calculate entropy of a label distribution
    @staticmethod
    def _entropy(y: pd.Series) -> float:
        #Calculate relative frecuencies for each of the classes
        p = y.value_counts(normalize=True)
        #Calculate entropy using the formula, adding a small constant to avoid log(0)
        return -(p * np.log2(p + 1e-15)).sum()

    #Calculate information gain for a categorical feature
    def _information_gain_categorical(self, X_col: pd.Series, y: pd.Series):
        #Calculate the base entropy of the labels
        base_entropy = self._entropy(y)
        #Handle missing values by treating them as a separate category (Not used here because the datasets do not have missing values)
        Xc = X_col.fillna("__MISSING__")
        #Get unique values in the feature column
        values = Xc.unique()
        ent = 0.0
        split_map = {}
        #For each unique value, calculate the weighted entropy of the subset
        for v in values:
            #Create a mask for the current value
            mask = (Xc == v)
            #Subset the labels based on the mask
            y_child = y[mask]
            #Update the weighted entropy
            ent += (len(y_child) / len(y)) * self._entropy(y_child)
            #Store the mask for this value in the split map
            split_map[v] = mask
        #Information gain is the reduction in entropy after the split
        gain = base_entropy - ent
        return gain, split_map

    def _best_split(self, X: pd.DataFrame, y: pd.Series):
        best_feature, best_info, best_gain = None, None, -np.inf
        for col in X.columns:
            gain, split_map = self._information_gain_categorical(X[col], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = col
                best_info = {"children": split_map}
        if best_gain < self.min_gain or best_feature is None:
            return None, None, 0.0
        return best_feature, best_info, best_gain

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int):
        if len(y.unique()) == 1:
            return {"type": "leaf", "class": y.iloc[0]}
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or \
           (X.shape[1] == 0):
            return {"type": "leaf", "class": self._majority_class(y)}
        best_feature, split_info, gain = self._best_split(X, y)
        if best_feature is None:
            return {"type": "leaf", "class": self._majority_class(y)}
        node = {
            "type": "node",
            "feature": best_feature,
            "feature_type": "categorical",
            "gain": gain,
            "majority_class": self._majority_class(y)
        }
        node["children"] = {}
        for val, mask in split_info["children"].items():
            X_child = X[mask].drop(columns=[best_feature])
            y_child = y[mask]
            node["children"][val] = self._build_tree(X_child, y_child, depth + 1)
        return node

    def _predict_row(self, row: pd.Series, node):
        while node["type"] != "leaf":
            feat = node["feature"]
            if feat not in row.index:
                return node.get("majority_class", None)
            val = row[feat]
            key = val if pd.notna(val) else "__MISSING__"
            child = node["children"].get(key)
            if child is None:
                return node.get("majority_class", None)
            node = child
        return node["class"]

    @staticmethod
    def _majority_class(y: pd.Series):
        counts = y.value_counts()
        m = counts.max()
        return sorted(counts[counts == m].index.tolist())[0]

    def _print_node(self, node, indent=""):
        if node["type"] == "leaf":
            print(f"{indent}Leaf: predict = {node['class']}")
            return
        feat = node["feature"]
        print(f"{indent}if {feat} in ... (gain={node['gain']:.5f})")
        for val in sorted(node["children"].keys(), key=lambda x: str(x)):
            print(f"{indent}  [{feat} == {val!r}]")
            self._print_node(node["children"][val], indent + "    ")
    
    def pretty_print(self):
        self._pretty_print_node(self.tree_)
        
    def _pretty_print_node(self, node, indent="", is_last=True):
        prefix = indent + ("└── " if is_last else "├── ")
        if node["type"] == "leaf":
            print(prefix + f"Predict: {node['class']}")
            return
        feature = node["feature"]
        print(prefix + f"{feature}? (gain={node['gain']:.3f})")
        new_indent = indent + ("    " if is_last else "│   ")
        children = list(sorted(node["children"].items(), key=lambda kv: str(kv[0])))
        for i, (val, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            val_prefix = new_indent + ("└── " if is_last_child else "├── ")
            if child["type"] == "leaf":
                print(val_prefix + f"{feature} = {val} → {child['class']}")
            else:
                print(val_prefix + f"{feature} = {val}")
                self._pretty_print_node(child, new_indent + ("    " if is_last_child else "│   "), is_last_child)


def train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True):
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []

    for cls, idxs in y.groupby(y).groups.items():
        idxs = np.fromiter(idxs, dtype=int)
        if shuffle:
            rng.shuffle(idxs)
        n = len(idxs)

        if n == 1:
            # keep singleton class in train to avoid zero-shot class in training
            train_idx.extend(idxs)
            continue

        n_test = int(round(test_size * n))
        n_test = max(1, min(n_test, n - 1))  # at least 1 in each split
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    train_idx = np.array(train_idx, dtype=int)
    test_idx  = np.array(test_idx, dtype=int)

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )

def confusion_matrix_df(y_true, y_pred):
    labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    idx = {lab:i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        mat[idx[yt], idx[yp]] += 1
    return pd.DataFrame(mat, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def per_class_prf(y_true, y_pred):
    cm = confusion_matrix_df(y_true, y_pred).values
    k = cm.shape[0]
    out = []
    for i in range(k):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        out.append((prec, rec, f1))
    return out  


data = pd.read_csv("tennis.csv", dtype=str)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

print("Train size:", len(X_train), " Test size:", len(X_test))
print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

# Fit on train, evaluate on train and test
tree = ID3DecisionTree(max_depth=None, min_samples_split=2, min_gain=1e-9, random_state=0)
tree.fit(X_train, y_train)

print("\nPretty tree (train):")
tree.pretty_print()

y_train_pred = tree.predict(X_train)
y_test_pred  = tree.predict(X_test)

print("\nAccuracy")
print("  Train:", accuracy(y_train, y_train_pred))
print("  Test :", accuracy(y_test,  y_test_pred))

print("\nConfusion Matrix (Test)")
cm_test = confusion_matrix_df(y_test, y_test_pred)
print(cm_test)

