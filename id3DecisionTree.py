import numpy as np
import pandas as pd
from collections import Counter

class ID3DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-12, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.tree_ = None
        self.feature_types_ = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)
        X.index = y.index
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Column '{col}' is numeric. This pure ID3 implementation accepts only categorical features.")
            mask = X[col].notna()
            X.loc[mask, col] = X.loc[mask, col].astype(str).str.strip()
        self.classes_ = np.array(sorted(y.unique()))
        self.feature_types_ = {col: "categorical" for col in X.columns}
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                mask = X[col].notna()
                X.loc[mask, col] = X.loc[mask, col].astype(str).str.strip()
        preds = [self._predict_row(row, self.tree_) for _, row in X.iterrows()]
        return np.array(preds)

    def print_tree(self):
        self._print_node(self.tree_)

    @staticmethod
    def _entropy(y: pd.Series) -> float:
        p = y.value_counts(normalize=True)
        return -(p * np.log2(p + 1e-15)).sum()

    def _information_gain_categorical(self, X_col: pd.Series, y: pd.Series):
        base_entropy = self._entropy(y)
        Xc = X_col.fillna("__MISSING__")
        values = Xc.unique()
        ent = 0.0
        split_map = {}
        for v in values:
            mask = (Xc == v)
            y_child = y[mask]
            ent += (len(y_child) / len(y)) * self._entropy(y_child)
            split_map[v] = mask
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


data = pd.read_csv("mushrooms.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

tree = ID3DecisionTree(max_depth=None, min_samples_split=2, min_gain=1e-9, random_state=0)
tree.fit(X, y)
tree.print_tree()

preds = tree.predict(X)
acc = (preds == y.values).mean()
print("Training accuracy:", acc)
print("Pretty tree:")
tree.pretty_print()
