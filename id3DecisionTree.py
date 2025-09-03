import numpy as np
import pandas as pd
from collections import Counter

class ID3DecisionTree:
    """
    ID3 Decision Tree (Trying to make it work with numeric features as well).
    - Supports categorical and numeric features.
    - Uses Information Gain (entropy) to choose splits.
    - Stopping criteria: max_depth, min_samples_split, min_gain.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-12, max_numeric_thresholds=50, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_numeric_thresholds = max_numeric_thresholds
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.tree_ = None
        self.feature_types_ = None  # "categorical" or "numeric"
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)
        X.index = y.index

        #Normalize categorical columns (coerce to string, strip whitespace; keep NaN as NaN)
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                mask = X[col].notna()
                X.loc[mask, col] = X.loc[mask, col].astype(str).str.strip()

        # Remember classes for consistency
        self.classes_ = np.array(sorted(y.unique()))
        # feature types (categorical vs numeric)
        self.feature_types_ = {
            col: self._infer_feature_type(X[col]) for col in X.columns
        }

        # Build the tree
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        preds = [self._predict_row(row, self.tree_) for _, row in X.iterrows()]
        return np.array(preds)

    def print_tree(self):
        self._print_node(self.tree_)

    # HELPER FUNCTIONS

    @staticmethod
    def _entropy(y: pd.Series) -> float:
        counts = y.value_counts(normalize=True)
        # Use logarithm base 2
        return -(counts * np.log2(counts + 1e-15)).sum()

    def _information_gain_categorical(self, X_col: pd.Series, y: pd.Series) -> (float, dict):
        """Split by each category value; return (gain, split_map)."""
        base_entropy = self._entropy(y)
        # Treat NaN as its own category to avoid dropping data
        X_col = X_col.fillna("__MISSING__")
        values = X_col.unique()

        # Weighted child entropy
        ent = 0.0
        split_map = {}
        for v in values:
            mask = (X_col == v)
            y_child = y[mask]
            ent += (len(y_child) / len(y)) * self._entropy(y_child)
            split_map[v] = mask

        gain = base_entropy - ent
        return gain, split_map

    def _information_gain_numeric(self, X_col: pd.Series, y: pd.Series) -> (float, dict, float):
        """
        Binary split on X <= threshold vs > threshold.
        We search candidate thresholds (quantiles or unique midpoints).
        Returns (gain, split_map, best_threshold).
        """
        # Drop rows where feature is NaN from threshold search
        notna_mask = X_col.notna()
        x = X_col[notna_mask].astype(float)
        y_clean = y[notna_mask]

        if len(x) < 2:
            return 0.0, {}, None

        base_entropy = self._entropy(y)

        # Candidate thresholds:
        # Use unique sorted values, sample up to max_numeric_thresholds midpoints
        x_sorted = x.sort_values()
        uniq = np.unique(x_sorted.values)
        if len(uniq) == 1:
            return 0.0, {}, None

        # Midpoints between consecutive unique values
        mids = (uniq[:-1] + uniq[1:]) / 2.0

        if len(mids) > self.max_numeric_thresholds:
            # Sample evenly spaced candidate thresholds
            idxs = np.linspace(0, len(mids) - 1, self.max_numeric_thresholds).astype(int)
            mids = mids[idxs]

        best_gain = -np.inf
        best_thr = None
        best_split = None

        for thr in mids:
            left_mask_local = (X_col <= thr) & X_col.notna()
            right_mask_local = (X_col > thr) & X_col.notna()

            if left_mask_local.sum() == 0 or right_mask_local.sum() == 0:
                continue

            y_left = y[left_mask_local]
            y_right = y[right_mask_local]

            ent = (len(y_left)/len(y_clean)) * self._entropy(y_left) + \
                  (len(y_right)/len(y_clean)) * self._entropy(y_right)
            gain = base_entropy - ent

            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_split = {"left": left_mask_local, "right": right_mask_local}

        if best_gain == -np.inf:
            return 0.0, {}, None

        # Add a third branch for NaN values: send them where majority of non-NaN goes (or to left by default)
        nan_mask = X_col.isna()
        if nan_mask.any():
            # Heuristic: attach NaNs to the larger child to reduce entropy
            left_n = best_split["left"].sum()
            right_n = best_split["right"].sum()
            if left_n >= right_n:
                best_split["left"] = best_split["left"] | nan_mask
            else:
                best_split["right"] = best_split["right"] | nan_mask

        return best_gain, best_split, float(best_thr)

    def _best_split(self, X: pd.DataFrame, y: pd.Series):
        """
        Returns:
            best_feature, split_type ('categorical'|'numeric'|None),
            split_info (dict with masks, and maybe 'threshold'), best_gain
        """
        best_feature = None
        best_type = None
        best_info = None
        best_gain = -np.inf

        for col in X.columns:
            ftype = self.feature_types_[col]
            if ftype == "categorical":
                gain, split_map = self._information_gain_categorical(X[col], y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_type = "categorical"
                    best_info = {"children": split_map}
            else:  # numeric
                gain, split_map, thr = self._information_gain_numeric(X[col], y)
                if gain > best_gain and thr is not None:
                    best_gain = gain
                    best_feature = col
                    best_type = "numeric"
                    best_info = {"left": split_map.get("left"),
                                 "right": split_map.get("right"),
                                 "threshold": thr}

        if best_gain < self.min_gain:
            return None, None, None, 0.0
        return best_feature, best_type, best_info, best_gain

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int):
        # Leaf conditions
        if len(y.unique()) == 1:
            return {"type": "leaf", "class": y.iloc[0]}

        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or \
           (X.shape[1] == 0):
            return {"type": "leaf", "class": self._majority_class(y)}

        # Find best split
        best_feature, best_type, split_info, gain = self._best_split(X, y)
        if best_feature is None:
            return {"type": "leaf", "class": self._majority_class(y)}

        node = {
            "type": "node",
            "feature": best_feature,
            "feature_type": best_type,
            "gain": gain,
            "majority_class": self._majority_class(y) # For falling back in case of missing features
        }

        if best_type == "categorical":
            node["children"] = {}
            for val, mask in split_info["children"].items():
                X_child = X[mask].drop(columns=[best_feature])
                y_child = y[mask]
                node["children"][val] = self._build_tree(X_child, y_child, depth + 1)

        else:  # numeric
            thr = split_info["threshold"]
            node["threshold"] = thr
            # Build left and right
            left_mask = split_info["left"]
            right_mask = split_info["right"]

            X_left = X[left_mask].copy()
            X_right = X[right_mask].copy()
            y_left = y[left_mask]
            y_right = y[right_mask]

            node["left"] = self._build_tree(X_left, y_left, depth + 1)
            node["right"] = self._build_tree(X_right, y_right, depth + 1)

        return node

    def _predict_row(self, row: pd.Series, node):
        while node["type"] != "leaf":
            feat = node["feature"]
            ftype = node["feature_type"]

            # If feature missing at predict-time, use majority fallback
            if feat not in row.index:
                return node.get("majority_class", None)

            val = row[feat]

            if ftype == "categorical":
                key = val if pd.notna(val) else "__MISSING__"
                child = node["children"].get(key)
                if child is None:
                    # unseen category -> fallback
                    return node.get("majority_class", None)
                node = child
            else:  # numeric
                thr = node["threshold"]
                # NaN -> fallback to majority
                if pd.isna(val):
                    return node.get("majority_class", None)
                node = node["left"] if float(val) <= thr else node["right"]

        return node["class"]

    @staticmethod
    def _infer_feature_type(s: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        return "categorical"

    @staticmethod
    def _majority_class(y: pd.Series):
        # Break ties deterministically by class name
        counts = y.value_counts()
        max_count = counts.max()
        candidates = sorted(counts[counts == max_count].index.tolist())
        return candidates[0]

    # Pretty print the tree

    def _print_node(self, node, indent=""):
        if node["type"] == "leaf":
            print(f"{indent}Leaf: predict = {node['class']}")
            return
        feat = node["feature"]
        if node["feature_type"] == "categorical":
            print(f"{indent}if {feat} in ... (gain={node['gain']:.5f})")
            for val, child in node["children"].items():
                print(f"{indent}  [{feat} == {val!r}]")
                self._print_node(child, indent + "    ")
        else:
            thr = node["threshold"]
            print(f"{indent}if {feat} <= {thr:.6g} (gain={node['gain']:.5f})")
            print(f"{indent}  [True]")
            self._print_node(node["left"], indent + "    ")
            print(f"{indent}  [False]")
            self._print_node(node["right"], indent + "    ")

# Example usage 
df = pd.DataFrame({
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": [85,80,83,70,68,65,64,72,69,75,75,72,81,71],
    "Humidity": [85,90,86,96,80,70,65,95,70,80,70,90,75,80],
    "Windy": ["False","True","False","False","False","True","True","False","False","False","True","True","False","True"],
})
y = pd.Series(["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"], name="Play")

# Fit
tree = ID3DecisionTree(max_depth=None, min_samples_split=2, min_gain=1e-9, random_state=0)
tree.fit(df, y)

# Inspect tree
tree.print_tree()

# Predict
preds = tree.predict(df)
acc = (preds == y.values).mean()
print("Training accuracy:", acc)
