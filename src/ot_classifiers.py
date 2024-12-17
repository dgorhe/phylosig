import numpy as np
import pandas as pd
from io import StringIO
from pdb import set_trace

from ete3 import Tree # type: ignore
from skbio import TreeNode
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Classifiers to test
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.sample_matrices import generate_ot_sample_cost_df
from src.sample_matrices import generate_unifrac_df
from src.sample_matrices import generate_metadata_df
from src.sample_matrices import load_metadata, load_otus



def run_classifiers(sample_cost_dfs, metadata_df, metadata_train):
    classifiers = {"random_forest": RandomForestClassifier(random_state=42),
                "decision_tree": DecisionTreeClassifier(random_state=42)}


    METHODS_SHAPE = len(sample_cost_dfs.keys())
    CLASSIFIERS_SHAPE = len(classifiers.keys())
    empty_arr = np.zeros((CLASSIFIERS_SHAPE, METHODS_SHAPE), dtype=float)
    accuracy_df = pd.DataFrame(data=empty_arr,
                                columns=list(sample_cost_dfs.keys()),
                                index=list(classifiers.keys()),
                                copy=True)

    f1_score_df = pd.DataFrame(data=empty_arr,
                                columns=list(sample_cost_dfs.keys()),
                                index=list(classifiers.keys()),
                                copy=True)


    def generate_input_matrix(cost_df, _type):
        if _type == "Metadata":
            enc = OrdinalEncoder(encoded_missing_value=10)
            # X = enc.fit_transform(subset_metadata_df)
            X = enc.fit_transform(metadata_train)
        else:
            X = cost_df.to_numpy()
        
        return X

    for name, cost_df in sample_cost_dfs.items():
        # labels = ibd_metadata_diagnosis.loc[cost_df.index].values.ravel()
        labels = metadata_df.loc[cost_df.index].values.ravel()
        X = generate_input_matrix(cost_df, _type=name)
        
        for cl_name, classifier in classifiers.items():
            le = LabelEncoder()
            y = le.fit_transform(labels)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(classifier, X, y, cv=cv)
            
            accuracy_df.loc[cl_name, name] = float(accuracy_score(y_true=y, y_pred=y_pred))
            f1_score_df.loc[cl_name, name] = float(f1_score(y_true=y, y_pred=y_pred, average='weighted'))
    
    set_trace()

def get_sample_cost_dfs():
    metadata = load_metadata(metadata_path="ihmp/ibd_metadata_new.csv")
    otus = load_otus(data_path="ihmp/ibd_data.csv", metadata=metadata)
    tree = Tree("data/gg_13_5_otus_99_annotated.tree", format=1, quoted_node_names=True)
    skbio_tree = TreeNode.read(StringIO(tree.write(format_root_node=True))) # type: ignore
    subtree = skbio_tree.shear(otus.index)

    levenshtein_cost_matrix = np.load("levenshtein_cost_matrix.npy")
    alignment_cost_matrix = np.load("alignment_cost_matrix.npy")

    return {"OT_Levenshtein": generate_ot_sample_cost_df(levenshtein_cost_matrix, otus=otus),
            "OT_Alignment": generate_ot_sample_cost_df(alignment_cost_matrix, otus=otus),
            "Metadata": generate_metadata_df(metadata=metadata),
            "Unweighted UniFrac": generate_unifrac_df(otus=otus, tree=subtree, _type="unweighted"),
            "Weighted UniFrac": generate_unifrac_df(otus=otus, tree=subtree, _type="weighted")}


if __name__ == "__main__":
    metadata_df = load_metadata(metadata_path="ihmp/ibd_metadata_new.csv").set_index("sample")
    metadata_train = generate_metadata_df(metadata_df)
    run_classifiers(get_sample_cost_dfs(), 
                    metadata_df=metadata_df, 
                    metadata_train=metadata_train)