# %% [code]
import time
import json
import joblib
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc


ord_categorical_features = [
    "sex",
    "tbp_lv_location",
    "tbp_tile_type",
    "tbp_lv_location_simple",
]

ohe_categorical_features = [
    "anatom_site_general",
    "attribution",
]

attribution_mapper = {
    "Memorial Sloan Kettering Cancer Center": "MSKCC",
    "ACEMID MIA": "ACEMIDMIA",
    "Department of Dermatology, Hospital Clínic de Barcelona": "DoD_HCB",
    "University Hospital of Basel": "UHB",
    "Frazer Institute, The University of Queensland, Dermatology Research Centre": "FI_TUQ-DRC",
    "Department of Dermatology, University of Athens, Andreas Syggros Hospital of Skin and Venereal Diseases, Alexander Stratigos, Konstantinos Liopyris": "DoD_UA",
    "ViDIR Group, Department of Dermatology, Medical University of Vienna": "ViDIR"
}


def compute_pauc(y_true, y_pred, min_tpr: float = 0.80) -> float:
    """
    2024 ISIC Challenge metric: pAUC

    Given a solution file and submission file, this function returns the
    partial area under the receiver operating characteristic (pAUC)
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        min_tpr: minimum true positive rate
        y_true: ground truth of 1s and 0s
        y_pred: predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    """

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(y_true - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * y_pred

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc



def boosting_norm_feature(df, value_col, group_cols, err=1e-5):
    stats = ["mean", "std"]
    tmp = df.groupby(group_cols)[value_col].agg(stats)
    tmp.columns = [f"{value_col}_{stat}" for stat in stats]
    tmp.reset_index(inplace=True)
    df = df.merge(tmp, on=group_cols, how="left")
    feature_name = f"{value_col}_patient_norm"
    df[feature_name] = ((df[value_col] - df[f"{value_col}_mean"]) /
                        (df[f"{value_col}_std"] + err))
    return df, feature_name


def boosting_feature_engineering(df):
    df["sex"] = df["sex"].fillna("missing_sex")
    df["anatom_site_general"] = df["anatom_site_general"].fillna("missing_anatom_site_general")
    df["tbp_tile_type"] = df["tbp_tile_type"].map({"3D: white": "white", "3D: XP": "XP"})
    df["attribution"] = df["attribution"].map(attribution_mapper)

    cols_to_norm = [
        "age_approx",
        "clin_size_long_diam_mm",
        "tbp_lv_A", "tbp_lv_Aext",
        "tbp_lv_B", "tbp_lv_Bext",
        "tbp_lv_C", "tbp_lv_Cext",
        "tbp_lv_H", "tbp_lv_Hext",
        "tbp_lv_L", "tbp_lv_Lext",
        "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence", "tbp_lv_norm_border",
        "tbp_lv_norm_color", "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max", "tbp_lv_stdL", "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle",
        "tbp_lv_x", "tbp_lv_y", "tbp_lv_z"
    ]
    numerical_features = cols_to_norm[:]
    for col in cols_to_norm:
        df, feature_name = boosting_norm_feature(df, col, ["patient_id"])
        numerical_features += [feature_name]

    df["num_images"] = df["patient_id"].map(df.groupby("patient_id")["isic_id"].count())
    numerical_features += ["num_images"]
    return df, numerical_features


class PAUC:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        y_true = target.astype(int)
        y_pred = approxes[0].astype(float)
        score = compute_pauc(y_true, y_pred, min_tpr=0.8)
        return score, 1.0


def pauc_80(y_train, y_pred):
    score_value = compute_pauc(y_train, y_pred, min_tpr=0.8)
    return score_value


def run(train_metadata, test_metadata, model_name, version, model_dir, folds_to_run):
    print(f"Predicting for {model_name}_{version}")
    start_time = time.time()
    with open(f"{model_dir}/{model_name}_{version}_run_metadata.json", "r") as f:
        run_metadata = json.load(f)
    pprint(run_metadata)

    with open(f"{model_dir}/{model_name}_{version}_encoder.joblib", "rb") as f:
        mixed_encoded_preprocessor = joblib.load(f)

    enc = mixed_encoded_preprocessor.fit(train_metadata)
    test = enc.transform(test_metadata)

    columns_for_model = len(test.columns)
    print(f"Total number of columns: {columns_for_model}")

    test_probs = 0
    for fold in folds_to_run:
        print(f"\nFold {fold}")
        model_filepath = f"{model_dir}/models/{model_name}_{version}_fold_{fold}.txt"
        with open(model_filepath, "rb") as f:
            estimator = joblib.load(f)
        test_probs_fold = estimator.predict_proba(test)[:, -1]
        if fold == 1:
            test_probs = test_probs_fold
        else:
            test_probs += test_probs_fold
    test_probs /= len(folds_to_run)
    oof_df = pd.DataFrame(
        {
            "isic_id": test_metadata["isic_id"],
            f"oof_{model_name}_{version}": test_probs.flatten(),
        }
    )
    runtime = time.time() - start_time
    print(f"Time taken: {runtime:.2f} s")
    print(f"Predictions generated for {model_name}_{version}")
    return oof_df, runtime
