#functions.py - Contains reusable function modules

#==================================================
#IMPORTS
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import mrmr
from mrmr.pandas import mrmr_classif, mrmr_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, ParameterSampler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
from scipy.stats import pointbiserialr
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import io
import contextlib
#==================================================

def eda(development_df, univariate_column,target):
    

    print("1. Development Dataset summmary table")

    # Dataset summary values
    n_rows, n_cols = development_df.shape
    age = development_df["age"].dropna()
    sex_counts = development_df["sex"].value_counts()
    sex_percent = development_df["sex"].value_counts(normalize=True) * 100

    # Build summary table
    summary_table = pd.DataFrame({
        "Characteristic": [
            "Number of samples",
            "Number of variables",
            "Age, mean ± SD (years)",
            "Age range (years)",
            "Male, n (%)",
            "Female, n (%)"
        ],
        "Value": [
            f"{n_rows}",
            f"{n_cols}",
            f"{age.mean():.1f} ± {age.std():.1f}",
            f"{age.min():.0f}–{age.max():.0f}",
            f"{sex_counts.get('M', 0)} ({sex_percent.get('M', 0):.1f}%)",
            f"{sex_counts.get('F', 0)} ({sex_percent.get('F', 0):.1f}%)"
        ]
    })

    # Style and display table
    styled_table = (
        summary_table.style
        .hide(axis="index")
        .hide(axis="columns")
        .set_caption("Table 1. Summary characteristics of the development dataset")
        .set_properties(subset=["Characteristic"], **{
            "font-weight": "bold",
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_properties(subset=["Value"], **{
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "12pt"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("padding-bottom", "8px")
                ]
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "65%")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #cccccc")
                ]
            }
        ])
    )

    display(styled_table)

    print("2. Sex Class imbalance visualization")

    # Sex Class imbalance plots
   
    # Count categories
    counts = (
        development_df["sex"]
        .value_counts()
        .reindex(["M", "F"])
    )
    counts = counts.rename(index={"M": "Male", "F": "Female"})

    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

    # Bar plot
    counts.plot(kind="bar", ax=ax)

    # Titles and labels
    ax.set_title("Sex Class Imbalance in the Development Dataframe", fontsize=12)
    ax.set_xlabel("Sex", fontsize=11)
    ax.set_ylabel("Number of samples (n)", fontsize=11)

    # Improve readability
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Remove top/right spines for cleaner style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: add counts above bars
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    print("3. Distribution of Age in development dataframe")

    # Age Distribution Violin plot

    if pd.api.types.is_numeric_dtype(development_df[univariate_column]):
        values = development_df[univariate_column].dropna()

        fig, axes = plt.subplots(
            1, 2, figsize=(5.6, 3.0), dpi=300,
            gridspec_kw={"width_ratios": [0.9, 1.4]}
        )

        # Violin
        ax = axes[0]
        parts = ax.violinplot(
            values,
            positions=[1],
            widths=0.5,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
            pc.set_linewidth(1)

        parts["cmedians"].set_linewidth(1.2)

        np.random.seed(42)
        x = np.random.normal(1, 0.025, size=len(values))
        ax.scatter(x, values, s=6, alpha=0.2, linewidths=0)

        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("Age (years)", fontsize=10)

        ax.tick_params(axis="y", labelsize=9, width=1)
        ax.tick_params(axis="x", width=1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

        # Histogram
        ax = axes[1]
        ax.hist(values, bins=15, edgecolor="black", linewidth=0.8)

        ax.set_xlabel("Age (years)", fontsize=10)
        ax.set_ylabel("Number of samples (n)", fontsize=10)

        ax.tick_params(axis="both", labelsize=9, width=1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)

        fig.suptitle("Distribution of Age in the development dataset", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.show()



    print("4.Feature Relationships Visualization")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=development_df, x="sex", y="age", hue="ethnicity")
    plt.title("Age by sex and ethnicity")
    plt.xlabel("Sex")
    plt.ylabel("Age (years)")
    plt.tight_layout()
    plt.show()

    #Correlation matrix
    print("Correlation Matrix")

    corr_matrix = development_df.corr(numeric_only=True)
    corr_with_target = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)

    # Keep only top N correlations
    top_n = 20
    corr_table = pd.DataFrame({
        "Variable": corr_with_target.index[:top_n],
        f"Correlation with {target}": corr_with_target.values[:top_n]
    })

    styled_corr_table = (
        corr_table.style
        .hide(axis="index")
        .set_caption(f"Top {top_n} numeric features correlated with {target}")
        .set_properties(subset=["Variable"], **{
            "font-weight": "bold",
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_properties(subset=[f"Correlation with {target}"], **{
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .format({f"Correlation with {target}": "{:.3f}"})
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "12pt"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("padding-bottom", "8px")
                ]
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "70%")
                ]
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "bold"),
                    ("border-bottom", "1px solid #cccccc"),
                    ("padding", "6px 12px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #cccccc")
                ]
            }
        ])
    )

    display(styled_corr_table)

    
    #Correlation plot with best features vs target
    print ("Correlation Plot of Age and the most correlated features")
    top_10_features = corr_with_target.abs().sort_values(ascending=False).head(10).index.tolist()    
    
    for feature in top_10_features:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=development_df, x=feature, y= target, scatter_kws={"alpha": 0.5})
        plt.title(f"Age vs {feature} (r={corr_with_target[feature]:.3f})")
        plt.xlabel(f"{feature} (β-value)")
        plt.ylabel(f"{target} (years)")
        plt.tight_layout()
        plt.show()

    #Distribution of 10 most correlated features with the target.
    axes = development_df[top_10_features].hist(figsize=(24, 16), bins=50)
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_xlabel():
                ax.set_xlabel(f"{ax.get_xlabel()} (β-value)", fontsize=9)
            ax.set_ylabel("Number of samples (n)", fontsize=9)
    plt.suptitle("Distribution of top 10 features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



#Visualize the heteroscedasticity differences between beta- and M- value representation of methylation data.

def plot_heteroscedasticity_beta_vs_mvalues(
    development_df,
    heteroscedasticity_prefix,
    beta_clip=1e-6
):

    # Select methylation columns
    heteroscedasticity_cols = [
        col for col in development_df.columns
        if col.startswith(heteroscedasticity_prefix)
    ]

    heteroscedasticity_df = development_df[heteroscedasticity_cols].copy()

    # -------------------------
    # Beta-value mean/variance
    # -------------------------
    beta_mean = heteroscedasticity_df.mean()
    beta_variance = heteroscedasticity_df.var()

    # -------------------------
    # M-value transformation
    # M = log2(beta / (1 - beta))
    # -------------------------
    beta_clipped = heteroscedasticity_df.clip(lower=beta_clip, upper=1 - beta_clip)
    mvalue_df = np.log2(beta_clipped / (1 - beta_clipped))

    m_mean = mvalue_df.mean()
    m_variance = mvalue_df.var()

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Beta values panel
    sns.scatterplot(
        x=beta_mean,
        y=beta_variance,
        alpha=0.5,
        ax=axes[0]
    )
    sns.regplot(
        x=beta_mean,
        y=beta_variance,
        scatter=False,
        order=2,
        ax=axes[0],
        line_kws={"linewidth": 2}
    )
    axes[0].set_title("Beta values", fontsize=11)
    axes[0].set_xlabel("Mean β-value (0–1)", fontsize=10)
    axes[0].set_ylabel("Variance (β-value²)", fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].tick_params(axis="both", labelsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # M-values panel
    sns.scatterplot(
        x=m_mean,
        y=m_variance,
        alpha=0.5,
        ax=axes[1]
    )
    sns.regplot(
        x=m_mean,
        y=m_variance,
        scatter=False,
        order=2,
        ax=axes[1],
        line_kws={"linewidth": 2}
    )
    axes[1].set_title("M-values", fontsize=11)
    axes[1].set_xlabel("Mean M-value (log₂ scale)", fontsize=10)
    axes[1].set_ylabel("Variance (M-value²)", fontsize=10)
    axes[1].tick_params(axis="both", labelsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


#Feature Matrices generation

def feature_select(dataframe, feature_cols, label_col=None):
    X = dataframe[feature_cols].copy()

    if label_col is not None:
        y = dataframe[label_col].copy()
        return X, y

    return X


def build_feature_matrices(dataframe):
    metadata_cols = ["sex", "ethnicity"]
    methylation_cols = [col for col in dataframe.columns if col.startswith("cg")]
    sex_and_methylation_cols = methylation_cols + ["sex"]
    ethnicity_and_methylation_cols = methylation_cols + ["ethnicity"]
    all_cols = methylation_cols + metadata_cols

    feature_matrices = {
        "1. Metadata Only": feature_select(dataframe, metadata_cols),
        "2. Methylation Only": feature_select(dataframe, methylation_cols),
        "3. Sex + Methylation": feature_select(dataframe, sex_and_methylation_cols),
        "4. Ethnicity + Methylation": feature_select(dataframe, ethnicity_and_methylation_cols),
        "5. All Features": feature_select(dataframe, all_cols)
    }

    return feature_matrices



def stratified_split(X,y,seed,training_size,strata_quantity, classification = False):

    #Bin the continous feature using equal frequency binning 
    if classification == False:
        y_stratify = pd.qcut(y, q = strata_quantity, duplicates = "drop")
    else:
        y_stratify = y


    #Perform a stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed, train_size = training_size,
                                    stratify = y_stratify)

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True)
    )



def transform_beta_to_m(X, epsilon=1e-6):
    #Clip X to avoid absolute 1 and 0, to not break M value logit function
    X_clipped = np.clip(X, epsilon, 1-epsilon)
    
    #Calculate M_values
    return np.log2(X_clipped / (1-X_clipped))


def preprocessing_df(X_train,impute_strategy_num, impute_strategy_cat): 

    #Find numerical and categorical columns.
    numerical_columns = X_train.select_dtypes(include=["number"]).columns
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns

    #Create a simple column transformer for imputation-scaling-encoding

    numercal_preprocessor = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy=impute_strategy_num)),
        ("m_transform", FunctionTransformer(transform_beta_to_m)),
        ("scaler",StandardScaler())
    ])
    
    categorical_preprocessor = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy=impute_strategy_cat)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))        
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numercal_preprocessor, numerical_columns),
        ("cat", categorical_preprocessor, categorical_columns)
    ])

    preprocessor.set_output(transform="pandas")
    return preprocessor

def preprocessing(X_train,impute_strategy_num, impute_strategy_cat): 

    #Find numerical and categorical columns.
    numerical_columns = X_train.select_dtypes(include=["number"]).columns
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns

    #Create a simple column transformer for imputation-scaling-encoding

    numercal_preprocessor = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy=impute_strategy_num)),
        ("m_transform", FunctionTransformer(transform_beta_to_m)),
        ("scaler",StandardScaler())
    ])
    
    categorical_preprocessor = Pipeline(steps = [
        ("imputer",SimpleImputer(strategy=impute_strategy_cat)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))        
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numercal_preprocessor, numerical_columns),
        ("cat", categorical_preprocessor, categorical_columns)
    ])

    return preprocessor

import numpy as np
import pandas as pd
from scipy import stats

def bootstrap_evaluation(confidence, prediction, y_test, resamples, seed):
    
    prediction = np.asarray(prediction)
    y_test = np.asarray(y_test)    
    
    errors = prediction - y_test
    squared_errors = errors ** 2
    abs_errors = np.abs(errors)

    def rmse(squared_errors, axis=0):
        return np.sqrt(np.mean(squared_errors, axis=axis))
    
    def mae(abs_errors, axis=0):
        return np.mean(abs_errors, axis=axis)

    def r_squared(prediction, y_test, axis=0):
        ss_res = np.sum((y_test - prediction) ** 2, axis=axis)
        ss_tot = np.sum((y_test - y_test.mean(axis=axis, keepdims=True)) ** 2, axis=axis)
        return 1 - ss_res / ss_tot
    
    def pearson_r(prediction, y_test, axis=0):
        mean_prediction = prediction.mean(axis=axis, keepdims=True)
        mean_y_test = y_test.mean(axis=axis, keepdims=True)
        numerator = np.sum((prediction - mean_prediction) * (y_test - mean_y_test), axis=axis)
        denominator = np.sqrt(
            np.sum((prediction - mean_prediction) ** 2, axis=axis) * 
            np.sum((y_test - mean_y_test) ** 2, axis=axis)
        )
        return numerator / (denominator + 1e-12)

    # Set common bootstrap arguments for all metrics 
    common = dict(
        n_resamples=resamples,
        confidence_level=confidence,
        random_state=seed,
        axis=0
    )

    # Run bootstraps
    boot_rmse = stats.bootstrap((squared_errors,), statistic=rmse, **common)
    boot_mae  = stats.bootstrap((abs_errors,), statistic=mae, **common)
    boot_r2   = stats.bootstrap((prediction, y_test), statistic=r_squared, **common, paired=True)
    boot_pr   = stats.bootstrap((prediction, y_test), statistic=pearson_r, **common, paired=True)

    # Compile results to a df
    results = {
        "Metric": ["RMSE", "MAE", "R_squared", "Pearson_r"],
        "Estimate": [
            rmse(squared_errors),
            mae(abs_errors),
            r_squared(prediction, y_test),
            pearson_r(prediction, y_test)
        ],
        "SD": [
            np.std(boot_rmse.bootstrap_distribution, ddof=1),
            np.std(boot_mae.bootstrap_distribution, ddof=1),
            np.std(boot_r2.bootstrap_distribution, ddof=1),
            np.std(boot_pr.bootstrap_distribution, ddof=1)
        ],
        "CI_Low": [
            boot_rmse.confidence_interval.low,
            boot_mae.confidence_interval.low,
            boot_r2.confidence_interval.low,
            boot_pr.confidence_interval.low
        ],
        "CI_High": [
            boot_rmse.confidence_interval.high,
            boot_mae.confidence_interval.high,
            boot_r2.confidence_interval.high,
            boot_pr.confidence_interval.high
        ]
    }

    distributions = {
        "RMSE":      boot_rmse.bootstrap_distribution,
        "MAE":       boot_mae.bootstrap_distribution,
        "R_squared": boot_r2.bootstrap_distribution,
        "Pearson_r": boot_pr.bootstrap_distribution,
    }

    return pd.DataFrame(results).set_index("Metric"), distributions


def stability_selection(X_train, y_train, resamples, top_k, subsample_fraction, seed):

    # Count examples and features on the full matrix
    n_examples, n_features = X_train.shape
    # Define subsample size
    subsample_size = int(subsample_fraction * n_examples)
    # Initialize array with frequency of each feature
    selection_counts = np.zeros(n_features, dtype=int)
    # Set seed for random number generator
    random_number = np.random.default_rng(seed)

    for _ in range(resamples):
        subsamples_id = random_number.choice(n_examples, size=subsample_size, replace=False)
        # Define X and y in subsamples
        X_subsample = X_train.iloc[subsamples_id]
        y_subsample = y_train.iloc[subsamples_id]
        # Initialize scores array with zeros
        corr_scores = np.zeros(n_features)

        for feature in range(n_features):
            corr, pval = spearmanr(X_subsample.iloc[:, feature], y_subsample)

            if np.isnan(corr):
                corr = 0.0

            corr_scores[feature] = abs(corr)

        top_features = np.argsort(corr_scores)[-top_k:]
        selection_counts[top_features] += 1

    # Keep stable features only
    stable_features = np.where(selection_counts > (resamples / 2))[0]
    print("Number of stable features:", len(stable_features))

    #All features frequency histogram
    plt.hist(selection_counts, bins=range(resamples + 2), edgecolor='black')
    plt.xlabel("Selection frequency (number of bootstrap iterations)")
    plt.ylabel("Number of features (n)")
    plt.title("Selection-frequency distribution")
    plt.show()

    # Selection frequency histogram
    plt.hist(selection_counts[stable_features], bins=range(resamples + 2), edgecolor='black')
    plt.xlabel("Selection frequency (number of bootstrap iterations)")
    plt.ylabel("Number of stable features (n)")
    plt.title("Selection-frequency distribution (stable features only)")
    plt.show()

    # Diagnostics
    corr_scores = np.array([
        0.0 if np.isnan(spearmanr(X_train.iloc[:, f], y_train)[0]) 
        else abs(spearmanr(X_train.iloc[:, f], y_train)[0])
        for f in range(n_features)
    ])
    
    plt.hist(corr_scores, bins=50, edgecolor='black')
    plt.xlabel("Absolute Spearman correlation (unitless, 0–1)")
    plt.ylabel("Number of features (n)")
    plt.title("Full-data correlation distribution")
    plt.show()

    print("Top 10 correlations:     ", np.sort(corr_scores)[::-1][:10])
    print("Correlations at rank 11+:", np.sort(corr_scores)[::-1][10:20])
    print("Median correlation:      ", np.median(corr_scores))

    print("y_train unique values:", y_train.nunique())
    print("y_train value counts:\n", y_train.value_counts())
    print("y_train dtype:", y_train.dtype)
    print("X_train shape:", X_train.shape)
    print("Any NaN in X_train:", X_train.isna().any().any())
    print("Any NaN in y_train:", y_train.isna().any())

    return stable_features, selection_counts

from sklearn.model_selection import cross_val_score

def mrmr_k_tuning(X_train, y_train, X_test, y_test, K_values):
    results = []
    for K in K_values:
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        selected = mrmr_regression(X=X_train, y=y_train, K=K, cat_features=cat_cols)
        ols_pipeline = Pipeline(steps=[
            ('model', LinearRegression())
        ])

        cv_scores = cross_val_score(
            ols_pipeline, X_train[selected], y_train,
            scoring="neg_root_mean_squared_error", cv=5
        )
        results.append({
            "K": K,
            "cv_rmse": -cv_scores.mean(),
            "cv_rmse_sd": cv_scores.std(),
            "features": selected
        })

    results_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    best_K = results_df.loc[0, "K"]
    best_features = results_df.loc[0, "features"]

    # Final evaluation on test set
    final_pipeline = Pipeline(steps=[
        ('model', LinearRegression())
    ])

    final_pipeline.fit(X_train[best_features], y_train)
    test_rmse = np.sqrt(mean_squared_error(y_test, final_pipeline.predict(X_test[best_features])))

    #Plot 
    plot_df = pd.DataFrame(results).sort_values("K").reset_index(drop=True)

    BLUE   = "#2C6FAC"
    GREY   = "#A0A0A0"
    RED    = "#C0392B"
    LIGHT  = "#D6E4F0"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Error band
    ax.fill_between(
        plot_df["K"],
        plot_df["cv_rmse"] - plot_df["cv_rmse_sd"],
        plot_df["cv_rmse"] + plot_df["cv_rmse_sd"],
        alpha=0.18, color=BLUE, label="±1 SD"
    )

    # CV RMSE line
    ax.plot(plot_df["K"], plot_df["cv_rmse"],
            color=BLUE, linewidth=2, marker="o",
            markersize=7, markerfacecolor="white",
            markeredgewidth=2, label="CV RMSE (mean)")

    # Highlight best K
    best_row = plot_df[plot_df["K"] == best_K].iloc[0]
    ax.axvline(best_K, color=RED, linestyle="--", linewidth=1.4, alpha=0.7)
    ax.scatter(best_K, best_row["cv_rmse"],
               color=RED, zorder=5, s=90, label=f"Best K = {best_K}")

    # Test RMSE reference line
    ax.axhline(test_rmse, color=GREY, linestyle=":", linewidth=1.4,
               label=f"Test RMSE = {test_rmse:.4f}")

    # Annotations
    ax.annotate(
        f"K={best_K}\nRMSE={best_row['cv_rmse']:.4f}",
        xy=(best_K, best_row["cv_rmse"]),
        xytext=(12, 12), textcoords="offset points",
        fontsize=8.5, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2)
    )

    # Axes formatting
    ax.set_xlabel("Number of Features (K)", fontsize=11, labelpad=8)
    ax.set_ylabel("RMSE (years)", fontsize=11, labelpad=8)
    ax.set_title("mRMR Feature Selection — CV RMSE by K", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xticks(plot_df["K"])
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, color=GREY)

    legend = ax.legend(fontsize=9, frameon=True, framealpha=0.9,
                       edgecolor="#CCCCCC", loc="upper right")

    plt.tight_layout()
    plt.savefig("mrmr_k_tuning.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.show()

    return results_df, best_K, best_features, test_rmse

def mrmr_selection(X_train, y_train, K, classification=False):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if classification:
        selected_features = mrmr_classif(X=X_train, y=y_train, K=K, cat_features=cat_cols)
    else:
        selected_features = mrmr_regression(X=X_train, y=y_train, K=K, cat_features=cat_cols)
    print(selected_features)
    return selected_features
    

def final_tune_model_cv_mrmr(
    X,
    y,
    model,
    param_distributions,
    num_strategy="median",
    cat_strategy="most_frequent",
    k=30,
    trials=40,
    cv_num=5,
    seed=42
):
    sampled_params = list(
        ParameterSampler(
            param_distributions=param_distributions,
            n_iter=trials,
            random_state=seed
        )
    )

    cv_splitter = KFold(n_splits=cv_num, shuffle=True, random_state=seed)

    all_trial_results = []

    for i, params in enumerate(sampled_params, start=1):
        fold_rmses = []

        for train_idx, valid_idx in cv_splitter.split(X, y):
            X_train_fold = X.iloc[train_idx].copy()
            X_valid_fold = X.iloc[valid_idx].copy()
            y_train_fold = y.iloc[train_idx].copy()
            y_valid_fold = y.iloc[valid_idx].copy()

            # Preprocess on fold-train only
            preprocessor = preprocessing_df(X_train_fold, num_strategy, cat_strategy)
            X_train_fold_processed = preprocessor.fit_transform(X_train_fold)
            X_valid_fold_processed = preprocessor.transform(X_valid_fold)

            # mRMR on fold-train only
            selected_topk = mrmr_selection_quiet(X_train_fold_processed, y_train_fold, K=k)

            X_train_fold_topk = X_train_fold_processed[selected_topk]
            X_valid_fold_topk = X_valid_fold_processed[selected_topk]

            # Fit model
            candidate_model = clone(model)
            candidate_model.set_params(**params)
            candidate_model.fit(X_train_fold_topk, y_train_fold)

            y_pred = candidate_model.predict(X_valid_fold_topk)
            rmse = root_mean_squared_error(y_valid_fold, y_pred)
            fold_rmses.append(rmse)

        all_trial_results.append({
            "trial": i,
            "params": params,
            "mean_cv_rmse": np.mean(fold_rmses),
            "sd_cv_rmse": np.std(fold_rmses)
        })

    results_df = pd.DataFrame(all_trial_results).sort_values("mean_cv_rmse").reset_index(drop=True)
    best_params = results_df.loc[0, "params"]

    # Refit on full development set
    preprocessor_final = preprocessing_df(X, num_strategy, cat_strategy)
    X_processed = preprocessor_final.fit_transform(X)

    selected_topk_final = mrmr_selection_quiet(X_processed, y, K=k)
    X_topk_final = X_processed[selected_topk_final]

    best_model = clone(model)
    best_model.set_params(**best_params)
    best_model.fit(X_topk_final, y)

    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_rmse": results_df.loc[0, "mean_cv_rmse"],
        "cv_results": results_df,
        "preprocessor": preprocessor_final,
        "selected_features": selected_topk_final
    }

def tune_hyperparameters_cv(
    X,
    y,
    model,
    param_distributions,
    num_strategy="median",
    cat_strategy="most_frequent",
    trials = 40,
    cv_num = 5,
    score = "neg_root_mean_squared_error",
    seed = 42,
    n_jobs=-1
):
    preprocessor = preprocessing(X, num_strategy, cat_strategy)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter= trials,
        cv= cv_num,
        scoring= score,
        random_state=seed,
        n_jobs= n_jobs
    )

    search.fit(X, y)
    return search


def optuna_tune_model(model_name, pipeline, X_train, y_train, n_trials, cv=5, seed=42):
    def objective(trial):
        #Define the ML model name, to search for correct hyperparams
        model = pipeline.named_steps[model_name]
        model_type = type(model).__name__

        #Assign correct hyperparameter set to each model:

        if model_type == "ElasticNet":
            params = {
                f"{model_name}__alpha": trial.suggest_float(
                    f"{model_name}__alpha", 0.001, 10, log=True
                ),
                f"{model_name}__l1_ratio": trial.suggest_float(
                    f"{model_name}__l1_ratio", 0.1, 1.0
                )
            }

        elif model_type == "SVR":
            params = {
                f"{model_name}__C": trial.suggest_float(
                    f"{model_name}__C", 0.1, 500, log=True
                ),
                f"{model_name}__epsilon": trial.suggest_categorical(
                    f"{model_name}__epsilon", [0.01, 0.1, 0.5, 1.0]
                ),
                f"{model_name}__kernel": trial.suggest_categorical(
                    f"{model_name}__kernel", ["rbf", "linear"]
                )
            }

        elif model_type == "BayesianRidge":
            params = {
                f"{model_name}__alpha_1": trial.suggest_float(
                    f"{model_name}__alpha_1", 1e-7, 1e-3, log=True
                ),
                f"{model_name}__alpha_2": trial.suggest_float(
                    f"{model_name}__alpha_2", 1e-7, 1e-3, log=True
                ),
                f"{model_name}__lambda_1": trial.suggest_float(
                    f"{model_name}__lambda_1", 1e-7, 1e-3, log=True
                ),
                f"{model_name}__lambda_2": trial.suggest_float(
                    f"{model_name}__lambda_2", 1e-7, 1e-3, log=True
                )
            }         

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Supported models are ElasticNet, SVR, and BayesianRidge."
            )

        candidate_pipeline = clone(pipeline)
        candidate_pipeline.set_params(**params)

        #Define the scoring metric
        scores = cross_val_score(candidate_pipeline,
                                X_train,
                                y_train,
                                cv = cv,
                                scoring= "neg_root_mean_squared_error",
                                n_jobs=-1)        


        mean_rmse = -scores.mean()
        return mean_rmse
    
    #Create a n Optuna study
    study = optuna.create_study(
        direction = "minimize",
        sampler = TPESampler(seed = seed)
    )

    study.optimize(objective, n_trials = n_trials)

    best_params_clean = {
        k.replace(f"{model_name}__", ""): v
        for k, v in study.best_params.items()
    }
    best_params_clean["CV RMSE (best trial)"] = round(study.best_value, 4)

    params_df = pd.DataFrame(
        best_params_clean.items(),
        columns=["Hyperparameter", "Value"]
    )

    styled = (
        params_df.style
        .hide(axis="index")
        .set_caption(f"Best hyperparameters — {type(pipeline.named_steps[model_name]).__name__}")
        .set_properties(subset=["Hyperparameter"], **{
            "font-weight": "bold",
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_properties(subset=["Value"], **{
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_table_styles([
            {"selector": "caption", "props": [
                ("caption-side", "top"), ("font-size", "12pt"),
                ("font-weight", "bold"), ("text-align", "left"),
                ("padding-bottom", "8px")
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"), ("width", "55%")
            ]},
            {"selector": "td", "props": [
                ("border-bottom", "1px solid #cccccc")
            ]}
        ])
    )
    display(styled)

    best_pipeline = clone(pipeline)
    best_pipeline.set_params(**study.best_params)
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, study


def optuna_tune_model_mrmr(
    model,
    X_train,
    y_train,
    n_trials,
    k,
    num_strategy="median",
    cat_strategy="most_frequent",
    cv=5,
    seed=42
):
    def objective(trial):
        model_type = type(model).__name__

        if model_type == "ElasticNet":
            params = {
                "alpha": trial.suggest_float("alpha", 0.001, 10, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 1.0)
            }

        elif model_type == "SVR":
            params = {
                "C": trial.suggest_float("C", 0.1, 500, log=True),
                "epsilon": trial.suggest_categorical("epsilon", [0.01, 0.1, 0.5, 1.0]),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"])
            }

        elif model_type == "BayesianRidge":
            params = {
                "alpha_1": trial.suggest_float("alpha_1", 1e-7, 1e-3, log=True),
                "alpha_2": trial.suggest_float("alpha_2", 1e-7, 1e-3, log=True),
                "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-3, log=True),
                "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-3, log=True)
            }

        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Supported models are ElasticNet, SVR, and BayesianRidge."
            )

        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
        fold_rmses = []

        for train_idx, valid_idx in cv_splitter.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx].copy()
            X_fold_valid = X_train.iloc[valid_idx].copy()
            y_fold_train = y_train.iloc[train_idx].copy()
            y_fold_valid = y_train.iloc[valid_idx].copy()

            preprocessor = preprocessing_df(
                X_fold_train,
                impute_strategy_num=num_strategy,
                impute_strategy_cat=cat_strategy
            )

            X_fold_train_processed = preprocessor.fit_transform(X_fold_train)
            X_fold_valid_processed = preprocessor.transform(X_fold_valid)

            selected_topk = mrmr_selection_quiet(
                X_train=X_fold_train_processed,
                y_train=y_fold_train,
                K=k,
                classification=False
            )

            X_fold_train_topk = X_fold_train_processed[selected_topk]
            X_fold_valid_topk = X_fold_valid_processed[selected_topk]

            candidate_model = clone(model)
            candidate_model.set_params(**params)
            candidate_model.fit(X_fold_train_topk, y_fold_train)

            y_pred = candidate_model.predict(X_fold_valid_topk)
            rmse = root_mean_squared_error(y_fold_valid, y_pred)
            fold_rmses.append(rmse)

        return np.mean(fold_rmses)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed)
    )

    study.optimize(objective, n_trials=n_trials)

    best_params_clean = dict(study.best_params)
    best_params_clean["CV RMSE (best trial)"] = round(study.best_value, 4)

    params_df = pd.DataFrame(
        best_params_clean.items(),
        columns=["Hyperparameter", "Value"]
    )

    styled = (
        params_df.style
        .hide(axis="index")
        .set_caption(f"Best hyperparameters — {type(model).__name__}")
        .set_properties(subset=["Hyperparameter"], **{
            "font-weight": "bold",
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_properties(subset=["Value"], **{
            "text-align": "left",
            "padding": "6px 12px",
            "font-size": "11pt"
        })
        .set_table_styles([
            {"selector": "caption", "props": [
                ("caption-side", "top"), ("font-size", "12pt"),
                ("font-weight", "bold"), ("text-align", "left"),
                ("padding-bottom", "8px")
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"), ("width", "55%")
            ]},
            {"selector": "td", "props": [
                ("border-bottom", "1px solid #cccccc")
            ]}
        ])
    )
    display(styled)

    preprocessor_final = preprocessing_df(
        X_train,
        impute_strategy_num=num_strategy,
        impute_strategy_cat=cat_strategy
    )

    X_train_processed = preprocessor_final.fit_transform(X_train)

    selected_topk_final = mrmr_selection_quiet(
        X_train=X_train_processed,
        y_train=y_train,
        K=k,
        classification=False
    )

    X_train_topk_final = X_train_processed[selected_topk_final]

    best_model = clone(model)
    best_model.set_params(**study.best_params)
    best_model.fit(X_train_topk_final, y_train)

    return {
        "model": best_model,
        "preprocessor": preprocessor_final,
        "selected_features": selected_topk_final,
        "study": study,
        "best_params": study.best_params,
        "best_rmse": study.best_value
    }


def plot_optuna_history(study, title="Optuna optimisation history"):
    completed_trials = [trial for trial in study.trials if trial.value is not None]

    if not completed_trials:
        print("No completed trials with values found.")
        return

    trial_numbers = [trial.number for trial in completed_trials]
    trial_values = [trial.value for trial in completed_trials]

    best_values = np.minimum.accumulate(trial_values)

    plt.figure(figsize=(8, 5))
    plt.scatter(trial_numbers, trial_values, s=20, alpha=0.7, label="Objective Value")
    plt.step(trial_numbers, best_values, where="post", linewidth=2, label="Best Value")

    plt.xlabel("Trial number")
    plt.ylabel("Objective value — RMSE (years)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def summarize_split(X, y, split_name, sex_column="sex"):
    
    summary = {
        "Split": split_name,
        "n": len(y),
        "Age mean ± SD": f"{y.mean():.1f} ± {y.std():.1f}"
    }

    sex_counts = X[sex_column].value_counts(dropna=False)
    sex_percent = X[sex_column].value_counts(normalize=True, dropna=False) * 100

    summary["Male n (%)"] = f"{sex_counts.get('M', 0)} ({sex_percent.get('M', 0):.1f}%)"
    summary["Female n (%)"] = f"{sex_counts.get('F', 0)} ({sex_percent.get('F', 0):.1f}%)"

    return summary


def fit_and_evaluate_model(model, X_train, y_train, X_validate, y_validate,
                           numeric_strategy="median", categorical_strategy="most_frequent",
                           confidence=0.95, resamples=1000, seed=42,print_table = True):
    preprocessor = preprocessing(X_train, numeric_strategy, categorical_strategy)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_val = pipeline.predict(X_validate)

    boot_train_df, boot_train_dist = bootstrap_evaluation(
        confidence=confidence, prediction=y_pred_train,
        y_test=y_train, resamples=resamples, seed=seed
    )
    boot_val_df, boot_val_dist = bootstrap_evaluation(
        confidence=confidence, prediction=y_pred_val,
        y_test=y_validate, resamples=resamples, seed=seed
    )


    if print_table:
            print("Model performance in training set")
            display(boot_train_df)

            print("Model performance in validation set")
            display(boot_val_df)

    return pipeline, boot_train_df, boot_val_df, boot_train_dist, boot_val_dist


def plot_bootstrap_boxplots(dist_dict, metric_list=["RMSE", "R_squared"], title="Bootstrap Distribution — Validation Set"):
    """
    dist_dict : {"Model Name": dist_dict_from_bootstrap_evaluation, ...}
    metric_list : list of metrics to plot (any of RMSE, MAE, R_squared, Pearson_r)
    """
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    model_names = list(dist_dict.keys())
    dist_list   = list(dist_dict.values())

    # Map each metric to a y-axis label with units
    metric_ylabel_map = {
        "RMSE":      "RMSE (years)",
        "MAE":       "MAE (years)",
        "R_squared": "R² (unitless, 0–1)",
        "Pearson_r": "Pearson r (unitless, −1–1)"
    }

    fig, axes = plt.subplots(1, len(metric_list), figsize=(6 * len(metric_list), 5))
    if len(metric_list) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_list):
        data = [d[metric] for d in dist_list]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(metric.replace("_", " "), fontsize=13, fontweight="bold")
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylabel(metric_ylabel_map.get(metric, metric), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def run_models(models, X_train, y_train, X_validate, y_validate):
    all_summaries = []
    metric_names = ["RMSE", "MAE", "R_squared", "Pearson_r"]

    for name, model in models.items():
        _, boot_df_train, boot_df_val, _, _ = fit_and_evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_validate=X_validate,
            y_validate=y_validate,
            print_table=False
        )

        boot_df_train = boot_df_train.copy()
        boot_df_train["metric"] = metric_names
        boot_df_train["model"] = name
        boot_df_train["dataset"] = "train"

        boot_df_val = boot_df_val.copy()
        boot_df_val["metric"] = metric_names
        boot_df_val["model"] = name
        boot_df_val["dataset"] = "validate"

        all_summaries.extend([boot_df_train, boot_df_val])

    summary_df = pd.concat(all_summaries, ignore_index=True)
    return summary_df


def tune_multiple_models(
    X,
    y,
    models_to_tune,
    num_strategy="median",
    cat_strategy="most_frequent",
    k=30,
    trials=40,
    cv_num=5,
    seed=42
):
    all_results = {}

    for model_name, config in models_to_tune.items():
        print(f"\nTuning {model_name}...")

        tuning_results = final_tune_model_cv_mrmr(
            X=X,
            y=y,
            model=config["model"],
            param_distributions=config["param_distributions"],
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            k=k,
            trials=trials,
            cv_num=cv_num,
            seed=seed
        )

        best_rmse = tuning_results["best_rmse"]
        best_params = tuning_results["best_params"]

        print(f"Best CV RMSE: {best_rmse:.4f}")
        print(f"Best params: {best_params}")

        all_results[model_name] = {
            "model_name": model_name,
            "model": config["model"],
            "param_distributions": config["param_distributions"],
            "best_rmse": best_rmse,
            "best_params": best_params,
            "cv_results": tuning_results["cv_results"],
            "best_model": tuning_results["best_model"],
            "preprocessor": tuning_results["preprocessor"],
            "selected_features": tuning_results["selected_features"]
        }

    summary_df = pd.DataFrame([
        {
            "Model": model_name,
            "Best CV RMSE": result["best_rmse"]
        }
        for model_name, result in all_results.items()
    ]).sort_values("Best CV RMSE").reset_index(drop=True)

    best_model_name = summary_df.loc[0, "Model"]

    print(f"\nBest model type: {best_model_name}")

    return {
        "all_results": all_results,
        "summary_df": summary_df,
        "best_model_name": best_model_name,
        "final_model": all_results[best_model_name]["best_model"],
        "final_params": all_results[best_model_name]["best_params"],
        "final_cv_rmse": all_results[best_model_name]["best_rmse"],
        "final_preprocessor": all_results[best_model_name]["preprocessor"],
        "final_selected_features": all_results[best_model_name]["selected_features"],
        "final_cv_results": all_results[best_model_name]["cv_results"]
    }


def optuna_tune_multiple_models(
    models,
    X_train,
    y_train,
    k,
    n_trials=100,
    cv=5,
    num_strategy="median",
    cat_strategy="most_frequent",
    seed=42,
    plot_history=True
):
    all_results = {}

    for model_label, model in models.items():
        print(f"\nTuning {model_label}...")

        tuning_results = optuna_tune_model_mrmr(
            model=model,
            X_train=X_train,
            y_train=y_train,
            n_trials=n_trials,
            k=k,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            cv=cv,
            seed=seed
        )

        if plot_history:
            plot_optuna_history(
                tuning_results["study"],
                title=f"Optuna optimisation history - {model_label}"
            )

        print(f"Best CV RMSE: {tuning_results['best_rmse']:.4f}")
        print(f"Best params: {tuning_results['best_params']}")

        all_results[model_label] = tuning_results

    summary_df = pd.DataFrame([
        {
            "Model": model_label,
            "Best CV RMSE": result["best_rmse"]
        }
        for model_label, result in all_results.items()
    ]).sort_values("Best CV RMSE").reset_index(drop=True)

    best_model_name = summary_df.loc[0, "Model"]

    print(f"\nBest model type: {best_model_name}")

    return {
        "all_results": all_results,
        "summary_df": summary_df,
        "best_model_name": best_model_name,
        "final_model": all_results[best_model_name]["model"],
        "final_params": all_results[best_model_name]["best_params"],
        "final_cv_rmse": all_results[best_model_name]["best_rmse"],
        "final_preprocessor": all_results[best_model_name]["preprocessor"],
        "final_selected_features": all_results[best_model_name]["selected_features"],
        "final_study": all_results[best_model_name]["study"]
    }


def bootstrap_classification_evaluation(confidence, prediction, probability, y_test, resamples, seed):
    
    prediction = np.asarray(prediction)
    probability = np.asarray(probability)
    y_test = np.asarray(y_test)     

    def accuracy(prediction, y_test, axis=0):
        return np.mean(prediction == y_test, axis=axis)

    def f1(prediction, y_test, axis=0):
        # flatten to 1D — sklearn metrics don't accept 2D bootstrap arrays
        return f1_score(y_test.flatten(), prediction.flatten(), average="binary")

    def mcc(prediction, y_test, axis=0):
        return matthews_corrcoef(y_test.flatten(), prediction.flatten())

    def roc_auc(probability, y_test, axis=0):
        return roc_auc_score(y_test.flatten(), probability.flatten())

    def pr_auc(probability, y_test, axis=0):
        return average_precision_score(y_test.flatten(), probability.flatten())

    # Set common bootstrap arguments for all metrics 
    common = dict(
        n_resamples=resamples,
        confidence_level=confidence,
        random_state=seed,
        axis=0,
        paired=True,
        vectorized=False
    )

    boot_acc = stats.bootstrap((prediction, y_test), statistic=accuracy, **common)
    boot_f1  = stats.bootstrap((prediction, y_test), statistic=f1, **common)
    boot_mcc = stats.bootstrap((prediction, y_test), statistic=mcc, **common)
    boot_roc = stats.bootstrap((probability, y_test), statistic=roc_auc, **common)
    boot_pr  = stats.bootstrap((probability, y_test), statistic=pr_auc, **common)



    #Functions for SD and CI calculations
    def clean_sd(x):
        return np.nanstd(x, ddof=1)

    def clean_ci(x, confidence=0.95):
        alpha = 1 - confidence
        return (
            np.nanquantile(x, alpha / 2),
            np.nanquantile(x, 1 - alpha / 2)
        )

    # Run bootstraps
    acc_ci = clean_ci(boot_acc.bootstrap_distribution, confidence)
    f1_ci  = clean_ci(boot_f1.bootstrap_distribution, confidence)
    mcc_ci = clean_ci(boot_mcc.bootstrap_distribution, confidence)
    roc_ci = clean_ci(boot_roc.bootstrap_distribution, confidence)
    pr_ci  = clean_ci(boot_pr.bootstrap_distribution, confidence)


    # Compile results to a df
    acc_ci = clean_ci(boot_acc.bootstrap_distribution, confidence)
    f1_ci  = clean_ci(boot_f1.bootstrap_distribution, confidence)
    mcc_ci = clean_ci(boot_mcc.bootstrap_distribution, confidence)
    roc_ci = clean_ci(boot_roc.bootstrap_distribution, confidence)
    pr_ci  = clean_ci(boot_pr.bootstrap_distribution, confidence)

    results = {
        "Metric": ["Accuracy", "F1", "MCC", "ROC_AUC", "PR_AUC"],
        "Estimate": [
            accuracy(prediction, y_test),
            f1(prediction, y_test),
            mcc(prediction, y_test),
            roc_auc(probability, y_test),
            pr_auc(probability, y_test)
        ],
        "SD": [
            clean_sd(boot_acc.bootstrap_distribution),
            clean_sd(boot_f1.bootstrap_distribution),
            clean_sd(boot_mcc.bootstrap_distribution),
            clean_sd(boot_roc.bootstrap_distribution),
            clean_sd(boot_pr.bootstrap_distribution)
        ],
        "CI_Low": [acc_ci[0], f1_ci[0], mcc_ci[0], roc_ci[0], pr_ci[0]],
        "CI_High": [acc_ci[1], f1_ci[1], mcc_ci[1], roc_ci[1], pr_ci[1]]
    }

    distributions = {
        "Accuracy": boot_acc.bootstrap_distribution,
        "F1": boot_f1.bootstrap_distribution,
        "MCC": boot_mcc.bootstrap_distribution,
        "ROC_AUC": boot_roc.bootstrap_distribution,
        "PR_AUC": boot_pr.bootstrap_distribution
    }

    return pd.DataFrame(results).set_index("Metric"), distributions

def fit_and_evaluate_classifier(model, X_train, y_train, X_validate, y_validate,
                                  numeric_strategy="median", categorical_strategy="most_frequent",
                                  confidence=0.95, resamples=1000, seed=42, print_table=True):

    preprocessor = preprocessing(X_train, numeric_strategy, categorical_strategy)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_val   = pipeline.predict(X_validate)
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_val   = pipeline.predict_proba(X_validate)[:, 1]

    boot_train_df, boot_train_dist = bootstrap_classification_evaluation(
        confidence=confidence, prediction=y_pred_train, probability=y_prob_train,
        y_test=y_train, resamples=resamples, seed=seed
    )
    boot_val_df, boot_val_dist = bootstrap_classification_evaluation(
        confidence=confidence, prediction=y_pred_val, probability=y_prob_val,
        y_test=y_validate, resamples=resamples, seed=seed
    )

    if print_table:
        print("Model performance — Train set")
        display(boot_train_df)
        print("Model performance — Validation set")
        display(boot_val_df)

    return pipeline, boot_train_df, boot_val_df, boot_train_dist, boot_val_dist


def plot_roc_curves(models_dict, X_eval, y_eval, selected_features):
    COLORS = ["#2C6FAC", "#C0392B", "#27AE60", "#8E44AD", "#E67E22"]

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)

    for (name, pipeline), color in zip(models_dict.items(), COLORS):
        y_prob = pipeline.predict_proba(X_eval[selected_features])[:, 1]
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
        auc = roc_auc_score(y_eval, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False positive rate (proportion, 0–1)", fontsize=11)
    ax.set_ylabel("True positive rate (proportion, 0–1)", fontsize=11)
    ax.set_title("ROC curves — Evaluation set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig("roc_curves_combined.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_classification_bootstrap_boxplots(dist_dict,
                                            metric_list=["Accuracy", "F1", "MCC", "ROC_AUC"],
                                            title="Bootstrap distributions — Evaluation set"):
    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    model_names = list(dist_dict.keys())
    dist_list   = list(dist_dict.values())

    # Map each metric to a y-axis label with units
    metric_ylabel_map = {
        "Accuracy": "Accuracy (proportion, 0–1)",
        "F1":       "F1 score (0–1)",
        "MCC":      "MCC (unitless, −1–1)",
        "ROC_AUC":  "ROC AUC (unitless, 0–1)",
        "PR_AUC":   "PR AUC (unitless, 0–1)"
    }

    fig, axes = plt.subplots(1, len(metric_list),
                              figsize=(5 * len(metric_list), 5), dpi=300)
    if len(metric_list) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_list):
        data = [d[metric] for d in dist_list]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )

        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(metric.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(metric_ylabel_map.get(metric, metric), fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("bootstrap_boxplots_classification.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_sex_cpgs(X_development, y_development, selected_features, top_n=20):
    y_arr = np.asarray(y_development)

    pb_scores = {}
    for col in selected_features:
        if col in X_development.columns:
            r, _ = pointbiserialr(
                y_arr,
                X_development[col].fillna(X_development[col].median())
            )
            pb_scores[col] = abs(r)

    scores_series = (
        pd.Series(pb_scores)
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(7, top_n * 0.38 + 1.2), dpi=300)

    bars = ax.barh(
        scores_series.index[::-1],
        scores_series.values[::-1],
        color="#2C6FAC", edgecolor="white", linewidth=0.5
    )

    for bar, val in zip(bars, scores_series.values[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlabel("|Point-biserial r| (unitless, 0–1)", fontsize=11)
    ax.set_title(f"Top {top_n} sex-discriminative CpGs", fontsize=13,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, scores_series.max() * 1.12)
    plt.tight_layout()
    plt.savefig("top_sex_cpgs.png", dpi=300, bbox_inches="tight")
    plt.show()

    return scores_series


def classification_pipeline(models_dict,X_development,y_development, X_evaluation, y_evaluation):

    #Change to binary classification problem
    y_development = y_development.map({"F": 0, "M": 1})
    y_evaluation = y_evaluation.map({"F": 0, "M": 1})

    #Split development dataset to train and validate: 
    X_train, X_validate, y_train, y_validate = stratified_split(
    X_development, 
    y_development, 
    seed = 42, 
    training_size = 0.8,
    strata_quantity=2,
    classification=True
    )

    #Preprocessing for MRMR selection.
    preprocessor = preprocessing(X_train, impute_strategy_num='median', impute_strategy_cat='most_frequent')
    preprocessor.set_output(transform="pandas")

    #Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train) 
    X_validate_processed = preprocessor.transform(X_validate)

    selected_topk = mrmr_selection(X_train_processed, y_train, K=30, classification=True)
    selected_topk = [
    col.replace("num__", "").replace("cat__", "")
    for col in selected_topk
]

    #Initialize results dict 
    results = {}

    for model_name, model in models_dict.items():
        print(f" Running {model_name}")

        pipeline, _, _, _, _ = fit_and_evaluate_classifier(
            model=model,
            X_train=X_development[selected_topk],  
            y_train=y_development,
            X_validate=X_evaluation[selected_topk],
            y_validate=y_evaluation,
            print_table=False
        )

        #Predictions
        y_pred_eval = pipeline.predict(X_evaluation[selected_topk])
        y_prob_eval = pipeline.predict_proba(X_evaluation[selected_topk])[:, 1]

        #Bootstrap evaluation
        boot_eval_df, boot_eval_dist = bootstrap_classification_evaluation(
            confidence=0.95,
            prediction=y_pred_eval,
            probability=y_prob_eval,
            y_test=y_evaluation,
            resamples=1000,
            seed=42
        )

        print(f"\nModel performance — {model_name} — Evaluation set")
        display(boot_eval_df)

        # Confusion matrix
        cm = confusion_matrix(y_evaluation, y_pred_eval)
        fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=300)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Female (0)", "Male (1)"],
                    yticklabels=["Female (0)", "Male (1)"],
                    ax=ax, linewidths=0.5, annot_kws={"size": 12})
        ax.set_title(f"Confusion matrix — {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png",
                    dpi=300, bbox_inches="tight")
        plt.show()

        # Single-model ROC
        fpr, tpr, _ = roc_curve(y_evaluation, y_prob_eval)
        auc = roc_auc_score(y_evaluation, y_prob_eval)
        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
        ax.plot(fpr, tpr, color="#2C6FAC", linewidth=2,
                label=f"{model_name}  (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
        ax.set_xlabel("False positive rate (proportion, 0–1)", fontsize=11)
        ax.set_ylabel("True positive rate (proportion, 0–1)", fontsize=11)
        ax.set_title(f"ROC curve — {model_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"roc_{model_name.replace(' ', '_')}.png",
                    dpi=300, bbox_inches="tight")
        plt.show()


        #Store results
                # Store everything
        results[model_name] = {
            "pipeline":   pipeline,
            "boot_df":    boot_eval_df,
            "boot_dist":  boot_eval_dist,
            "y_pred":     y_pred_eval,
            "y_prob":     y_prob_eval,
        }

    # Overlaid ROC curves
    plot_roc_curves(
        models_dict={n: r["pipeline"] for n, r in results.items()},
        X_eval=X_evaluation,
        y_eval=y_evaluation,
        selected_features=selected_topk
    )

    # Bootstrap boxplots across all models
    plot_classification_bootstrap_boxplots(
        dist_dict={n: r["boot_dist"] for n, r in results.items()}
    )

    # Top 20 sex-discriminative CpGs (once, shared features)
    plot_top_sex_cpgs(X_development, y_development, selected_topk, top_n=20)

    return results, selected_topk



def mrmr_selection_quiet(X_train, y_train, K, classification=False):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if classification:
            selected_features = mrmr_classif(
                X=X_train, y=y_train, K=K, cat_features=cat_cols
            )
        else:
            selected_features = mrmr_regression(
                X=X_train, y=y_train, K=K, cat_features=cat_cols
            )

    return selected_features