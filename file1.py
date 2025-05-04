
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report, 
                        confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lime import lime_tabular

# Load dataset
df = pd.read_csv('kidney_disease.csv')

# Data preprocessing
def clean_data(df):
    # Convert target variable
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    
    # Handle missing values
    df.replace('\t?', np.nan, inplace=True)
    df.replace('?', np.nan, inplace=True)
    
    # Convert numerical columns
    num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    
    # Handle categorical features
    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    df[cat_cols] = df[cat_cols].apply(lambda x: x.str.lower())
    
    # Label encoding
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

df_clean = clean_data(df)


# Feature engineering
X = df_clean.drop(['id', 'classification'], axis=1)
y = df_clean['classification']

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Drop rows where target (y) is NaN
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Feature selection using XGBoost importance
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
selector = SelectFromModel(estimator=xgb, threshold='median')
X_selected = selector.fit_transform(X_scaled, y_res)

# Get selected features
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Model training with hyperparameter tuning
models = {
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [100, 200]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(),
        'params': {
            'num_leaves': [31, 50],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0),
        'params': {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.03, 0.1]
        }
    }
}

best_models = {}
for name, config in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.2%}")

# Model evaluation
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# Explainability with SHAP
explainer = shap.TreeExplainer(best_models['XGBoost'])
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=selected_features, plot_type='bar')
plt.title('Feature Importance using SHAP')
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()

# Explainability with LIME
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=selected_features,
    class_names=['No CKD', 'CKD'],
    mode='classification'
)

exp = explainer_lime.explain_instance(
    X_test[0], 
    best_models['XGBoost'].predict_proba, 
    num_features=10
)
exp.save_to_file('lime_explanation.html')

# Save best model
import joblib
joblib.dump(best_models['XGBoost'], 'best_ckd_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')