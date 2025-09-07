import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from config import RANDOM_STATE, TEST_SIZE, FIGURE_SIZE_SMALL

def perform_logistic_regression(df):
    print(" LOGISTIC REGRESSION ANALYSIS \n")
    
    feature_columns = [
        'age', 'gender_encoded', 'race_encoded', 'education_encoded', 
        'employment_encoded', 'prior_arrests', 'prior_convictions',
        'offense_type_encoded', 'offense_severity_encoded', 'diverted'
    ]
    
    X = df[feature_columns]
    y = df['recidivated']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                        random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("MODEL PERFORMANCE:")
    print(f"AUC Score: {auc_score:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFEATURE IMPORTANCE (Coefficients):")
    print(feature_importance.round(3))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Recidivism Prediction')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    top_features = feature_importance.head(8)
    plt.barh(range(len(top_features)), top_features['coefficient'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top Feature Coefficients')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n DIVERSION EFFECT ANALYSIS ")
    diversion_coef = feature_importance[feature_importance['feature'] == 'diverted']['coefficient'].iloc[0]
    odds_ratio = np.exp(diversion_coef)
    
    print(f"Diversion coefficient: {diversion_coef:.3f}")
    print(f"Odds ratio for diversion: {odds_ratio:.3f}")
    print(f"Interpretation: Diversion is associated with {((1-odds_ratio)*100):.1f}% lower odds of recidivism")
    
    return lr_model, scaler, feature_importance, auc_score
