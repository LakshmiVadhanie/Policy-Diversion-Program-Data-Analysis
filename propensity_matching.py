import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from config import RANDOM_STATE, FIGURE_SIZE_MEDIUM

def perform_propensity_matching(df):
    print(" PROPENSITY SCORE MATCHING ANALYSIS \n")
    
    print("1. PROPENSITY SCORE ESTIMATION")
    print("-" * 40)
    
    ps_features = ['age', 'gender_encoded', 'race_encoded', 'education_encoded',
                   'employment_encoded', 'prior_arrests', 'prior_convictions',
                   'offense_type_encoded', 'offense_severity_encoded']
    
    X_ps = df[ps_features]
    y_ps = df['diverted']
    
    ps_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    ps_model.fit(X_ps, y_ps)
    
    df_ps = df.copy()
    df_ps['propensity_score'] = ps_model.predict_proba(X_ps)[:, 1]
    
    print(f"Propensity score range: {df_ps['propensity_score'].min():.3f} - {df_ps['propensity_score'].max():.3f}")
    print(f"Mean propensity score - Diverted: {df_ps[df_ps['diverted']==1]['propensity_score'].mean():.3f}")
    print(f"Mean propensity score - Not Diverted: {df_ps[df_ps['diverted']==0]['propensity_score'].mean():.3f}")
    
    print(f"\n2. NEAREST NEIGHBOR MATCHING")
    print("-" * 40)
    
    treated = df_ps[df_ps['diverted'] == 1].copy()
    control = df_ps[df_ps['diverted'] == 0].copy()
    
    nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn_model.fit(control[['propensity_score']])
    
    distances, indices = nn_model.kneighbors(treated[['propensity_score']])
    
    matched_control_indices = control.iloc[indices.flatten()].index
    matched_treated = treated.copy()
    matched_control = control.loc[matched_control_indices].copy()
    
    matched_df = pd.concat([matched_treated, matched_control]).reset_index(drop=True)
    
    print(f"Original sample size: {len(df_ps)}")
    print(f"Matched sample size: {len(matched_df)}")
    print(f"Treated units: {len(matched_treated)}")
    print(f"Control units: {len(matched_control)}")
    
    print(f"\n3. COVARIATE BALANCE CHECK")
    print("-" * 40)
    
    def calculate_standardized_diff(treated_vals, control_vals):
        pooled_std = np.sqrt((treated_vals.var() + control_vals.var()) / 2)
        return (treated_vals.mean() - control_vals.mean()) / pooled_std if pooled_std > 0 else 0
    
    balance_results = []
    
    for feature in ps_features:
        orig_treated = df_ps[df_ps['diverted'] == 1][feature]
        orig_control = df_ps[df_ps['diverted'] == 0][feature]
        orig_smd = calculate_standardized_diff(orig_treated, orig_control)
        
        matched_treated_vals = matched_df[matched_df['diverted'] == 1][feature]
        matched_control_vals = matched_df[matched_df['diverted'] == 0][feature]
        matched_smd = calculate_standardized_diff(matched_treated_vals, matched_control_vals)
        
        balance_results.append({
            'variable': feature,
            'original_smd': orig_smd,
            'matched_smd': matched_smd,
            'improvement': abs(orig_smd) - abs(matched_smd)
        })
    
    balance_df = pd.DataFrame(balance_results)
    print("Standardized Mean Differences:")
    print(balance_df.round(3))
    
    print(f"\n4. TREATMENT EFFECT ESTIMATION")
    print("-" * 40)
    
    naive_treated_outcome = df_ps[df_ps['diverted'] == 1]['recidivated'].mean()
    naive_control_outcome = df_ps[df_ps['diverted'] == 0]['recidivated'].mean()
    naive_ate = naive_treated_outcome - naive_control_outcome
    
    matched_treated_outcome = matched_df[matched_df['diverted'] == 1]['recidivated'].mean()
    matched_control_outcome = matched_df[matched_df['diverted'] == 0]['recidivated'].mean()
    matched_ate = matched_treated_outcome - matched_control_outcome
    
    print(f"NAIVE ESTIMATE (without matching):")
    print(f"  Treated outcome: {naive_treated_outcome:.3f}")
    print(f"  Control outcome: {naive_control_outcome:.3f}")
    print(f"  Average Treatment Effect: {naive_ate:.3f}")
    
    print(f"\nMATCHED ESTIMATE:")
    print(f"  Treated outcome: {matched_treated_outcome:.3f}")
    print(f"  Control outcome: {matched_control_outcome:.3f}")
    print(f"  Average Treatment Effect: {matched_ate:.3f}")
    
    print(f"\nBias correction: {abs(naive_ate - matched_ate):.3f}")
    
    matched_treated_recid = matched_df[matched_df['diverted'] == 1]['recidivated']
    matched_control_recid = matched_df[matched_df['diverted'] == 0]['recidivated']
    
    t_stat, p_value = stats.ttest_ind(matched_treated_recid, matched_control_recid)
    print(f"T-test for matched sample: t={t_stat:.3f}, p={p_value:.4f}")
    
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    
    plt.subplot(1, 3, 1)
    plt.hist(df_ps[df_ps['diverted'] == 0]['propensity_score'], bins=20, alpha=0.6, 
             label='Not Diverted', color='red', density=True)
    plt.hist(df_ps[df_ps['diverted'] == 1]['propensity_score'], bins=20, alpha=0.6, 
             label='Diverted', color='green', density=True)
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Distribution')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(balance_df['original_smd'], balance_df['matched_smd'], alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.plot([-1, 1], [-1, 1], color='blue', linestyle='--', alpha=0.5, label='No change')
    plt.xlabel('Original SMD')
    plt.ylabel('Matched SMD')
    plt.title('Covariate Balance Improvement')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    estimates = ['Naive', 'Matched']
    effects = [naive_ate, matched_ate]
    colors = ['orange', 'blue']
    
    bars = plt.bar(estimates, effects, color=colors, alpha=0.7)
    plt.ylabel('Average Treatment Effect')
    plt.title('Treatment Effect Estimates')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    for bar, effect in zip(bars, effects):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return matched_df, balance_df, matched_ate, naive_ate
