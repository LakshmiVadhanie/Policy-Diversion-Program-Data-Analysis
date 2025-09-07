import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from config import FIGURE_SIZE_MEDIUM

def perform_survival_analysis(df):
    print(" SURVIVAL ANALYSIS \n")
    
    df_survival = df.copy()
    df_survival['event'] = df_survival['recidivated'].astype(int)
    df_survival['duration'] = df_survival['time_to_recidivism']
    
    print("1. KAPLAN-MEIER SURVIVAL ANALYSIS")
    print("-" * 40)
    
    kmf_overall = KaplanMeierFitter()
    kmf_overall.fit(df_survival['duration'], df_survival['event'], label='Overall')
    
    kmf_diverted = KaplanMeierFitter()
    kmf_not_diverted = KaplanMeierFitter()
    
    diverted_mask = df_survival['diverted'] == 1
    not_diverted_mask = df_survival['diverted'] == 0
    
    kmf_diverted.fit(
        df_survival.loc[diverted_mask, 'duration'], 
        df_survival.loc[diverted_mask, 'event'], 
        label='Diverted'
    )
    
    kmf_not_diverted.fit(
        df_survival.loc[not_diverted_mask, 'duration'], 
        df_survival.loc[not_diverted_mask, 'event'], 
        label='Not Diverted'
    )
    
    results = logrank_test(
        df_survival.loc[diverted_mask, 'duration'], 
        df_survival.loc[not_diverted_mask, 'duration'],
        df_survival.loc[diverted_mask, 'event'], 
        df_survival.loc[not_diverted_mask, 'event']
    )
    
    print(f"Log-rank test between diverted and non-diverted:")
    print(f"Test statistic: {results.test_statistic:.3f}")
    print(f"P-value: {results.p_value:.4f}")
    
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    
    plt.subplot(1, 3, 1)
    kmf_diverted.plot_survival_function(ax=plt.gca(), color='green', alpha=0.8)
    kmf_not_diverted.plot_survival_function(ax=plt.gca(), color='red', alpha=0.8)
    plt.title('Survival Curves by Diversion Status')
    plt.xlabel('Days')
    plt.ylabel('Probability of No Recidivism')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    kmf_diverted.plot_cumulative_hazard(ax=plt.gca(), color='green', alpha=0.8)
    kmf_not_diverted.plot_cumulative_hazard(ax=plt.gca(), color='red', alpha=0.8)
    plt.title('Cumulative Hazard by Diversion Status')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Hazard')
    plt.grid(alpha=0.3)
    
    print(f"\nMedian survival times:")
    print(f"Diverted: {kmf_diverted.median_survival_time_:.0f} days")
    print(f"Not Diverted: {kmf_not_diverted.median_survival_time_:.0f} days")
    
    print(f"\n2. COX PROPORTIONAL HAZARDS MODEL")
    print("-" * 40)
    
    cox_data = df_survival[['duration', 'event', 'age', 'gender_encoded', 'race_encoded',
                           'prior_arrests', 'prior_convictions', 'diverted']].copy()
    
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='duration', event_col='event')
    
    print("Cox Regression Summary:")
    print(cph.summary)
    
    plt.subplot(1, 3, 3)
    hazard_ratios = np.exp(cph.summary['coef'])
    ci_lower = np.exp(cph.summary['coef lower 95%'])
    ci_upper = np.exp(cph.summary['coef upper 95%'])
    
    features = hazard_ratios.index
    y_pos = range(len(features))
    
    plt.errorbar(hazard_ratios, y_pos, 
                xerr=[hazard_ratios - ci_lower, ci_upper - hazard_ratios],
                fmt='o', capsize=5, capthick=2, color='blue', alpha=0.7)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No Effect')
    plt.yticks(y_pos, features)
    plt.xlabel('Hazard Ratio')
    plt.title('Cox Model - Hazard Ratios')
    plt.grid(axis='x', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    diversion_hr = hazard_ratios['diverted']
    print(f"\n DIVERSION EFFECT ON SURVIVAL ")
    print(f"Hazard ratio for diversion: {diversion_hr:.3f}")
    print(f"95% CI: [{ci_lower['diverted']:.3f}, {ci_upper['diverted']:.3f}]")
    
    if diversion_hr < 1:
        reduction = (1 - diversion_hr) * 100
        print(f"Interpretation: Diversion reduces the hazard of recidivism by {reduction:.1f}%")
    else:
        increase = (diversion_hr - 1) * 100
        print(f"Interpretation: Diversion increases the hazard of recidivism by {increase:.1f}%")
    
    return kmf_diverted, kmf_not_diverted, cph, results
