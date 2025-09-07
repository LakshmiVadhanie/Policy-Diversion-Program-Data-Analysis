import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def conduct_statistical_tests(df):
    print(" STATISTICAL TESTING RESULTS \n")
    
    print("1. ASSOCIATION TESTS")
    print("-" * 40)
    
    gender_crosstab = pd.crosstab(df['gender'], df['diverted'])
    chi2_gender, p_gender = stats.chi2_contingency(gender_crosstab)[:2]
    print(f"Gender vs Diversion - Chi-square: {chi2_gender:.3f}, p-value: {p_gender:.4f}")
    
    race_crosstab = pd.crosstab(df['race_ethnicity'], df['diverted'])
    chi2_race, p_race = stats.chi2_contingency(race_crosstab)[:2]
    print(f"Race vs Diversion - Chi-square: {chi2_race:.3f}, p-value: {p_race:.4f}")
    
    print(f"\n2. GROUP COMPARISON TESTS")
    print("-" * 40)
    
    diverted_ages = df[df['diverted'] == 1]['age']
    non_diverted_ages = df[df['diverted'] == 0]['age']
    t_stat_age, p_age = stats.ttest_ind(diverted_ages, non_diverted_ages)
    print(f"Age - Diverted vs Non-diverted: t={t_stat_age:.3f}, p={p_age:.4f}")
    print(f"  Mean age diverted: {diverted_ages.mean():.1f}")
    print(f"  Mean age non-diverted: {non_diverted_ages.mean():.1f}")
    
    diverted_arrests = df[df['diverted'] == 1]['prior_arrests']
    non_diverted_arrests = df[df['diverted'] == 0]['prior_arrests']
    t_stat_arrests, p_arrests = stats.ttest_ind(diverted_arrests, non_diverted_arrests)
    print(f"\nPrior Arrests - Diverted vs Non-diverted: t={t_stat_arrests:.3f}, p={p_arrests:.4f}")
    print(f"  Mean arrests diverted: {diverted_arrests.mean():.2f}")
    print(f"  Mean arrests non-diverted: {non_diverted_arrests.mean():.2f}")
    
    print(f"\n3. PRIMARY OUTCOME ANALYSIS")
    print("-" * 40)
    
    diverted_recidivism = df[df['diverted'] == 1]['recidivated']
    non_diverted_recidivism = df[df['diverted'] == 0]['recidivated']
    
    count_diverted = diverted_recidivism.sum()
    count_non_diverted = non_diverted_recidivism.sum()
    n_diverted = len(diverted_recidivism)
    n_non_diverted = len(non_diverted_recidivism)
    
    z_stat, p_recidivism = proportions_ztest([count_diverted, count_non_diverted], 
                                           [n_diverted, n_non_diverted])
    
    print(f"Recidivism Rate Comparison:")
    print(f"  Diverted: {count_diverted}/{n_diverted} ({diverted_recidivism.mean():.1%})")
    print(f"  Non-diverted: {count_non_diverted}/{n_non_diverted} ({non_diverted_recidivism.mean():.1%})")
    print(f"  Z-statistic: {z_stat:.3f}, p-value: {p_recidivism:.4f}")
    
    p1 = diverted_recidivism.mean()
    p2 = non_diverted_recidivism.mean()
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    print(f"  Effect size (Cohen's h): {cohens_h:.3f}")
    
    print(f"\n4. PROGRAM EFFECTIVENESS ANALYSIS")
    print("-" * 40)
    
    diverted_df = df[df['diverted'] == 1]
    if len(diverted_df) > 0:
        completed = diverted_df[diverted_df['program_completion'] == 1]['recidivated']
        not_completed = diverted_df[diverted_df['program_completion'] == 0]['recidivated']
        
        if len(completed) > 0 and len(not_completed) > 0:
            count_completed = completed.sum()
            count_not_completed = not_completed.sum()
            n_completed = len(completed)
            n_not_completed = len(not_completed)
            
            z_stat_comp, p_comp = proportions_ztest([count_completed, count_not_completed], 
                                                  [n_completed, n_not_completed])
            
            print(f"Program Completion vs Recidivism:")
            print(f"  Completed: {count_completed}/{n_completed} ({completed.mean():.1%})")
            print(f"  Not completed: {count_not_completed}/{n_not_completed} ({not_completed.mean():.1%})")
            print(f"  Z-statistic: {z_stat_comp:.3f}, p-value: {p_comp:.4f}")
    
    return {
        'recidivism_test': {'z_stat': z_stat, 'p_value': p_recidivism, 'effect_size': cohens_h},
        'age_test': {'t_stat': t_stat_age, 'p_value': p_age},
        'arrests_test': {'t_stat': t_stat_arrests, 'p_value': p_arrests}
    }
