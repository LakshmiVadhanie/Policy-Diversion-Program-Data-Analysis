import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_generation import generate_diversion_data
from data_preprocessing import clean_and_engineer_features
from descriptive_stats import generate_descriptive_statistics
from statistical_tests import conduct_statistical_tests
from logistic_regression import perform_logistic_regression
from survival_analysis import perform_survival_analysis
from propensity_matching import perform_propensity_matching
from visualizations import create_demographic_visualizations, create_outcome_visualizations, create_interactive_dashboards
from advanced_analytics import perform_advanced_analytics
from policy_recommendations import generate_policy_recommendations
from final_report import create_final_report_and_exports

from config import N_SAMPLES, PLOT_STYLE

def main():
    plt.style.use(PLOT_STYLE)
    
    print(" PRE-TRIAL DIVERSION PROGRAM POLICY EVALUATION \n")
    print("Generating dataset")
    
    df = generate_diversion_data(N_SAMPLES)
    print(f"Dataset generated with {len(df)} participants")
    
    print("\nCleaning and engineering features")
    df_clean = clean_and_engineer_features(df)
    print(f"Final dataset shape: {df_clean.shape}")
    
    print("\nGenerating descriptive statistics")
    desc_stats = generate_descriptive_statistics(df_clean)
    
    print("\nConducting statistical tests")
    test_results = conduct_statistical_tests(df_clean)
    
    print("\nCreating demographic visualizations")
    create_demographic_visualizations(df_clean)
    
    print("\nCreating outcome visualizations")
    create_outcome_visualizations(df_clean)
    
    print("\nPerforming logistic regression analysis")
    lr_model, scaler, feature_importance, auc_score = perform_logistic_regression(df_clean)
    
    print("\nPerforming survival analysis")
    kmf_diverted, kmf_not_diverted, cph, logrank_results = perform_survival_analysis(df_clean)
    
    print("\nPerforming propensity score matching")
    matched_df, balance_df, matched_ate, naive_ate = perform_propensity_matching(df_clean)
    
    print("\nCreating interactive dashboards")
    create_interactive_dashboards(df_clean)
    
    print("\nPerforming advanced analytics")
    rf_model, rf_importance, subgroup_analysis, risk_analysis = perform_advanced_analytics(df_clean)
    
    print("\nGenerating policy recommendations")
    generate_policy_recommendations(df_clean, desc_stats, test_results, matched_ate, naive_ate, 
                                  rf_importance, subgroup_analysis, risk_analysis)
    
    print("\nCreating final report")
    summary_stats, model_results, recommendations = create_final_report_and_exports(
        df_clean, desc_stats, test_results, feature_importance, subgroup_analysis, risk_analysis
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - READY FOR STAKEHOLDER PRESENTATION")
    print("="*60)

if __name__ == "__main__":
    main()
