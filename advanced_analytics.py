import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from config import RANDOM_STATE, TEST_SIZE, FIGURE_SIZE_LARGE, COLOR_PALETTE, COST_PARAMETERS

def perform_advanced_analytics(df):
    print(" ADVANCED ANALYTICS \n")
    
    print("1. RANDOM FOREST ANALYSIS")
    print("-" * 40)
    
    feature_columns = [
        'age', 'gender_encoded', 'race_encoded', 'education_encoded',
        'employment_encoded', 'prior_arrests', 'prior_convictions',
        'offense_type_encoded', 'offense_severity_encoded', 'diverted'
    ]
    
    X = df[feature_columns]
    y = df['recidivated']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, 
                                                        random_state=RANDOM_STATE, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    rf_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Random Forest Feature Importance:")
    print(rf_importance)
    
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    print(f"\nRandom Forest AUC: {rf_auc:.3f}")
    
    print(f"\n2. SUBGROUP ANALYSIS")
    print("-" * 40)
    
    subgroups = [
        ('age_group', 'Age Group'),
        ('race_ethnicity', 'Race/Ethnicity'),
        ('offense_type', 'Offense Type'),
        ('gender', 'Gender')
    ]
    
    subgroup_results = []
    
    for var, label in subgroups:
        for category in df[var].unique():
            subgroup_df = df[df[var] == category]
            if len(subgroup_df) > 50:
                diverted_recid = subgroup_df[subgroup_df['diverted'] == 1]['recidivated'].mean()
                non_diverted_recid = subgroup_df[subgroup_df['diverted'] == 0]['recidivated'].mean()
                effect_size = non_diverted_recid - diverted_recid
                n_diverted = subgroup_df['diverted'].sum()
                n_total = len(subgroup_df)
                
                subgroup_results.append({
                    'subgroup_type': label,
                    'category': category,
                    'n_total': n_total,
                    'n_diverted': n_diverted,
                    'diverted_recid': diverted_recid,
                    'non_diverted_recid': non_diverted_recid,
                    'effect_size': effect_size
                })
    
    subgroup_df = pd.DataFrame(subgroup_results)
    print("Subgroup Analysis Results:")
    print(subgroup_df.round(3))
    
    print(f"\n3. RISK STRATIFICATION ANALYSIS")
    print("-" * 40)
    
    df['risk_score_calculated'] = (
        (df['age'] < 25).astype(int) * 1 +
        (df['prior_arrests'] > 2).astype(int) * 2 +
        (df['prior_convictions'] > 1).astype(int) * 1 +
        (df['offense_severity'] == 'High Felony').astype(int) * 2 +
        (df['employment_status'] == 'Unemployed').astype(int) * 1
    )
    
    df['risk_category'] = pd.cut(df['risk_score_calculated'], 
                                bins=[-1, 1, 3, 7], 
                                labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    risk_analysis = df.groupby(['risk_category', 'diverted'])['recidivated'].agg(['count', 'mean']).reset_index()
    risk_pivot = risk_analysis.pivot(index='risk_category', columns='diverted', values='mean')
    risk_pivot['effect_size'] = risk_pivot[False] - risk_pivot[True]
    
    print("Diversion Effectiveness by Risk Level:")
    print(risk_pivot.round(3))
    
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    
    plt.subplot(2, 4, 1)
    plt.barh(rf_importance['feature'][:8], rf_importance['importance'][:8], color=COLOR_PALETTE[2])
    plt.title('Random Forest\nFeature Importance')
    plt.xlabel('Importance Score')
    
    plt.subplot(2, 4, 2)
    models = ['Logistic Regression', 'Random Forest']
    aucs = [0.694, rf_auc]  # Using the AUC from logistic regression
    plt.bar(models, aucs, color=[COLOR_PALETTE[3], COLOR_PALETTE[1]])
    plt.title('Model Performance\n(AUC Scores)')
    plt.ylabel('AUC Score')
    plt.ylim(0.5, 1.0)
    
    plt.subplot(2, 4, 3)
    age_effects = subgroup_df[subgroup_df['subgroup_type'] == 'Age Group']
    plt.bar(age_effects['category'], age_effects['effect_size'], color='lightcoral')
    plt.title('Diversion Effect by Age Group')
    plt.ylabel('Effect Size')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 4, 4)
    risk_categories = risk_pivot.index
    plt.plot(risk_categories, risk_pivot[False], marker='o', label='Not Diverted', color=COLOR_PALETTE[4])
    plt.plot(risk_categories, risk_pivot[True], marker='s', label='Diverted', color=COLOR_PALETTE[5])
    plt.title('Recidivism by Risk Level')
    plt.ylabel('Recidivism Rate')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.subplot(2, 4, 5)
    demo_pivot = df.pivot_table(values='recidivated', index='race_ethnicity',
                               columns='diverted', aggfunc='mean')
    sns.heatmap(demo_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', cbar_kws={'label': 'Recidivism Rate'})
    plt.title('Recidivism by Race/Ethnicity\nand Diversion Status')
    
    plt.subplot(2, 4, 6)
    offense_effects = subgroup_df[subgroup_df['subgroup_type'] == 'Offense Type']
    plt.barh(offense_effects['category'], offense_effects['effect_size'], color=COLOR_PALETTE[2])
    plt.title('Diversion Effect by\nOffense Type')
    plt.xlabel('Effect Size')
    
    plt.subplot(2, 4, 7)
    diverted_only = df[df['diverted'] == 1]
    completion_impact = diverted_only.groupby('program_completion')['recidivated'].mean()
    plt.bar(['Did Not Complete', 'Completed'], completion_impact, color=['lightcoral', COLOR_PALETTE[2]])
    plt.title('Program Completion\nImpact on Recidivism')
    plt.ylabel('Recidivism Rate')
    
    plt.subplot(2, 4, 8)
    program_cost_per_participant = COST_PARAMETERS['program_cost_per_participant']
    incarceration_cost_per_day = COST_PARAMETERS['incarceration_cost_per_day']
    avg_sentence_days = COST_PARAMETERS['avg_sentence_days']
    
    total_diverted = df['diverted'].sum()
    diverted_recidivism = df[df['diverted'] == 1]['recidivated'].mean()
    non_diverted_recidivism = df[df['diverted'] == 0]['recidivated'].mean()
    effect_size = non_diverted_recidivism - diverted_recidivism
    
    recidivism_prevented = effect_size * total_diverted
    program_costs = total_diverted * program_cost_per_participant
    incarceration_savings = recidivism_prevented * incarceration_cost_per_day * avg_sentence_days
    net_savings = incarceration_savings - program_costs
    
    categories = ['Program Costs', 'Incarceration Savings', 'Net Savings']
    values = [-program_costs, incarceration_savings, net_savings]
    colors = [COLOR_PALETTE[4], COLOR_PALETTE[5], COLOR_PALETTE[6]]
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title('Cost-Benefit Analysis\n(Simulated)')
    plt.ylabel('Dollars')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, values):
        if value >= 0:
            va = 'bottom'
            y_pos = bar.get_height() + abs(value) * 0.01
        else:
            va = 'top'
            y_pos = bar.get_height() - abs(value) * 0.01
        plt.text(bar.get_x() + bar.get_width()/2, y_pos, f'${value:,.0f}',
                ha='center', va=va, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return rf_model, rf_importance, subgroup_df, risk_pivot
