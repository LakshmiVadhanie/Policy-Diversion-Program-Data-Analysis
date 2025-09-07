# Pre-Trial Diversion Program Policy Evaluation

A comprehensive statistical analysis evaluating the effectiveness of pre-trial diversion programs in reducing recidivism rates, using advanced data science techniques for policy evaluation and causal inference.

## Overview

This project analyzes a simulated dataset of 5,000 participants over 5 years to assess whether pre-trial diversion programs effectively reduce recidivism compared to traditional prosecution. The analysis employs multiple statistical methodologies to provide robust evidence for policy decision-making.

## Key Findings

- **29% reduction in recidivism** for diverted vs non-diverted participants (20.6% vs 49.6%)
- **Statistically significant results** (p < 0.0001) across all analytical approaches
- **Causal effect confirmed** through propensity score matching (22.7% reduction after controlling for selection bias)
- **77.6% ROI** based on cost-benefit analysis
- **Differential effectiveness** across risk groups and demographics

## Technical Skills Demonstrated

### Statistical Analysis
- Descriptive statistics and exploratory data analysis
- Hypothesis testing (t-tests, chi-square, z-tests)  
- Effect size calculations and confidence intervals
- Multiple comparison corrections

### Advanced Analytics
- **Logistic Regression**: AUC 0.694, feature importance analysis
- **Survival Analysis**: Kaplan-Meier curves, Cox proportional hazards
- **Machine Learning**: Random Forest for predictive modeling
- **Causal Inference**: Propensity score matching for bias control

### Data Science Pipeline
- Data generation with realistic selection bias
- Missing value imputation and feature engineering
- Model validation and performance evaluation
- Interactive dashboards and static visualizations

## Repository Structure

```
├── main.py                    # Main execution script
├── config.py                  # Configuration parameters
├── requirements.txt           # Python dependencies
├── data_generation.py         # Realistic dataset simulation
├── data_preprocessing.py      # Data cleaning and feature engineering
├── descriptive_stats.py       # Descriptive statistical analysis
├── statistical_tests.py       # Hypothesis testing and significance tests
├── logistic_regression.py     # Logistic regression modeling
├── survival_analysis.py       # Time-to-event analysis
├── propensity_matching.py     # Causal inference methodology
├── visualizations.py          # Static and interactive plots
├── advanced_analytics.py      # Machine learning and subgroup analysis
```


## Key Methodologies

### 1. Propensity Score Matching
Controls for selection bias by matching participants with similar likelihood of program assignment based on observable characteristics.


### 2. Survival Analysis
Analyzes time-to-recidivism using Kaplan-Meier estimation and Cox proportional hazards modeling.



## Visualizations

- **Demographic Analysis**: Population characteristics and program targeting
- **Outcome Comparisons**: Recidivism rates by treatment status
- **Survival Curves**: Time-to-event analysis with confidence intervals  
- **Interactive Dashboards**: Plotly-based exploration tools
- **Model Performance**: ROC curves, feature importance plots

## Policy Recommendations

1. **Program Expansion**: Scale to 95% of eligible population
2. **Risk-Based Targeting**: Focus on low-risk participants for optimal impact
3. **Completion Improvement**: Address 34% non-completion rate
4. **Equity Monitoring**: Track disparities across demographic groups
5. **Cost-Effectiveness**: Estimated break-even at 16% recidivism reduction


## Notes

- **Data Generation**: Realistic simulation with selection bias and confounding
- **Missing Data**: Multiple imputation for realistic data quality issues
- **Model Validation**: Cross-validation and holdout testing
- **Statistical Power**: >80% power for detecting 5% effect sizes
- **Reproducibility**: Fixed random seeds and modular code structure

