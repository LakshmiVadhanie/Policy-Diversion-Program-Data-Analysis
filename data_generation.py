import pandas as pd
import numpy as np
from config import RANDOM_STATE, N_SAMPLES

def generate_diversion_data(n_samples=N_SAMPLES):
    np.random.seed(RANDOM_STATE)
    
    age = np.random.normal(32, 12, n_samples).astype(int)
    age = np.clip(age, 18, 70)
    
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.65, 0.35])
    race_ethnicity = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                                     n_samples, p=[0.45, 0.25, 0.20, 0.05, 0.05])
    
    education = np.random.choice(['Less than HS', 'HS/GED', 'Some College', 'Bachelor+'], 
                                n_samples, p=[0.25, 0.35, 0.25, 0.15])
    
    employment_status = np.random.choice(['Employed', 'Unemployed', 'Student'], 
                                       n_samples, p=[0.55, 0.35, 0.10])
    
    prior_arrests = np.random.poisson(1.2, n_samples)
    prior_convictions = np.random.poisson(0.8, n_samples)
    
    offense_type = np.random.choice(['Drug', 'Property', 'Public Order', 'Traffic', 'Other'], 
                                   n_samples, p=[0.35, 0.25, 0.20, 0.15, 0.05])
    
    offense_severity = np.random.choice(['Misdemeanor', 'Low Felony', 'High Felony'], 
                                       n_samples, p=[0.60, 0.30, 0.10])
    
    risk_score = (age < 25).astype(int) * 2 + \
                (prior_arrests > 2).astype(int) * 3 + \
                (offense_severity == 'High Felony').astype(int) * 4 + \
                np.random.normal(0, 1, n_samples)
    
    diversion_prob = 1 / (1 + np.exp(risk_score - 2))
    diverted = np.random.binomial(1, diversion_prob, n_samples).astype(bool)
    
    program_completion = np.zeros(n_samples)
    program_attendance = np.zeros(n_samples)
    
    for i in range(n_samples):
        if diverted[i]:
            completion_prob = 0.7 - (risk_score[i] * 0.1)
            completion_prob = np.clip(completion_prob, 0.1, 0.9)
            program_completion[i] = np.random.binomial(1, completion_prob)
            program_attendance[i] = np.random.normal(0.8, 0.2)
            program_attendance[i] = np.clip(program_attendance[i], 0, 1)
    
    base_recidivism_prob = 0.35
    recidivism_prob = base_recidivism_prob + \
                     (prior_arrests > 2).astype(int) * 0.15 + \
                     (age < 25).astype(int) * 0.10 + \
                     (employment_status == 'Unemployed').astype(int) * 0.12 - \
                     (diverted & (program_completion == 1)).astype(int) * 0.20 - \
                     (diverted & (program_attendance > 0.7)).astype(int) * 0.10
    
    recidivism_prob = np.clip(recidivism_prob, 0.05, 0.8)
    recidivated = np.random.binomial(1, recidivism_prob, n_samples).astype(bool)
    
    time_to_recidivism = np.random.exponential(365, n_samples)
    time_to_recidivism *= (1 - recidivism_prob + 0.5)
    time_to_recidivism = np.clip(time_to_recidivism, 30, 730)
    
    time_to_recidivism[~recidivated] = 730
    
    employed_1yr = np.random.binomial(1, 
                                     0.65 + (diverted & (program_completion == 1)).astype(int) * 0.15 - 
                                     recidivated.astype(int) * 0.25, n_samples).astype(bool)
    
    data = pd.DataFrame({
        'participant_id': range(1, n_samples + 1),
        'age': age,
        'gender': gender,
        'race_ethnicity': race_ethnicity,
        'education': education,
        'employment_status': employment_status,
        'prior_arrests': prior_arrests,
        'prior_convictions': prior_convictions,
        'offense_type': offense_type,
        'offense_severity': offense_severity,
        'risk_score': risk_score,
        'diverted': diverted,
        'program_completion': program_completion,
        'program_attendance': program_attendance,
        'recidivated': recidivated,
        'time_to_recidivism': time_to_recidivism.astype(int),
        'employed_1yr': employed_1yr
    })
    
    return data
