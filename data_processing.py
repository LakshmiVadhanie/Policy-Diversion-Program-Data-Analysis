import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import RANDOM_STATE

def clean_and_engineer_features(df):
    df_clean = df.copy()
    
    np.random.seed(RANDOM_STATE)
    
    missing_mask = np.random.random(len(df_clean)) < 0.05
    df_clean.loc[missing_mask, 'education'] = np.nan
    
    missing_mask = np.random.random(len(df_clean)) < 0.03
    df_clean.loc[missing_mask, 'employment_status'] = np.nan
    
    df_clean['education'].fillna(df_clean['education'].mode()[0], inplace=True)
    df_clean['employment_status'].fillna(df_clean['employment_status'].mode()[0], inplace=True)
    
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 25, 35, 50, 100], 
                                  labels=['18-25', '26-35', '36-50', '50+'])
    
    df_clean['high_risk'] = ((df_clean['prior_arrests'] > 2) | 
                            (df_clean['age'] < 25) | 
                            (df_clean['offense_severity'] == 'High Felony')).astype(int)
    
    df_clean['successful_completion'] = ((df_clean['diverted'] == 1) & 
                                       (df_clean['program_completion'] == 1)).astype(int)
    
    df_clean['age_x_prior_arrests'] = df_clean['age'] * df_clean['prior_arrests']
    df_clean['diverted_x_completion'] = df_clean['diverted'] * df_clean['program_completion']
    
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_education = LabelEncoder()
    le_employment = LabelEncoder()
    le_offense_type = LabelEncoder()
    le_offense_severity = LabelEncoder()
    
    df_clean['gender_encoded'] = le_gender.fit_transform(df_clean['gender'])
    df_clean['race_encoded'] = le_race.fit_transform(df_clean['race_ethnicity'])
    df_clean['education_encoded'] = le_education.fit_transform(df_clean['education'])
    df_clean['employment_encoded'] = le_employment.fit_transform(df_clean['employment_status'])
    df_clean['offense_type_encoded'] = le_offense_type.fit_transform(df_clean['offense_type'])
    df_clean['offense_severity_encoded'] = le_offense_severity.fit_transform(df_clean['offense_severity'])
    
    return df_clean
