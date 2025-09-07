def generate_descriptive_statistics(df):
    print(" DESCRIPTIVE STATISTICS REPORT \n")
    
    total_participants = len(df)
    diverted_count = df['diverted'].sum()
    diversion_rate = diverted_count / total_participants
    
    print(f"Total Participants: {total_participants:,}")
    print(f"Diverted Participants: {diverted_count:,}")
    print(f"Diversion Rate: {diversion_rate:.1%}\n")
    
    print(" DEMOGRAPHICS ")
    print("\nAge Statistics:")
    print(df['age'].describe())
    
    print(f"\nGender Distribution:")
    print(df['gender'].value_counts(normalize=True).map('{:.1%}'.format))
    
    print(f"\nRace/Ethnicity Distribution:")
    print(df['race_ethnicity'].value_counts(normalize=True).map('{:.1%}'.format))
    
    print(f"\nEducation Distribution:")
    print(df['education'].value_counts(normalize=True).map('{:.1%}'.format))
    
    print(f"\n CRIMINAL HISTORY ")
    print("Prior Arrests:")
    print(df['prior_arrests'].describe())
    
    print("\nPrior Convictions:")
    print(df['prior_convictions'].describe())
    
    print(f"\n OFFENSE CHARACTERISTICS ")
    print("Offense Type Distribution:")
    print(df['offense_type'].value_counts(normalize=True).map('{:.1%}'.format))
    
    print(f"\nOffense Severity Distribution:")
    print(df['offense_severity'].value_counts(normalize=True).map('{:.1%}'.format))
    
    print(f"\n PROGRAM OUTCOMES ")
    diverted_df = df[df['diverted'] == 1]
    
    if len(diverted_df) > 0:
        completion_rate = diverted_df['program_completion'].mean()
        avg_attendance = diverted_df['program_attendance'].mean()
        
        print(f"Program Completion Rate: {completion_rate:.1%}")
        print(f"Average Attendance Rate: {avg_attendance:.1%}")
    
    print(f"\n RECIDIVISM OUTCOMES ")
    overall_recidivism = df['recidivated'].mean()
    diverted_recidivism = df[df['diverted'] == 1]['recidivated'].mean()
    non_diverted_recidivism = df[df['diverted'] == 0]['recidivated'].mean()
    
    print(f"Overall Recidivism Rate: {overall_recidivism:.1%}")
    print(f"Diverted Participants Recidivism: {diverted_recidivism:.1%}")
    print(f"Non-Diverted Participants Recidivism: {non_diverted_recidivism:.1%}")
    
    effect_size = non_diverted_recidivism - diverted_recidivism
    print(f"Raw Effect Size: {effect_size:.1%}")
    
    return {
        'total_participants': total_participants,
        'diversion_rate': diversion_rate,
        'completion_rate': completion_rate if len(diverted_df) > 0 else 0,
        'overall_recidivism': overall_recidivism,
        'diverted_recidivism': diverted_recidivism,
        'non_diverted_recidivism': non_diverted_recidivism,
        'effect_size': effect_size
    }
