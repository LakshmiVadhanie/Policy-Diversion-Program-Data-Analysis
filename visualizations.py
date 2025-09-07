import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from config import FIGURE_SIZE_LARGE, FIGURE_SIZE_MEDIUM, COLOR_PALETTE

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_demographic_visualizations(df):
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_LARGE)
    fig.suptitle('Pre-Trial Diversion Program: Demographic Analysis', fontsize=16, fontweight='bold')
    
    axes[0,0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['age'].mean(), color='red', linestyle='--', label=f'Mean: {df["age"].mean():.1f}')
    axes[0,0].legend()
    
    gender_diversion = pd.crosstab(df['gender'], df['diverted'], normalize='index')
    gender_diversion.plot(kind='bar', ax=axes[0,1], color=[COLOR_PALETTE[0], COLOR_PALETTE[1]])
    axes[0,1].set_title('Diversion Rate by Gender')
    axes[0,1].set_xlabel('Gender')
    axes[0,1].set_ylabel('Proportion')
    axes[0,1].legend(['Not Diverted', 'Diverted'])
    axes[0,1].tick_params(axis='x', rotation=0)
    
    race_counts = df['race_ethnicity'].value_counts()
    axes[0,2].pie(race_counts.values, labels=race_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,2].set_title('Race/Ethnicity Distribution')
    
    edu_diversion = pd.crosstab(df['education'], df['diverted'], normalize='index')
    edu_diversion.plot(kind='bar', ax=axes[1,0], color=[COLOR_PALETTE[0], COLOR_PALETTE[1]])
    axes[1,0].set_title('Diversion Rate by Education Level')
    axes[1,0].set_xlabel('Education Level')
    axes[1,0].set_ylabel('Proportion')
    axes[1,0].legend(['Not Diverted', 'Diverted'])
    axes[1,0].tick_params(axis='x', rotation=45)
    
    axes[1,1].hist(df['prior_arrests'], bins=range(0, df['prior_arrests'].max()+2), 
                  alpha=0.7, color='orange', edgecolor='black')
    axes[1,1].set_title('Prior Arrests Distribution')
    axes[1,1].set_xlabel('Number of Prior Arrests')
    axes[1,1].set_ylabel('Frequency')
    
    offense_counts = df['offense_type'].value_counts()
    axes[1,2].bar(offense_counts.index, offense_counts.values, color=COLOR_PALETTE[2], edgecolor='black')
    axes[1,2].set_title('Offense Type Distribution')
    axes[1,2].set_xlabel('Offense Type')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_outcome_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pre-Trial Diversion Program: Outcome Analysis', fontsize=16, fontweight='bold')
    
    recidivism_by_diversion = df.groupby('diverted')['recidivated'].agg(['count', 'sum', 'mean'])
    recidivism_rates = recidivism_by_diversion['mean']
    
    bars = axes[0,0].bar(['Not Diverted', 'Diverted'], recidivism_rates, 
                        color=[COLOR_PALETTE[0], COLOR_PALETTE[1]], alpha=0.8, edgecolor='black')
    axes[0,0].set_title('Recidivism Rate by Diversion Status')
    axes[0,0].set_ylabel('Recidivism Rate')
    axes[0,0].set_ylim(0, max(recidivism_rates) * 1.2)
    
    for bar, rate in zip(bars, recidivism_rates):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    diverted_df = df[df['diverted'] == 1]
    if len(diverted_df) > 0:
        completion_recidivism = diverted_df.groupby('program_completion')['recidivated'].mean()
        bars = axes[0,1].bar(['Did Not Complete', 'Completed'], completion_recidivism, 
                            color=['lightcoral', COLOR_PALETTE[2]], alpha=0.8, edgecolor='black')
        axes[0,1].set_title('Recidivism by Program Completion\n(Diverted Participants Only)')
        axes[0,1].set_ylabel('Recidivism Rate')
        axes[0,1].set_ylim(0, max(completion_recidivism) * 1.2)
        
        for bar, rate in zip(bars, completion_recidivism):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    recidivists = df[df['recidivated'] == 1]
    if len(recidivists) > 0:
        axes[1,0].hist(recidivists[recidivists['diverted'] == 0]['time_to_recidivism'], 
                      bins=20, alpha=0.6, label='Not Diverted', color=COLOR_PALETTE[0])
        axes[1,0].hist(recidivists[recidivists['diverted'] == 1]['time_to_recidivism'], 
                      bins=20, alpha=0.6, label='Diverted', color=COLOR_PALETTE[1])
        axes[1,0].set_title('Time to Recidivism Distribution')
        axes[1,0].set_xlabel('Days to Recidivism')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
    
    employment_by_diversion = df.groupby('diverted')['employed_1yr'].mean()
    bars = axes[1,1].bar(['Not Diverted', 'Diverted'], employment_by_diversion, 
                        color=[COLOR_PALETTE[0], COLOR_PALETTE[1]], alpha=0.8, edgecolor='black')
    axes[1,1].set_title('Employment Rate at 1 Year by Diversion Status')
    axes[1,1].set_ylabel('Employment Rate')
    axes[1,1].set_ylim(0, max(employment_by_diversion) * 1.2)
    
    for bar, rate in zip(bars, employment_by_diversion):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_interactive_dashboards(df):
    print(" CREATING INTERACTIVE DASHBOARDS \n")
    
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Participants by Demographics', 'Diversion Rates by Characteristics',
                       'Program Completion Analysis', 'Outcome Comparison'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    race_counts = df['race_ethnicity'].value_counts()
    fig1.add_trace(
        go.Pie(labels=race_counts.index, values=race_counts.values, name="Race/Ethnicity"),
        row=1, col=1
    )
    
    age_diversion = df.groupby('age_group')['diverted'].mean().reset_index()
    fig1.add_trace(
        go.Bar(x=age_diversion['age_group'], y=age_diversion['diverted'], 
               name="Diversion Rate", marker_color=COLOR_PALETTE[2]),
        row=1, col=2
    )
    
    diverted_df = df[df['diverted'] == 1]
    if len(diverted_df) > 0:
        completion_by_offense = diverted_df.groupby('offense_type')['program_completion'].mean().reset_index()
        fig1.add_trace(
            go.Bar(x=completion_by_offense['offense_type'], y=completion_by_offense['program_completion'],
                   name="Completion Rate", marker_color=COLOR_PALETTE[1]),
            row=2, col=1
        )
    
    outcome_comparison = df.groupby('diverted')[['recidivated', 'employed_1yr']].mean().reset_index()
    outcome_comparison['diverted'] = outcome_comparison['diverted'].map({0: 'Not Diverted', 1: 'Diverted'})
    
    fig1.add_trace(
        go.Bar(x=outcome_comparison['diverted'], y=outcome_comparison['recidivated'],
               name="Recidivism Rate", marker_color=COLOR_PALETTE[0]),
        row=2, col=2
    )
    fig1.add_trace(
        go.Bar(x=outcome_comparison['diverted'], y=outcome_comparison['employed_1yr'],
               name="Employment Rate", marker_color=COLOR_PALETTE[2]),
        row=2, col=2
    )
    
    fig1.update_layout(height=800, title_text="Pre-Trial Diversion Program Dashboard", showlegend=True)
    fig1.show()
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=df['age'],
        y=df['prior_arrests'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['recidivated'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Recidivated"),
            opacity=0.6
        ),
        text=[f"Gender: {g}<br>Race: {r}<br>Offense: {o}<br>Diverted: {d}" 
              for g, r, o, d in zip(df['gender'], df['race_ethnicity'], 
                                   df['offense_type'], df['diverted'])],
        hovertemplate='<b>Age:</b> %{x}<br><b>Prior Arrests:</b> %{y}<br>%{text}<extra></extra>',
        name='Participants'
    ))
    
    fig2.update_layout(
        title='Risk Factor Analysis: Age vs Prior Arrests',
        xaxis_title='Age',
        yaxis_title='Number of Prior Arrests',
        height=600
    )
    fig2.show()
    
    time_points = np.arange(0, 731, 30)
    survival_data = []
    for group, label in [(df[df['diverted']==1], 'Diverted'), (df[df['diverted']==0], 'Not Diverted')]:
        if len(group) > 0:
            survival_probs = []
            for t in time_points:
                at_risk = (group['time_to_recidivism'] >= t).sum()
                if at_risk > 0:
                    events_by_t = ((group['time_to_recidivism'] <= t) & (group['recidivated'] == 1)).sum()
                    survival_prob = 1 - (events_by_t / len(group))
                else:
                    survival_prob = 0
                survival_probs.append(survival_prob)
            
            survival_data.append({
                'time': time_points,
                'survival_prob': survival_probs,
                'group': label
            })
    
    fig3 = go.Figure()
    
    colors = ['green', 'red']
    for i, data in enumerate(survival_data):
        fig3.add_trace(go.Scatter(
            x=data['time'],
            y=data['survival_prob'],
            mode='lines',
            name=data['group'],
            line=dict(color=colors[i], width=3),
            hovertemplate='<b>Days:</b> %{x}<br><b>Survival Probability:</b> %{y:.3f}<extra></extra>'
        ))
    
    fig3.update_layout(
        title='Survival Curves: Probability of No Recidivism Over Time',
        xaxis_title='Days',
        yaxis_title='Probability of No Recidivism',
        height=500,
        hovermode='x unified'
    )
    fig3.show()
    
    print("Interactive dashboards created successfully!")
