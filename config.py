import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_SAMPLES = 5000
TEST_SIZE = 0.3

COST_PARAMETERS = {
    'program_cost_per_participant': 2500,
    'incarceration_cost_per_day': 85,
    'avg_sentence_days': 180
}

PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE_LARGE = (20, 12)
FIGURE_SIZE_MEDIUM = (15, 5)
FIGURE_SIZE_SMALL = (10, 6)

COLOR_PALETTE = ['coral', 'lightgreen', 'lightblue', 'orange', 'red', 'green', 'blue']
