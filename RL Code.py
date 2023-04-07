######################################################################################################
## Name:Ahmed Ali Ahmed                                                                             ##
## Assigment Multi-sensory integration network and perception of space around the body              ##
## 10/March/2023                                                                                  ##
## Refrences :                                                                                      ##
##                                                                                                  ##
######################################################################################################



################
#   Packages   #
################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.spatial import distance
from scipy.optimize import curve_fit
from scipy.stats import linregress,sem
import scipy.stats as stats
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d.axes3d import Axes3D
import string
import matplotlib.cm as cm

from matplotlib.ticker import LinearLocator
import seaborn as sns
from scipy.optimize import minimize


######################
#   Importing Data   #
######################

# Load data from CSV files
stai_scores = pd.read_csv('/Users/ahmed/Documents/UOE/Courses/Semester 2/Computational Cognitive Neuroscience /Assigments /Assigment 2/stai_scores.csv', header=None)
choices = pd.read_csv('/Users/ahmed/Documents/UOE/Courses/Semester 2/Computational Cognitive Neuroscience /Assigments /Assigment 2/inst_choices.csv', header=None)
outcomes = pd.read_csv('/Users/ahmed/Documents/UOE/Courses/Semester 2/Computational Cognitive Neuroscience /Assigments /Assigment 2/inst_outcomes.csv', header=None)

dddd
########
#  EDA #
########

'''
This code loads the STAI-Y2 scores, choice data, and outcome data from CSV files, 
and then computes various summary statistics based on the data. The mean, standard deviation,
and median of the STAI-Y2 scores are computed using NumPy functions. 
The number of healthy control subjects is found by using a cutoff value of 43 for the STAI-Y2 scores,
and the indices of these subjects are printed. The number of times each participant chose 
option A is computed by summing the number of times the value 1 appears in each row of the
choice data matrix, and then taking the average across all participants. Finally, the expected
number of aversive sounds experienced by random responses is computed based on the proportion 
of trials with aversive sounds in the outcome data.
'''

# Create a list to hold the data for each participant
data = []

# Loop through each row in the choices dataframe
for index, row in choices.iterrows():

    # Get the participant ID
    participant_id = index+1
    
    # Get the choices and outcomes for the participant
    choices_row = row.values
    outcomes_row = outcomes.loc[index].values
    
    # Combine the choices and outcomes into a single array
    trials_data = [[choice, outcome] for choice, outcome in zip(choices_row, outcomes_row)]
    
    # Add the STAI score for the participant
    stai_score = stai_scores.iloc[index-1][0]
    trials_data.append(stai_score)
    
    # Add the participant ID to the beginning of the array
    trials_data.insert(0, participant_id)
    
    # Add the trial data to the list
    data.append(trials_data)

# Create a new dataframe from the list of trial data
columns = ["Particpant"] + [f"trial_{i}" for i in range(1, 161)] + ["STAI-Y2"]
Behaviour_Data = pd.DataFrame(data, columns=columns)

#Population Grouping for the data
Behaviour_Data['Population'] = ['Anxious' if i < 25 else 'Calm' for i in range(50)]

#Populaiton Grouping based on STAI-Y2 Score
Behaviour_Data['Physical Condition'] = np.where(Behaviour_Data['STAI-Y2'] <= 43, 'Healthy', 'Not Healthy')


# Print the data info
print(Behaviour_Data.head())


# Compute mean, standard deviation, and median of STAI-Y2 scores
Mean_STAIY2 = np.mean(Behaviour_Data.loc[:,"STAI-Y2"].values.flatten())
Std_STAIY2 = np.std(Behaviour_Data.loc[:,"STAI-Y2"].values.flatten())
Median_STAIYS = np.median(Behaviour_Data.loc[:,"STAI-Y2"].values.flatten())
print(f'Mean STAI-Y2 score: {Mean_STAIY2:.2f}')
print(f'Standard deviation of STAI-Y2 scores: {Std_STAIY2:.2f}')
print(f'Median STAI-Y2 score: {Median_STAIYS:.2f}')

# Find patients ID of healthy particpants 
Healthy_Indicies = np.where(Behaviour_Data.loc[:,"Physical Condition"] =="Healthy")
Healthy_Patients = Behaviour_Data.iloc[Healthy_Indicies]["Particpant"]
print(f'Number of Healthy Particpants: {len(Healthy_Patients)}')
print(f'Healthy Particpants: {Healthy_Patients.values}')



# Compute the number of times each participant chose option A
Choices_A=[]
for trials in Behaviour_Data.values[:, 1:161]:
    Total=0
    for i in trials:
        if i[0]==1:
            Total=Total+1
    Choices_A.append(Total/160)


Avg_Choices_A = np.mean(Choices_A)

print('Number of times each participant chose Stimulus A:', Choices_A)
print('Average number of times participants chose Stimulus A:', str(Avg_Choices_A*100)+ '%')

 
# Compute the number of times each participant chose option B
Choices_B=[]
for trials in Behaviour_Data.values[:, 1:161]:
    Total=0
    for i in trials:
        if i[0]==2:
            Total=Total+1
    Choices_B.append(Total/160)


Avg_Choices_B = np.mean(Choices_B)

print('Number of times each participant chose Stimulus B:', Choices_B)
print('Average number of times participants chose Stimulus B:', str(Avg_Choices_B*100)+ '%')

# Compute the expected number of aversive sounds experienced by random responses
counts_11 = Behaviour_Data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1)
counts_21 = Behaviour_Data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1)
p_aversive = np.mean(counts_11.values + counts_21.values)

n_trials = 160
expected_aversive = p_aversive / n_trials


print('Expected number of aversive sounds experienced by random selected particpants:', str(expected_aversive*100)+'%')


############################################################
#Barplot of amount of stimuklus chosen by group if anxiety #
############################################################

# Separate the data by population
anxious_data = Behaviour_Data[Behaviour_Data['Population'] == 'Anxious']
calm_data = Behaviour_Data[Behaviour_Data['Population'] == 'Calm']

# Compute the number of times each participant chose stimulus A and B

anxious_A_choices = np.add(anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 0]), axis=1),anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1))
anxious_B_choices = np.add(anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 0]), axis=1),anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1))
calm_A_choices = np.add(calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 0]), axis=1),calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1))
calm_B_choices = np.add(calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 0]), axis=1),calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1))

# Create the bar plots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax1.bar(anxious_data.iloc[:, 0], anxious_A_choices, color='blue', alpha=0.5, label='Stimulus A')
ax1.bar(anxious_data.iloc[:, 0], anxious_B_choices, color='green', alpha=0.5, label='Stimulus B')
ax1.set_xlabel('Particpant')
ax1.set_ylabel('Number of choices')
ax1.set_title('Anxious population')
ax1.legend(loc="upper right", frameon=False)

ax2.bar(calm_data.iloc[:, 0], calm_A_choices, color='blue', alpha=0.5, label='Stimulus A')
ax2.bar(calm_data.iloc[:, 0], calm_B_choices, color='green', alpha=0.5, label='Stimulus B')
ax2.set_xlabel('Participant')
ax2.set_ylabel('Number of choices')
ax2.set_title('Calm population')
ax2.legend(loc="upper right", frameon=False)

plt.show()





#######################################################################
#Barplot of  the outcome based on stimuli chosen by group if anxiety #
#######################################################################

anxious_A_outcome = anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1)
anxious_B_outcome = anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1)
calm_A_outcome = calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1)
calm_B_outcome = calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1)

# Create the bar plots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax1.bar(anxious_data.iloc[:, 0], anxious_A_outcome, color='blue', alpha=0.5, label='Stimulus A')
ax1.bar(anxious_data.iloc[:, 0], anxious_B_outcome, color='green', alpha=0.5, label='Stimulus B')
ax1.set_xlabel('Particpant')
ax1.set_ylabel('Outcome')
ax1.set_title('Anxious population')
ax1.legend(loc="upper right", frameon=False)

ax2.bar(calm_data.iloc[:, 0], calm_A_outcome, color='blue', alpha=0.5, label='Stimulus A')
ax2.bar(calm_data.iloc[:, 0], calm_B_outcome, color='green', alpha=0.5, label='Stimulus B')
ax2.set_xlabel('Participant')
ax2.set_ylabel('Outcome')
ax2.set_title('Calm population')
ax2.legend(loc="upper right", frameon=False)

plt.show()


################################################################
#Scatterplot of amount of stimuklus chosen by group if anxiety #
################################################################
import matplotlib.pyplot as plt
import seaborn as sns

# Separate data by anxious population
high_anxious = Behaviour_Data[Behaviour_Data['Population'] == 'Anxious']
low_anxious = Behaviour_Data[Behaviour_Data['Population'] == 'Calm']

# Compute number of times each participant chose A or B
choices_A_high = np.add(anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 0]), axis=1),anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1))
choices_B_high = np.add(anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 0]), axis=1),anxious_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1))
choices_A_low = np.add(calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 0]), axis=1),calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([1, 1]), axis=1))
choices_B_low = np.add(calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 0]), axis=1),calm_data.iloc[:, 1:161].apply(lambda row: row.tolist().count([2, 1]), axis=1))


# Create scatter plot
plt.scatter(choices_A_high, choices_B_high, label='High Anxious')
plt.scatter(choices_A_low, choices_B_low, label='Low Anxious')
plt.xlabel('Number of times choosing stimulus A')
plt.ylabel('Number of times choosing stimulus B')
plt.title('Stimulus A vs. Stimulus B choices by anxiety group')
plt.legend(loc="upper right", frameon=False)
plt.show()


#######################################################################################################################################
#Scatterplot of STAI-Y2 scores and outcome of stimulus chosen , faceted per ancious and calm population, overlayed by Healthy group: #
#######################################################################################################################################
#Scatter plot of STAI-Y2 scores and outcome of stimulus chosen , faceted per ancious and calm population, overlayed by Healthy group:
import seaborn as sns
import matplotlib.pyplot as plt
# Extract aversive sound outcome from each trial in Behaviour_Data
outcomes_A = [sum([trial[1] for trial in row[1:161] if trial[0] == 1]) for index, row in Behaviour_Data.iterrows()]
outcomes_B = [sum([trial[1] for trial in row[1:161] if trial[0] == 2]) for index, row in Behaviour_Data.iterrows()]
print(outcomes_A)


# Add columns for total number of aversive sounds experienced by choosing stimulus A and B
Behaviour_Data['outcome_A'] = outcomes_A
Behaviour_Data['outcome_B'] = outcomes_B

# Create a FacetGrid plot with different colors for high and low anxious populations
g = sns.FacetGrid(Behaviour_Data, col='Physical Condition', row='Population', hue='Population', 
                  hue_kws={"marker": ["o", "s"]}, palette=['green', 'red'])

# Create scatter plot with STAI-Y2 scores on the x-axis, total number of aversive sounds experienced by choosing stimulus A on the y-axis,
# and the total number of aversive sounds experienced by choosing stimulus B as the size of the markers
g.map(sns.scatterplot, 'outcome_A','STAI-Y2', alpha=0.7)
g.map(sns.scatterplot,'outcome_B','STAI-Y2', alpha=0.7)

# Set plot titles and labels
g.fig.suptitle('STAI-Y2 scores and aversive sound by anxiety group and physical condition')
g.set_titles(row_template='{row_name}', col_template='{col_name}')
g.set_ylabels('STAI-Y2 Scores')
g.set_xlabels('Number of aversive sounds')
g.add_legend(loc="upper right", frameon=False)

plt.show()


##########################################################################################################
# Outcome of aversive sound for stimulus A or B according to the physical condition and population group #
##########################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the mean count of aversive sound for stimulus A and stimulus B for each participant
Behaviour_Data['mean_A'] = Behaviour_Data.iloc[:, 1:161].apply(lambda x: sum([1 for trial in x if trial[0] == 1 and trial[1] == 1])/160, axis=1)
Behaviour_Data['mean_B'] = Behaviour_Data.iloc[:, 1:161].apply(lambda x: sum([1 for trial in x if trial[0] == 2 and trial[1] == 1])/160, axis=1)

# Create box plots of the mean count of aversive sound for stimulus A and stimulus B, faced per population and physical condition
sns.set(style="whitegrid")

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

sns.boxplot(x='Population', y='mean_A', hue='Physical Condition', data=Behaviour_Data, ax=ax[0])
sns.boxplot(x='Population', y='mean_B', hue='Physical Condition', data=Behaviour_Data, ax=ax[1])

ax[0].set_title('Mean count of aversive sound for Stimulus A')
ax[1].set_title('Mean count of aversive sound for Stimulus B')
ax[0].set_ylabel("Mean Count")
ax[1].set_ylabel("Mean Count ")
plt.show()









############
# Model 1  #
############


# Simualtion #
##############
def simulate(alpha, beta, v0, n_trials, n_simulations):

    """
    Generate data for a reinforcement learning task.
    
    Args:
    - alpha: the learning rate for updating the value of the chosen stimulus
    - beta: the inverse temperature for the decision model
    - v0: Inital evolution values 
    - n_trials: the number of trials in the task
    - n_simulations:  the number of simualtions of expirment
    
    Returns:
    - VA: the value of stimulus A at each trial
    - VB: the value of stimulus B at each trial
    - diff: the difference between VA and VB at each trial
    - choices: the stimulus chosen at each trial, represented as 0 for stimulus B and 1 for stimulus A
    - outcomes: the outcome at each trial, represented as 0 for no sound and 1 for sound
    """

    # Store the values of V(A) and V(B) for each trial and simulation
    VA = np.zeros((n_trials, n_simulations))
    VB = np.zeros((n_trials, n_simulations))
    diff = np.zeros((n_trials, n_simulations))
    choices = np.zeros((n_trials, n_simulations)) 
    outcomes = np.zeros((n_trials, n_simulations))

    
    # Simulate n_simulations number of experiments
    for sim in range(n_simulations):
        VA[0] = v0
        VB[0] = v0
        for t in range(0,n_trials-1):
            # Determine the probability of choosing stimulus A and stimulus B
            p_choice_A = np.exp(-beta * VA[t,sim]) / (np.exp(-beta * VA[t,sim]) + np.exp(-beta * VB[t,sim]))
            p_choice_B= 1-p_choice_A

            # Determine the probability of the aversive sound given the choice
            if t < 40:
                prob_aversive_A = 0.7
                prob_aversive_B = 0.3
            elif t < 80:
                prob_aversive_A = 0.8
                prob_aversive_B = 0.2
            elif t < 120:
                prob_aversive_A = 0.6
                prob_aversive_B = 0.4
            else:
                prob_aversive_A = 0.65
                prob_aversive_B = 0.35
                

            # Generate the choices and outcome based on the choice and the probability of aversive sound
            if p_choice_B < p_choice_A:
                choices[t+1,sim] = 1
                outcomes[t+1,sim] = np.random.choice([0, 1], p=[ prob_aversive_B, prob_aversive_A])
                VA[t, sim] = VA[t, sim] + alpha * (outcomes[t, sim] - VA[t, sim])
                VB[t, sim] = VB[t-1, sim]
            else:
                choices[t,sim] = 2
                outcomes[t,sim] = np.random.choice([0, 1], p=[ prob_aversive_A, prob_aversive_B])
                VA[t+1, sim] = VA[t, sim] 
                VB[t+1, sim] = VB[t, sim] + alpha * (outcomes[t, sim] - VB[t, sim])


            # Store difference of evolutions
            diff[t,sim] = VA[t,sim] - VB[t,sim]
        
            
    return VA, VB, diff, choices,outcomes




# Likelihood fitting of parameters to individuals   #
#####################################################

def compute_NLL_model1(alpha, beta,choice,outcome, v0):
    choices = choice
    outcomes = outcome
    
    # Initialize the values of VA and VB
    VA = np.zeros(len(choices))
    VB = np.zeros(len(choices))
    VA[0] = v0
    VB[0] = v0
    NLL = 0

    for t in range(0, len(choices)-1):
        # Determine the probability of choosing stimulus A and stimulus B
        p_choice_A = np.exp(-beta * VA[t]) / (np.exp(-beta * VA[t]) + np.exp(-beta * VB[t]))
        p_choice_B = 1-p_choice_A

        # Determine the probability of the aversive sound given the choice
        if t < 40:
            prob_aversive_A = 0.7
            prob_aversive_B = 0.3
        elif t < 80:
            prob_aversive_A = 0.8
            prob_aversive_B = 0.2
        elif t < 120:
            prob_aversive_A = 0.6
            prob_aversive_B = 0.4
        else:
            prob_aversive_A = 0.65
            prob_aversive_B = 0.35

        # Compute the negative log likelihood
        if choices[t] == 1:
            NLL -= np.log(p_choice_A)
            #NLL -= np.log(prob_aversive_A if outcomes[t] == 1 else 1 - prob_aversive_B)
            VA[t+1] = VA[t] + alpha * (outcomes[t] - VA[t])
            VB[t+1] = VB[t]
        else:
            NLL -= np.log(p_choice_B)
            #NLL -= np.log(prob_aversive_B if outcomes[t] == 1 else prob_aversive_A)
            VA[t+1] = VA[t] 
            VB[t+1] = VB[t] + alpha * (outcomes[t] - VB[t])
            
    return NLL

def calculate_NLL_for_all_participants_MODEL1(data, alpha, beta, V0):
    NLLs = []
    for i in range(data.shape[0]):
        choices = [t[0] for t in data.iloc[i, 1:-3]]
        outcomes = [t[1] for t in data.iloc[i, 1:-3]]
        NLLs.append(compute_NLL_model1(alpha, beta,choices, outcomes, V0))
         
    return NLLs

def plot_NLL(NLL, participants):
    plt.plot(participants, NLL, 'o-')
    plt.xlabel('Participant')
    plt.ylabel('Average NLL')
    plt.title('Average NLL vs Participant')
    plt.axhline(NLL[3], color='red', linestyle='--')
    plt.axhline(NLL[4], color='red', linestyle='--')
    plt.annotate(f'NLL for participant 4: {NLL[3]:.2f}', (3.5, NLL[3]), color='red', fontsize=12)
    plt.annotate(f'NLL for participant 5: {NLL[4]:.2f}', (4.5, NLL[4]), color='red', fontsize=12)
    plt.show()




# Parameter Fitting   #
#######################
from scipy.optimize import minimize
from scipy.stats import pearsonr


from scipy.optimize import minimize
def NLL_params_model1_fitting(params,i, data, V0):
    alpha, beta = params[0],params[1]
    NLLs = calculate_NLL_for_participant_model1_fitting(i,data, alpha, beta, V0)
    return NLLs
def calculate_NLL_for_participant_model1_fitting(i,data, alpha, beta, V0):
    if data.shape[1] == 165:
        choices = [t[0] for t in data.iloc[i, 1:-4]]
        outcomes = [t[1] for t in data.iloc[i, 1:-4]]
    else:
        choices = [t[0] for t in data.iloc[i, 1:-3]]
        outcomes = [t[1] for t in data.iloc[i, 1:-3]]
    NLLs=compute_NLL_model1(alpha, beta,choices, outcomes, V0)
        
    return NLLs

def fit_params_for_particpant_model1_fitting(i,data, V0, starting_params):
    result = minimize(NLL_params_model1_fitting, starting_params, args=(i,data, V0), method='Nelder-Mead')
    return result.x


def fit_params_for_all_participants_model1_fitting(data, V0, starting_params):
    fitted_params = []
    for i in range(data.shape[0]):
        fitted_params.append(fit_params_for_particpant_model1_fitting(i, data, V0, starting_params))
    return np.array(fitted_params)


# Paraemter Revoery#
####################


def simulate_parameters_model1(mean, variance, n_samples):
    return np.random.multivariate_normal(mean, np.diag(variance), size=n_samples)

def simulate_data_model1(params,V0, n_trials , n_simulations):
    _, _, _, choice, outcome = simulate(params[0], params[1], V0, n_trials, n_simulations)
    return choice, outcome

def compute_NLL_model1_sim_param(alpha,beta, choice, outcome, v0):

    choices = choice
    outcomes = outcome
    # Initialize the values of VA and VB
    VA = np.zeros(len(choices))
    VB = np.zeros(len(choices))
    VA[0] = v0
    VB[0] = v0
    NLL = 0

    for t in range(0, len(choices)-1):
        # Determine the probability of choosing stimulus A and stimulus B
        p_choice_A = np.exp(-beta * VA[t]) / (np.exp(-beta * VA[t]) + np.exp(-beta * VB[t]))
        p_choice_B = 1-p_choice_A
        # Determine the probability of the aversive sound given the choice
        if t < 40:
            prob_aversive_A = 0.7
            prob_aversive_B = 0.3
        elif t < 80:
            prob_aversive_A = 0.8
            prob_aversive_B = 0.2
        elif t < 120:
            prob_aversive_A = 0.6
            prob_aversive_B = 0.4
        else:
            prob_aversive_A = 0.65
            prob_aversive_B = 0.35
        # Compute the negative log likelihood
        if choices[t] == 1:
            NLL -= np.log(p_choice_A)
            #NLL -= np.log(prob_aversive_A if outcomes[t] == 1 else 1 - prob_aversive_B)
            VA[t+1] = VA[t-1] + alpha * (outcomes[t] - VA[t])
            VB[t+1] = VB[t-1]
        else:
            NLL -= np.log(p_choice_B)
            #NLL -= np.log(prob_aversive_B if outcomes[t] == 1 else prob_aversive_A)
            VA[t+1] = VA[t] 
            VB[t+1] = VB[t] + alpha * (outcomes[t] - VB[t])
            
    return NLL


def fit_params_model1(choice, outcome, alpha, beta, v0):
    def NLL_params(params):
        alpha, beta = params[0],params[1]
        NLL = compute_NLL_model1_sim_param(alpha, beta, choice, outcome, v0)
        return NLL
    
    starting_params = [alpha, beta]
    result = minimize(NLL_params, starting_params, method='Nelder-Mead')
    return result.x


# Model  Recovery #
###################
def NLL_params_model1_recovery(params,i, data, V0):
    alpha, beta = params[0],params[1]
    NLLs = calculate_NLL_for_participant_model1_recovery(i,data, alpha, beta, V0)
    return NLLs
def calculate_NLL_for_participant_model1_recovery(i,data, alpha, beta, V0):
    choices = [t[0] for t in data.iloc[i,1:]]
    outcomes = [t[1] for t in data.iloc[i, 1:]]
    NLLs=compute_NLL_model1(alpha, beta,choices, outcomes, V0)
        
    return NLLs

def fit_params_for_particpant_model1_recovery(i,data, V0, starting_params):
    result = minimize(NLL_params_model1_recovery, starting_params, args=(i,data, V0), method='Nelder-Mead')
    return result.x


def fit_params_for_all_participants_model1_recovery(data, V0, starting_params):
    fitted_params = []
    for i in range(data.shape[0]):
        fitted_params.append(fit_params_for_particpant_model1_recovery(i, data, V0, starting_params))
    return np.array(fitted_params)


def simulate_data_model1(params,V0, n_trials , n_simulations):
    _, _, _, choice, outcome = simulate(params[0], params[1], V0, n_trials, n_simulations)
    return choice, outcome


#########################
#      Model Two        #
#########################


# Simualtion #
##############

def simulate_model2(alpha, beta, v0,A, n_trials, n_simulations):

    """
    Generate data for a reinforcement learning task.
    
    Args:
    - alpha: the learning rate for updating the value of the chosen stimulus
    - beta: the inverse temperature for the decision model
    - v0: Inital evolution values 
    - n_trials: the number of trials in the task
    - n_simulations:  the number of simualtions of expirment
    
    Returns:
    - VA: the value of stimulus A at each trial
    - VB: the value of stimulus B at each trial
    - diff: the difference between VA and VB at each trial
    - choices: the stimulus chosen at each trial, represented as 0 for stimulus B and 1 for stimulus A
    - outcomes: the outcome at each trial, represented as 0 for no sound and 1 for sound
    """

    # Store the values of V(A) and V(B) for each trial and simulation
    VA = np.zeros((n_trials, n_simulations))
    VB = np.zeros((n_trials, n_simulations))
    diff = np.zeros((n_trials, n_simulations))
    choices = np.zeros((n_trials, n_simulations)) 
    outcomes = np.zeros((n_trials, n_simulations))

    
    # Simulate n_simulations number of experiments
    for sim in range(n_simulations):
        VA[0] = v0
        VB[0] = v0
        for t in range(0,n_trials-1):
            # Determine the probability of choosing stimulus A and stimulus B
            p_choice_A = np.exp(-beta * VA[t,sim]) / (np.exp(-beta * VA[t,sim]) + np.exp(-beta * VB[t,sim]))
            p_choice_B= 1-p_choice_A

            # Determine the probability of the aversive sound given the choice
            if t < 40:
                prob_aversive_A = 0.7
                prob_aversive_B = 0.3
            elif t < 80:
                prob_aversive_A = 0.8
                prob_aversive_B = 0.2
            elif t < 120:
                prob_aversive_A = 0.6
                prob_aversive_B = 0.4
            else:
                prob_aversive_A = 0.65
                prob_aversive_B = 0.35
                

            # Generate the choices and outcome based on the choice and the probability of aversive sound
            if p_choice_B < p_choice_A:
                choices[t,sim] = 1
                outcomes[t,sim] = np.random.choice([0, 1], p=[ prob_aversive_B, prob_aversive_A])
                VA[t+1, sim] = A * VA[t, sim] + alpha * (outcomes[t, sim] - VA[t, sim])
                VB[t+1, sim] = VB[t, sim]
            else:
                choices[t,sim] = 2
                outcomes[t,sim] = np.random.choice([0, 1], p=[ prob_aversive_A, prob_aversive_B])
                VA[t+1, sim] = VA[t, sim] 
                VB[t+1, sim] = A * VB[t, sim] + alpha * (outcomes[t, sim] - VB[t, sim])


            # Store difference of evolutions
            diff[t,sim] = VA[t,sim] - VB[t,sim]
        
            
    return VA, VB, diff, choices,outcomes



# Lilkelihoof Fitting to Individuals #
######################################

def compute_NLL_model2(alpha, beta,A,choice,outcome, v0):
    choices = choice
    outcomes = outcome
    
    # Initialize the values of VA and VB
    VA = np.zeros(len(choices))
    VB = np.zeros(len(choices))
    VA[0] = v0
    VB[0] = v0
    NLL = 0

    for t in range(0, len(choices)-1):
        # Determine the probability of choosing stimulus A and stimulus B
        p_choice_A = np.exp(-beta * VA[t]) / (np.exp(-beta * VA[t]) + np.exp(-beta * VB[t]))
        p_choice_B = 1-p_choice_A

        # Compute the negative log likelihood
        if choices[t] == 1:
            NLL -= np.log(p_choice_A)
            #NLL -= np.log(prob_aversive_A if outcomes[t] == 1 else 1 - prob_aversive_B)
            VA[t+1] = A*VA[t] + alpha * (outcomes[t] - VA[t])
            VB[t+1] = VB[t]
        else:
            NLL -= np.log(p_choice_B)
            #NLL -= np.log(prob_aversive_B if outcomes[t] == 1 else prob_aversive_A)
            VA[t+1] = VA[t] 
            VB[t+1] = A*VB[t] + alpha * (outcomes[t] - VB[t])
            
    return NLL

def calculate_NLL_for_all_participants_model2(data, alpha, beta,A, V0):
    NLLs = []
    for i in range(data.shape[0]):

        choices = [t[0] for t in data.iloc[i, 1:-4]]
        outcomes = [t[1] for t in data.iloc[i, 1:-4]]
        NLLs.append(compute_NLL_model2(alpha, beta,A,choices, outcomes, V0))
         
    return NLLs

def plot_NLL_model2(NLL, participants):
    plt.plot(participants, NLL, 'o-')
    plt.xlabel('Participant')
    plt.ylabel('Average NLL')
    plt.title('Average NLL vs Participant')
    plt.axhline(NLL[3], color='red', linestyle='--')
    plt.axhline(NLL[4], color='red', linestyle='--')
    plt.annotate(f'NLL for participant 4: {NLL[3]:.2f}', (3.5, NLL[3]), color='red', fontsize=12)
    plt.annotate(f'NLL for participant 5: {NLL[4]:.2f}', (4.5, NLL[4]), color='red', fontsize=12)
    plt.show()


# Model Fitting   #
###################
from scipy.optimize import minimize
from scipy.stats import pearsonr


from scipy.optimize import minimize
def NLL_params_model2_fitted(params,i, data, V0):
    alpha, beta,A = params[0],params[1],params[2]

    NLLs = calculate_NLL_for_participant_model2_fitted(i,data, alpha, beta,A, V0)
    return NLLs

def calculate_NLL_for_participant_model2_fitted(i,data, alpha, beta,A, V0):

    if data.shape[1] == 166:
        choices = [t[0] for t in data.iloc[i, 1:-5]]
        outcomes = [t[1] for t in data.iloc[i, 1:-5]]
    else:
        choices = [t[0] for t in data.iloc[i, 1:-3]]
        outcomes = [t[1] for t in data.iloc[i, 1:-3]]
    NLLs=compute_NLL_model2(alpha, beta,A,choices, outcomes, V0)
        
    return NLLs

def fit_params_for_particpant_model2_fitted(i,data, V0, starting_params):
    result = minimize(NLL_params_model2_fitted, starting_params, args=(i,data, V0), bounds = ((0, 1), (0, 10),(-5,5)),method='Nelder-Mead')
    return result.x


def fit_params_for_all_participants_model2_fitted(data, V0, starting_params):
    fitted_params = []
    for i in range(data.shape[0]):
        fitted_params.append(fit_params_for_particpant_model2_fitted(i, data, V0, starting_params))
    return np.array(fitted_params)


# Paraemter Revoery#
####################


def simulate_parameters_model2(mean, variance, n_samples):
    return np.random.multivariate_normal(mean, np.diag(variance), size=n_samples)

def simulate_data_model2(params,V0, n_trials , n_simulations):
    _, _, _, choice, outcome = simulate_model2(params[0], params[1], V0,params[2], n_trials, n_simulations)
    return choice, outcome



def compute_NLL_model2_sim_param(alpha, beta,A,choice,outcome, v0):

    choices = choice
    outcomes = outcome
    # Initialize the values of VA and VB
    VA = np.zeros(len(choices))
    VB = np.zeros(len(choices))
    VA[0] = v0
    VB[0] = v0
    NLL = 0

    for t in range(0, len(choices)-1):
        # Determine the probability of choosing stimulus A and stimulus B
        p_choice_A = np.exp(-beta * VA[t]) / (np.exp(-beta * VA[t]) + np.exp(-beta * VB[t]))
        p_choice_B = 1-p_choice_A
        # Determine the probability of the aversive sound given the choice
        if t < 40:
            prob_aversive_A = 0.7
            prob_aversive_B = 0.3
        elif t < 80:
            prob_aversive_A = 0.8
            prob_aversive_B = 0.2
        elif t < 120:
            prob_aversive_A = 0.6
            prob_aversive_B = 0.4
        else:
            prob_aversive_A = 0.65
            prob_aversive_B = 0.35
        # Compute the negative log likelihood
        if choices[t] == 1:
            NLL -= np.log(p_choice_A)
            #NLL -= np.log(prob_aversive_A if outcomes[t] == 1 else 1 - prob_aversive_B)
            VA[t+1] = A*VA[t] + alpha * (outcomes[t] - VA[t])
            VB[t+1] = VB[t]
        else:
            NLL -= np.log(p_choice_B)
            #NLL -= np.log(prob_aversive_B if outcomes[t] == 1 else prob_aversive_A)
            VA[t+1] = VA[t] 
            VB[t+1] =  A*VB[t] + alpha * (outcomes[t] - VB[t])
            
    return NLL


def fit_params_model2(choice, outcome, alpha, beta,A, v0):
    def NLL_params(params):
        alpha, beta,A = params[0],params[1],params[2]
        NLL = compute_NLL_model2_sim_param(alpha, beta,A, choice, outcome, v0)
        return NLL
    
    starting_params = [alpha, beta,A]
    result = minimize(NLL_params, starting_params, method='Nelder-Mead')
    return result.x


# Model Recovery #
##################

# Set parameters 
alpha=0.4 
beta=5 
A=0.5
V0 = 0.5
n_trials=160
n_simulations= 1

def NLL_params_model2_recovery(params,i, data, V0):
    alpha, beta,A = params[0],params[1],params[2]

    NLLs = calculate_NLL_for_participant_model2_recovery(i,data, alpha, beta,A, V0)
    return NLLs

def calculate_NLL_for_participant_model2_recovery(i,data, alpha, beta,A, V0):
    choices = [t[0] for t in data.iloc[i, 1:]]
    outcomes = [t[1] for t in data.iloc[i, 1:]]
    NLLs=compute_NLL_model2(alpha, beta,A,choices, outcomes, V0)
        
    return NLLs

def fit_params_for_particpant_model2_recovery(i,data, V0, starting_params):
    result = minimize(NLL_params_model2_recovery, starting_params, args=(i,data, V0), method='Nelder-Mead')
    return result.x


def fit_params_for_all_participants_model2_recovery(data, V0, starting_params):
    fitted_params = []
    for i in range(data.shape[0]):
        fitted_params.append(fit_params_for_particpant_model2_recovery(i, data, V0, starting_params))
    return np.array(fitted_params)





################
# Model Three  #
################


# Simualtion #
##############
def simulate_model3(alpha_pos, beta, v0,alpha_neg, n_trials, n_simulations):

    """
    Generate data for a reinforcement learning task.
    
    Args:
    - alpha: the learning rate for updating the value of the chosen stimulus
    - beta: the inverse temperature for the decision model
    - v0: Inital evolution values 
    - n_trials: the number of trials in the task
    - n_simulations:  the number of simualtions of expirment
    
    Returns:
    - VA: the value of stimulus A at each trial
    - VB: the value of stimulus B at each trial
    - diff: the difference between VA and VB at each trial
    - choices: the stimulus chosen at each trial, represented as 0 for stimulus B and 1 for stimulus A
    - outcomes: the outcome at each trial, represented as 0 for no sound and 1 for sound
    """

    # Store the values of V(A) and V(B) for each trial and simulation
    VA = np.zeros((n_trials, n_simulations))
    VB = np.zeros((n_trials, n_simulations))
    diff = np.zeros((n_trials, n_simulations))
    choices = np.zeros((n_trials, n_simulations)) 
    outcomes = np.zeros((n_trials, n_simulations))

    
    # Simulate n_simulations number of experiments
    for sim in range(n_simulations):
        VA[0] = v0
        VB[0] = v0
        for t in range(0,n_trials-1):
            # Determine the probability of choosing stimulus A and stimulus B
            p_choice_A = np.exp(-beta * VA[t,sim]) / (np.exp(-beta * VA[t,sim]) + np.exp(-beta * VB[t,sim]))
            p_choice_B= 1-p_choice_A

            # Determine the probability of the aversive sound given the choice
            if t < 40:
                prob_aversive_A = 0.7
                prob_aversive_B = 0.3
            elif t < 80:
                prob_aversive_A = 0.8
                prob_aversive_B = 0.2
            elif t < 120:
                prob_aversive_A = 0.6
                prob_aversive_B = 0.4
            else:
                prob_aversive_A = 0.65
                prob_aversive_B = 0.35
          
            # Generate the choices and outcome based on the choice and the probability of aversive sound
            if p_choice_B < p_choice_A:
                choices[t,sim] = 1
                outcomes[t,sim] = np.random.choice([0, 1], p=[ prob_aversive_B, prob_aversive_A])
                VA[t+1, sim] = VA[t, sim] + (((1-outcomes[t, sim])*alpha_pos)+outcomes[t, sim]*alpha_neg)*(outcomes[t, sim] - VA[t, sim])
                VB[t+1, sim] = VB[t, sim]
            else:
                choices[t,sim] = 2
                outcomes[t,sim] = np.random.choice([0, 1], p=[ prob_aversive_A, prob_aversive_B])
                VA[t+1, sim] = VA[t, sim] 
                VB[t+1, sim] = VB[t, sim] + (((1-outcomes[t, sim])*alpha_pos)+outcomes[t, sim]*alpha_neg)*(outcomes[t, sim] - VB[t, sim])


            # Store difference of evolutions
            diff[t,sim] = VA[t,sim] - VB[t,sim]
        
            
    return VA, VB, diff, choices,outcomes



# Lilkelihoof Fitting to Individuals #
######################################


def compute_NLL_model3(alpha_pos,alpha_neg, beta,choice,outcome, v0):
    choices = choice
    outcomes = outcome
    
    # Initialize the values of VA and VB
    VA = np.zeros(len(choices))
    VB = np.zeros(len(choices))
    VA[0] = v0
    VB[0] = v0
    NLL = 0

    for t in range(0, len(choices)-1):
        # Determine the probability of choosing stimulus A and stimulus B
        p_choice_A = np.exp(-beta * VA[t]) / (np.exp(-beta * VA[t]) + np.exp(-beta * VB[t]))
        p_choice_B = 1-p_choice_A

        # Compute the negative log likelihood
        if choices[t] == 1:
            NLL -= np.log(p_choice_A)
            #NLL -= np.log(prob_aversive_A if outcomes[t] == 1 else 1 - prob_aversive_B)
            VA[t+1] = VA[t] + (((1-outcomes[t])*alpha_pos)+outcomes[t]*alpha_neg)*(outcomes[t] - VA[t])
            VB[t+1] = VB[t]
        else:
            NLL -= np.log(p_choice_B)
            #NLL -= np.log(prob_aversive_B if outcomes[t] == 1 else prob_aversive_A)
            VA[t+1] = VA[t] 
            VB[t+1] = VB[t] + (((1-outcomes[t])*alpha_pos)+outcomes[t]*alpha_neg)*(outcomes[t] - VB[t])
            
    return NLL


def calculate_NLL_for_all_participants_model3(data, alpha_pos, beta,alpha_neg, V0):
    NLLs = []

    for i in range(data.shape[0]):
        choices = [t[0] for t in data.iloc[i, 1:-5]]
        outcomes = [t[1] for t in data.iloc[i, 1:-5]]
        NLLs.append(compute_NLL_model3(alpha_pos,alpha_neg,beta,choices, outcomes, V0))
         
    return NLLs

def plot_NLL_model3(NLL, participants):
    plt.plot(participants, NLL, 'o-')
    plt.xlabel('Participant')
    plt.ylabel('Average NLL')
    plt.title('Average NLL vs Participant')
    plt.axhline(NLL[3], color='red', linestyle='--')
    plt.axhline(NLL[4], color='red', linestyle='--')
    plt.annotate(f'NLL for participant 4: {NLL[3]:.2f}', (3.5, NLL[3]), color='red', fontsize=12)
    plt.annotate(f'NLL for participant 5: {NLL[4]:.2f}', (4.5, NLL[4]), color='red', fontsize=12)
    plt.show()


# Model Fitting   #
###################



from scipy.optimize import minimize
def NLL_params_model3_fitted(params,i, data, V0):
    alpha_pos, alpha_neg,beta = params[0],params[1],params[2]

    NLLs = calculate_NLL_for_participant_model3_fitted(i,data, alpha_pos, alpha_neg,beta, V0)
    return NLLs

def calculate_NLL_for_participant_model3_fitted(i,data, alphapos, alphaneg,beta, V0):

    if data.shape[1] == 167:
        choices = [t[0] for t in data.iloc[i, 1:-6]]
        outcomes = [t[1] for t in data.iloc[i, 1:-6]]
    else:
        choices = [t[0] for t in data.iloc[i, 1:-3]]
        outcomes = [t[1] for t in data.iloc[i, 1:-3]]
    NLLs=compute_NLL_model3(alphapos, alphaneg,beta,choices, outcomes, V0)
        
    return NLLs

def fit_params_for_particpant_model3_fitted(i,data, V0, starting_params):
    result = minimize(NLL_params_model3_fitted, starting_params, args=(i,data, V0), bounds = ((-1, 1), (-1, 1),(0,10)),method='Nelder-Mead')
    return result.x


def fit_params_for_all_participants_model3_fitted(data, V0, starting_params):
    fitted_params = []
    for i in range(data.shape[0]):
        fitted_params.append(fit_params_for_particpant_model3_fitted(i, data, V0, starting_params))
    return np.array(fitted_params)


# Parameter  recovery #
#######################

def simulate_parameters_model3(mean, variance, n_samples):
    return np.random.multivariate_normal(mean, np.diag(variance), size=n_samples)

def simulate_data_model3(params,V0, n_trials , n_simulations):
    _, _, _, choice, outcome = simulate_model3(params[0], params[1], V0,params[2], n_trials, n_simulations)
    return choice, outcome



def compute_NLL_model3_sim_param(alpha_pos, beta,alpha_neg,choice,outcome, v0):

    choices = choice
    outcomes = outcome
    # Initialize the values of VA and VB
    VA = np.zeros(len(choices))
    VB = np.zeros(len(choices))
    VA[0] = v0
    VB[0] = v0
    NLL = 0

    for t in range(0, len(choices)-1):
        # Determine the probability of choosing stimulus A and stimulus B
        p_choice_A = np.exp(-beta * VA[t]) / (np.exp(-beta * VA[t]) + np.exp(-beta * VB[t]))
        p_choice_B = 1-p_choice_A
        # Determine the probability of the aversive sound given the choice
        if t < 40:
            prob_aversive_A = 0.7
            prob_aversive_B = 0.3
        elif t < 80:
            prob_aversive_A = 0.8
            prob_aversive_B = 0.2
        elif t < 120:
            prob_aversive_A = 0.6
            prob_aversive_B = 0.4
        else:
            prob_aversive_A = 0.65
            prob_aversive_B = 0.35
        # Compute the negative log likelihood
        if choices[t] == 1:
            NLL -= np.log(p_choice_A)
             
            #NLL -= np.log(prob_aversive_A if outcomes[t] == 1 else 1 - prob_aversive_B)
            VA[t+1] = VA[t] + (((1-outcomes[t])*alpha_pos)+outcomes[t]*alpha_neg)*(outcomes[t] - VA[t])
            VB[t+1] = VB[t]
        else:
            NLL -= np.log(p_choice_B)
            #NLL -= np.log(prob_aversive_B if outcomes[t] == 1 else prob_aversive_A)
            VA[t+1] = VA[t] 
            VB[t+1] =  VB[t] + (((1-outcomes[t])*alpha_pos)+outcomes[t]*alpha_neg)*(outcomes[t] - VB[t])
            
    return NLL


def fit_params_model3(choice, outcome, alpha_pos, beta,alpha_neg, v0):
    def NLL_params(params):
        alpha_pos, beta,alpha_neg = params[0],params[1],params[2]
        NLL = compute_NLL_model3_sim_param(alpha_pos, beta,alpha_neg, choice, outcome, v0)
        return NLL
    
    starting_params = [alpha_pos, beta,alpha_neg]
    result = minimize(NLL_params, starting_params, method='Nelder-Mead')
    return result.x


# Model Recovery #
###################

def NLL_params_model3_recvoery(params,i, data, V0):
    alpha_pos, beta,alpha_neg= params[0],params[1],params[2]

    NLLs = calculate_NLL_for_participant_model3_recvoery(i,data, alpha_pos, beta,alpha_neg, V0)
    return NLLs

def calculate_NLL_for_participant_model3_recvoery(i,data, alpha_pos, beta,alpha_neg, V0):

    choices = [t[0] for t in data.iloc[i, 1:]]
    outcomes = [t[1] for t in data.iloc[i, 1:]]
    NLLs=compute_NLL_model3(alpha_pos, alpha_neg,beta,choices, outcomes, V0)
        
    return NLLs

def fit_params_for_particpant_model3_recvoery(i,data, V0, starting_params):
    result = minimize(NLL_params_model3_recvoery, starting_params, args=(i,data, V0), method='Nelder-Mead')
    return result.x


def fit_params_for_all_participants_model3_recovery(data, V0, starting_params):
    fitted_params = []
    for i in range(data.shape[0]):
        fitted_params.append(fit_params_for_particpant_model3_recvoery(i, data, V0, starting_params))
    return np.array(fitted_params)

#Simulate Data with those paremters 

def simulate_data_model3(params,V0, n_trials , n_simulations):
    _, _, _, choice, outcome = simulate_model3(params[0], params[2], V0,params[1], n_trials, n_simulations)
    return choice, outcome














'''
##################
# Model Fitting  #
##################

# Intial Model Exploaration  #
##############################
## Set the parameters
alpha1 = 0.4
beta1 = 7
v01 = 0.5
alpha2 = 0.4
beta2 = 5
v02 = 0.5
A=0.5
alpha3_pos = 0.25
alpha3_neg=0.5
beta3 = 6
v03 = 0.5
n_trials = 160
n_simulations = 1000


##Evolution of V(A) and V(B)

# Simulate the experiments for the first model
VA_values1, VB_values1, diff_values1, choices1, outcomes1 = simulate(alpha1, beta1, v01, n_trials, n_simulations)

# Simulate the experiments for the second model
VA_values2, VB_values2, diff_values2, choices2, outcomes2 = simulate_model2(alpha2, beta2, v02,A, n_trials, n_simulations)

# Simulate the experiments for the third model
VA_values3, VB_values3, diff_values3, choices3, outcomes3 = simulate_model3(alpha3_pos,beta3,v03,alpha3_neg,n_trials, n_simulations)

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot the average evolution of values V(A) and V(B) for all the models
ax1.plot(np.mean(VA_values1, axis=1), label='Model 1')
ax1.plot(np.mean(VA_values2, axis=1), label='Model 2')
ax1.plot(np.mean(VA_values3, axis=1), label='Model 3')
ax1.set_xlabel('Trials')
ax1.set_ylabel('V(A)')
ax1.legend(loc="upper right", frameon=False)

# Plot the average evolution of values V(B) for all the models
ax2.plot(np.mean(VB_values1, axis=1), label='Model 1')
ax2.plot(np.mean(VB_values2, axis=1), label='Model 2')
ax2.plot(np.mean(VB_values3, axis=1), label='Model 3')
ax2.set_xlabel('Trials')
ax2.set_ylabel('Value')
ax2.legend(loc="upper right", frameon=False)

plt.show()


##  V(A)-V(B)  
fig, ax = plt.subplots()
ax.plot(np.mean(diff_values1, axis=1), label='Model 1')
ax.plot(np.mean(diff_values2, axis=1), label='Model 2')
ax.plot(np.mean(diff_values3, axis=1), label='Model 3')
ax.set_xlabel('Trials')
ax.set_ylabel('V(A) - V(B)')
ax.legend(loc="upper right", frameon=False)
plt.show()


##Aversaive sound
# Plot the average number of aversive sounds across each trial
averages1 = np.mean(outcomes1, axis=1)
averages2 = np.mean(outcomes2, axis=1)
averages3 = np.mean(outcomes3, axis=1)

plt.plot(range(n_trials), averages1*160, label='Model 1')
plt.plot(range(n_trials), averages2*160, label='Model 2')
plt.plot(range(n_trials), averages3*160, label='Model 3')
plt.xlabel('Trial')
plt.ylabel('Average number of aversive sounds')
plt.legend(loc="upper right", frameon=False)
plt.show()
'''

























######################
# Paremter Fitting  #
#####################

# Model One  #
##############

V0 = 0.5
starting_params = [0.4, 7]
fitted_params = fit_params_for_all_participants_model1_fitting(Behaviour_Data, V0, starting_params)
high_anxious = fitted_params[:25, :]
low_anxious = fitted_params[25:, :]

print("mean of fitted paremters:",np.mean(fitted_params,axis=0))
print("varaince of fitted paremters:",np.var(fitted_params,axis=0))

'''
# Fitted parameers before and after outlier

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Fitted Parameter Values before removing the outlier
fitted_params_before = fitted_params
mean_before = np.mean(fitted_params_before, axis=0)
var_before = np.var(fitted_params_before, axis=0)

ax[0].plot(fitted_params_before[:, 0], 'o', label='Learning rate (alpha)')
ax[0].plot(fitted_params_before[:, 1], 'o', label='Inverse temperature (beta)')
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_xlabel('Participant Index')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Fitted parameter values (before outlier removal)')
ax[0].annotate(f'Mean: {np.round(mean_before,3)}', (0.05, 0.9), xycoords='axes fraction', fontsize=12)
ax[0].annotate(f'Variance: {np.round(var_before,3)}', (0.05, 0.85), xycoords='axes fraction', fontsize=12)

# Fitted Parameter Values after removing the outlier
fitted_params_after = np.delete(fitted_params, 29, axis=0)
mean_after = np.mean(fitted_params_after, axis=0)
var_after = np.var(fitted_params_after, axis=0)

ax[1].plot(fitted_params_after[:, 0], 'o', label='Learning rate (alpha)')
ax[1].plot(fitted_params_after[:, 1], 'o', label='Inverse temperature (beta)')
ax[1].legend(loc="upper right", frameon=False)
ax[1].set_xlabel('Participant Index')
ax[1].set_ylabel('Parameter Value')
ax[1].set_title('Fitted parameter values (after outlier removal)')
ax[1].annotate(f'Mean: {np.round(mean_after,3)}', (0.05, 0.9), xycoords='axes fraction', fontsize=12)
ax[1].annotate(f'Variance: {np.round(var_after,3)}', (0.05, 0.85), xycoords='axes fraction', fontsize=12)

plt.tight_layout()
plt.show()



# Fitted parameers before and after outlier, facet per group of anxious and calm



# Fitted Parameter Values before removing the outlier
fitted_params_before = fitted_params
mean_before = np.mean(fitted_params_before, axis=0)
var_before = np.var(fitted_params_before, axis=0)

# Fitted Parameter Values after removing the outlier
fitted_params_after = np.delete(fitted_params, 29, axis=0)
mean_after = np.mean(fitted_params_after, axis=0)
var_after = np.var(fitted_params_after, axis=0)




# Fitted parameers before and after outlier, facet per group of anxious and calm

high_anxious = fitted_params[:25, :]
low_anxious_without_outlier = np.concatenate((fitted_params[25:28, :], fitted_params[30:, :]))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Calm group before excluding outlier
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 0], ax=ax[0, 0], color='blue', label='Learning rate (alpha)')
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 1], ax=ax[0, 0], color='red', label='Inverse temperature (beta)')
ax[0, 0].set_xlabel('Participant')
ax[0, 0].set_ylabel('Parameter Value')
ax[0, 0].set_title('Calm Group Before Excluding Outlier')
ax[0, 0].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(low_anxious[:, 0])
var_alpha = np.var(low_anxious[:, 0])
mean_beta = np.mean(low_anxious[:, 1])
var_beta = np.var(low_anxious[:, 1])
ax[0, 0].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
ax[0, 0].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)

# Calm group after excluding outlier
sns.scatterplot(x=np.arange(low_anxious_without_outlier.shape[0]), y=low_anxious_without_outlier[:, 0], ax=ax[0, 1], color='blue', label='Learning rate (alpha)')
sns.scatterplot(x=np.arange(low_anxious_without_outlier.shape[0]), y=low_anxious_without_outlier[:, 1], ax=ax[0, 1], color='red', label='Inverse temperature (beta)')
ax[0, 1].set_xlabel('Participant')
ax[0, 1].set_ylabel('Parameter Value')
ax[0, 1].set_title('Calm Group')
ax[0, 1].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(low_anxious_without_outlier[:, 0])
var_alpha = np.var(low_anxious_without_outlier[:, 0])
mean_beta = np.mean(low_anxious_without_outlier[:, 1])
var_beta = np.var(low_anxious_without_outlier[:, 1])
ax[0, 1].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}',xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
ax[0, 1].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)



sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 0], ax=ax[1, 0], color='blue', label='Learning rate (alpha)')
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 1], ax=ax[1, 0], color='red', label='Inverse temperature (beta)')
ax[1, 0].set_xlabel('Participant')
ax[1, 0].set_ylabel('Parameter Value')
ax[1, 0].set_title('Anxious Group Before Excluding Outlier')
ax[1, 0].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(high_anxious[:, 0])
var_alpha = np.var(high_anxious[:, 0])
mean_beta = np.mean(high_anxious[:, 1])
var_beta = np.var(high_anxious[:, 1])
ax[1, 0].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
ax[1, 0].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)


sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 0], ax=ax[1, 1], color='blue', label='Learning rate (alpha)')
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 1], ax=ax[1, 1], color='red', label='Inverse temperature (beta)')
ax[1, 1].set_xlabel('Participant')
ax[1, 1].set_ylabel('Parameter Value')
ax[1, 1].set_title('Anxious Group')
ax[1, 1].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(high_anxious[:, 0])
var_alpha = np.var(high_anxious[:, 0])
mean_beta = np.mean(high_anxious[:, 1])
var_beta = np.var(high_anxious[:, 1])
ax[1, 1].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
ax[1, 1].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.8),xycoords='axes fraction', fontsize=10)

plt.show()




mean_fitted_params = np.mean(fitted_params_after, axis=0)
var_fitted_params = np.var(fitted_params_after, axis=0)


mean_fitted_params_calm = np.mean(low_anxious_without_outlier, axis=0)
var_fitted_params_calm = np.var(low_anxious_without_outlier, axis=0)

mean_fitted_params_anxious = np.mean(high_anxious, axis=0)
var_fitted_params_anxious = np.var(high_anxious, axis=0)



corr_coeff = np.corrcoef(fitted_params_after[:, 0], fitted_params_after[:, 1])[0, 1]
print("Pearson's correlation coefficient between estimated parameters for all Particpants: ", corr_coeff)


corr_coeff_high = np.corrcoef(high_anxious[:, 0], high_anxious[:, 1])[0, 1]
corr_coeff_low = np.corrcoef(low_anxious_without_outlier[:, 0], low_anxious_without_outlier[:, 1])[0, 1]
print("Pearson's correlation coefficient between estimated parameters for high anxious group: ", corr_coeff_high)
print("Pearson's correlation coefficient between estimated parameters for low anxious group: ", corr_coeff_low)




#Plotting correlationn between alpha and beta 

fig, ax = plt.subplots()
sns.regplot(x=fitted_params_after[:, 0], y=fitted_params_after[:, 1], scatter_kws={'s': 50}, ax=ax)
plt.xlabel('Learning rate (alpha)')
plt.ylabel('Inverse temperature (beta)')
plt.title('Fitted parameters for all participants')

#Calculate the slope and intercept of the regression line

slope, intercept, r_value, p_value, std_err = linregress(fitted_params_after[:, 0], fitted_params_after[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
ax.legend([legend],loc="upper right", frameon=False)

#Add error bars for the SEM

x_errors = sem(fitted_params_after[:, 0])
y_errors = sem(fitted_params_after[:, 1])
plt.errorbar(fitted_params_after[:, 0], fitted_params_after[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()

#Plotting fitted parameters for high anxious group

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
sns.regplot(x=high_anxious[:, 0], y=high_anxious[:, 1], scatter_kws={'s': 50}, ax=ax[0])
ax[0].set_xlabel('Learning rate (alpha)')
ax[0].set_ylabel('Inverse temperature (beta)')
ax[0].set_title('Fitted parameters for Anxious Group')

#Calculate the slope and intercept of the regression line

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 0], high_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
ax[0].legend([legend],loc="upper right", frameon=False)

#Add error bars for the SEM

x_errors = sem(high_anxious[:, 0])
y_errors = sem(high_anxious[:, 1])
ax[0].errorbar(high_anxious[:, 0], high_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')



#Plotting fitted parameters for low anxious group
low_anxious=low_anxious_without_outlier

sns.regplot(x=low_anxious[:, 0], y=low_anxious[:, 1], scatter_kws={'s': 50}, ax=ax[1])
ax[1].set_xlabel('Learning rate (alpha)')
ax[1].set_ylabel('Inverse temperature (beta)')
ax[1].set_title('Fitted parameters for Calm Group')


#Calculate the slope and intercept of the regression line

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 0], low_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
ax[1].legend([legend],loc="upper right", frameon=False)

#Add error bars for the SEM

x_errors = sem(low_anxious[:, 0])
y_errors = sem(low_anxious[:, 1])
ax[1].errorbar(low_anxious[:, 0], low_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()
'''


# Model Two  #
##############

V0 = 0.5
starting_params = [0.4, 5,0.5]
fitted_params_model2 = fit_params_for_all_participants_model2_fitted(Behaviour_Data, V0, starting_params)
high_anxious = fitted_params_model2[:25, :]
low_anxious = fitted_params_model2[25:, :]

print("mean of fitted paremters:",np.mean(fitted_params_model2,axis=0))
print("varaince of fitted paremters:",np.var(fitted_params_model2,axis=0))

'''
# Fitted parameers before and after outlier

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

# Fitted Parameter Values before removing the outlier

mean = np.mean(fitted_params_model2, axis=0)
var = np.var(fitted_params_model2, axis=0)

ax[0].plot(fitted_params_model2[:, 0], 'o', label='Learning rate (alpha)')
ax[0].plot(fitted_params_model2[:, 1], 'o', label='Inverse temperature (beta)')
ax[0].plot(fitted_params_model2[:, 2], 'o', label='Aversaive Amplifier (A)')
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_xlabel('Participant Index')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Fitted parameter values (before outlier removal)')
ax[0].annotate(f'Mean: {np.round(mean,3)}', (0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[0].annotate(f'Variance: {np.round(var,3)}', (0.05, 0.85), xycoords='axes fraction', fontsize=9)




ax[1].plot(fitted_params_model2[:, 0], 'o', label='Learning rate (alpha)')
ax[1].plot(fitted_params_model2[:, 1], 'o', label='Inverse temperature (beta)')
ax[1].plot(fitted_params_model2[:, 2], 'o', label='Aversaive Amplifier (A)')
ax[1].legend(loc="upper right", frameon=False)
ax[1].set_xlabel('Participant Index')
ax[1].set_ylabel('Parameter Value')
ax[1].set_title('Fitted parameter values')
ax[1].annotate(f'Mean: {np.round(mean,3)}', (0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[1].annotate(f'Variance: {np.round(var,3)}', (0.05, 0.85), xycoords='axes fraction', fontsize=9)
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15, 10))

# Calm group before excluding outlier
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 0], ax=ax[0], color='blue', label='Learning rate (alpha)')
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 1], ax=ax[0], color='red', label='Inverse temperature (beta)')
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 2], ax=ax[0], color='green', label='Aversaive Amplifier (A)')
ax[0].set_xlabel('Participant')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Calm Group ')
ax[0].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(low_anxious[:, 0])
var_alpha = np.var(low_anxious[:, 0])
mean_beta = np.mean(low_anxious[:, 1])
var_beta = np.var(low_anxious[:, 1])
mean_A = np.mean(low_anxious[:, 2])
var_A = np.var(low_anxious[:, 2])
ax[0].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[0].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9)
ax[0].annotate(f'Mean A: {mean_A:.2f}\nVariance A: {var_A:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=9)


# Nxious group 
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 0], ax=ax[1], color='blue', label='Learning rate (alpha)')
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 1], ax=ax[1], color='red', label='Inverse temperature (beta)')
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 2], ax=ax[1], color='green', label='Aversaive Amplifier (A)')
ax[1].set_xlabel('Participant')
ax[1].set_ylabel('Parameter Value')
ax[1].set_title('Anxious Group ')
ax[1].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(high_anxious[:, 0])
var_alpha = np.var(high_anxious[:, 0])
mean_beta = np.mean(high_anxious[:, 1])
var_beta = np.var(high_anxious[:, 1])
mean_A = np.mean(high_anxious[:, 2])
var_A = np.var(high_anxious[:, 2])
ax[1].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[1].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9)
ax[1].annotate(f'Mean A: {mean_A:.2f}\nVariance A: {var_A:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=9)

plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

ax[0].plot(fitted_params_model2[:, 0], 'o', label='Learning rate (alpha)')
ax[0].plot(fitted_params_model2[:, 1], 'o', label='Inverse temperature (beta)')
ax[0].plot(fitted_params_model2[:, 2], 'o', label='A')
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_xlabel('Participant Index')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Fitted parameter values')

plt.tight_layout()
plt.show()

#correlation all parameters 
fig, axs = plt.subplots(1, 3,figsize=(15, 5))

# Plot the Alpha vs Beta 
sns.regplot(x=fitted_params_model2[:, 0], y=fitted_params_model2[:, 1], scatter_kws={'s': 50}, ax=axs[0])
axs[0].set_xlabel('Learning rate (alpha)')
axs[0].set_ylabel('Inverse temperature (beta)')
axs[0].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(fitted_params_model2[:, 0], fitted_params_model2[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[0].legend([legend], loc="upper right", frameon=False)

x_errors = sem(fitted_params_model2[:, 0])
y_errors = sem(fitted_params_model2[:, 1])
axs[0].errorbar(fitted_params_model2[:, 0], fitted_params_model2[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')


# Plot the xz side
sns.regplot(x=fitted_params_model2[:, 2], y=fitted_params_model2[:, 0], scatter_kws={'s': 50}, ax=axs[1])
axs[1].set_ylabel('Learning rate (alpha)')
axs[1].set_xlabel('Aversaive Amplifier (A)')
axs[1].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(fitted_params_model2[:, 2], fitted_params_model2[:, 0])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[1].legend([legend], loc="upper right", frameon=False)

x_errors = sem(fitted_params_model2[:, 2])
y_errors = sem(fitted_params_model2[:, 0])
axs[1].errorbar(fitted_params_model2[:, 2], fitted_params_model2[:, 0], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

# Plot the xz side
sns.regplot(x=fitted_params_model2[:, 2], y=fitted_params_model2[:, 1], scatter_kws={'s': 50}, ax=axs[2])
axs[2].set_ylabel('Inverse temperature (beta)')
axs[2].set_xlabel('Aversaive Amplifier (A)')
axs[2].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(fitted_params_model2[:, 2], fitted_params_model2[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[2].legend([legend], loc="upper right", frameon=False)

x_errors = sem(fitted_params_model2[:, 2])
y_errors = sem(fitted_params_model2[:, 1])
axs[2].errorbar(fitted_params_model2[:, 2], fitted_params_model2[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()


#plot correlation 
fig, axs = plt.subplots(1, 3,figsize=(15, 5))

# Plot the Alpha vs Beta 
sns.regplot(x=high_anxious[:, 0], y=high_anxious[:, 1], scatter_kws={'s': 50}, ax=axs[0])
axs[0].set_xlabel('Learning rate (alpha)')
axs[0].set_ylabel('Inverse temperature (beta)')
axs[0].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 0], high_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[0].legend([legend], loc="upper right", frameon=False)

x_errors = sem(high_anxious[:, 0])
y_errors = sem(high_anxious[:, 1])
axs[0].errorbar(high_anxious[:, 0], high_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')


# Plot the xz side
sns.regplot(x=high_anxious[:, 2], y=high_anxious[:, 0], scatter_kws={'s': 50}, ax=axs[1])
axs[1].set_ylabel('Learning rate (alpha)')
axs[1].set_xlabel('Aversaive Amplifier (A)')
axs[1].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 2], high_anxious[:, 0])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[1].legend([legend], loc="upper right", frameon=False)

x_errors = sem(high_anxious[:, 2])
y_errors = sem(high_anxious[:, 0])
axs[1].errorbar(high_anxious[:, 2], high_anxious[:, 0], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

# Plot A versus Beta 
sns.regplot(x=high_anxious[:, 2], y=high_anxious[:, 1], scatter_kws={'s': 50}, ax=axs[2])
axs[2].set_ylabel('Inverse temperature (beta)')
axs[2].set_xlabel('Aversaive Amplifier (A)')
axs[2].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 2], high_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[2].legend([legend], loc="upper right", frameon=False)

x_errors = sem(high_anxious[:, 2])
y_errors = sem(high_anxious[:, 1])
axs[2].errorbar(high_anxious[:, 2], high_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()




fig, axs = plt.subplots(1,3,figsize=(15, 10))

# Plot the Alpha vs Beta 
sns.regplot(x=low_anxious[:, 0], y=low_anxious[:, 1], scatter_kws={'s': 50}, ax=axs[0])
axs[0].set_xlabel('Learning rate (alpha)')
axs[0].set_ylabel('Inverse temperature (beta)')
axs[0].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 0], low_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[0].legend([legend], loc="upper right", frameon=False)

x_errors = sem(low_anxious[:, 0])
y_errors = sem(low_anxious[:, 1])
axs[0].errorbar(low_anxious[:, 0], low_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')


# Plot the alpha verses A
sns.regplot(x=low_anxious[:, 2], y=low_anxious[:, 0], scatter_kws={'s': 50}, ax=axs[1])
axs[1].set_ylabel('Learning rate (alpha)')
axs[1].set_xlabel('Aversaive Amplifier (A)')
axs[1].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 2], low_anxious[:, 0])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[1].legend([legend], loc="upper right", frameon=False)

x_errors = sem(low_anxious[:, 2])
y_errors = sem(low_anxious[:, 0])
axs[1].errorbar(low_anxious[:, 2], low_anxious[:, 0], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

# Plot Beta versus A 
sns.regplot(x=low_anxious[:, 2], y=low_anxious[:, 1], scatter_kws={'s': 50}, ax=axs[2])
axs[2].set_ylabel('Inverse temperature (beta)')
axs[2].set_xlabel('Aversaive Amplifier (A)')
axs[2].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 2], low_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[2].legend([legend], loc="upper right", frameon=False)

x_errors = sem(low_anxious[:, 2])
y_errors = sem(low_anxious[:, 1])
axs[2].errorbar(low_anxious[:, 2], low_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()
'''







# Model Three  #
################

V0 = 0.5
starting_params = [0.25, 0.5,6] #alpha_pos,alpha_neg,beta
fitted_params_model3 = fit_params_for_all_participants_model3_fitted(Behaviour_Data, V0, starting_params)
high_anxious = fitted_params_model3[:25, :]
low_anxious = fitted_params_model3[25:, :]

print("mean of fitted paremters:",np.mean(fitted_params_model3,axis=0))
print("varaince of fitted paremters:",np.var(fitted_params_model3,axis=0))

'''
# Fitted parameers before and after outlier

fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

# Fitted Parameter Values before removing the outlier

mean = np.mean(fitted_params_model3, axis=0)
var = np.var(fitted_params_model3, axis=0)

ax[0].plot(fitted_params_model3[:, 0], 'o', label='Learning rate (alpha+)')
ax[0].plot(fitted_params_model3[:, 1], 'o', label='Aversaive Learning rate (alpha-)')
ax[0].plot(fitted_params_model3[:, 2], 'o', label='Inverse temperature (beta)')
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_xlabel('Participant Index')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Fitted parameter values')
ax[0].annotate(f'Mean: {np.round(mean,3)}', (0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[0].annotate(f'Variance: {np.round(var,3)}', (0.05, 0.85), xycoords='axes fraction', fontsize=9)

plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15, 10))


sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 0], ax=ax[0], color='blue', label='Learning rate (alpha+)')
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 1], ax=ax[0], color='red', label='Aversaive Learning rate (alpha-)')
sns.scatterplot(x=np.arange(low_anxious.shape[0]), y=low_anxious[:, 2], ax=ax[0], color='green', label='Inverse temperature (beta)')
ax[0].set_xlabel('Participant')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Calm Group ')
ax[0].legend(loc="upper right", frameon=False)

mean_alpha = np.mean(low_anxious[:, 0])
var_alpha = np.var(low_anxious[:, 0])
mean_beta = np.mean(low_anxious[:, 1])
var_beta = np.var(low_anxious[:, 1])
mean_A = np.mean(low_anxious[:, 2])
var_A = np.var(low_anxious[:, 2])
ax[0].annotate(f'Mean Alpha: {mean_alpha:.2f}\nVariance Alpha: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[0].annotate(f'Mean Beta: {mean_beta:.2f}\nVariance Beta: {var_beta:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9)
ax[0].annotate(f'Mean A: {mean_A:.2f}\nVariance A: {var_A:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=9)



sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 0], ax=ax[1], color='blue', label='Learning rate (alpha+)')
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 1], ax=ax[1], color='red', label='Aversaive Learning rate (alpha-)')
sns.scatterplot(x=np.arange(high_anxious.shape[0]), y=high_anxious[:, 2], ax=ax[1], color='green', label='Inverse temperature (beta)')
ax[1].set_xlabel('Participant')
ax[1].set_ylabel('Parameter Value')
ax[1].set_title('Anxious Group ')
ax[1].legend(loc="upper right", frameon=False)

mean_alphapos = np.mean(high_anxious[:, 0])
var_alphapos = np.var(high_anxious[:, 0])
mean_alphaneg = np.mean(high_anxious[:, 1])
var_alphaneg= np.var(high_anxious[:, 1])
mean_beta = np.mean(high_anxious[:, 2])
var_beta = np.var(high_anxious[:, 2])
ax[1].annotate(f'Mean Alpha+: {mean_alpha:.2f}\nVariance Alpha+: {var_alpha:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=9)
ax[1].annotate(f'Mean Alpha-: {mean_beta:.2f}\nVariance Alpha-: {var_beta:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9)
ax[1].annotate(f'Mean A: {mean_A:.2f}\nVariance A: {var_A:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=9)

plt.show()




fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

ax[0].plot(fitted_params_model3[:, 0], 'o', label='Learning rate (alpha+)')
ax[0].plot(fitted_params_model3[:, 1], 'o', label='Aversaive Learning rate (alpha-)')
ax[0].plot(fitted_params_model3[:, 2], 'o', label='Inverse temperature (beta)')
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_xlabel('Participant Index')
ax[0].set_ylabel('Parameter Value')
ax[0].set_title('Fitted parameter values')

plt.tight_layout()
plt.show()




fig, axs = plt.subplots(1, 3,figsize=(15, 5))

# Plot the Alpha vs Beta 
sns.regplot(x=high_anxious[:, 0], y=high_anxious[:, 1], scatter_kws={'s': 50}, ax=axs[0])
axs[0].set_xlabel('Learning rate (alpha+)')
axs[0].set_ylabel('Aversaive Learning rate (alpha-)')
axs[0].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 0], high_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[0].legend([legend], loc="upper right", frameon=False)

x_errors = sem(high_anxious[:, 0])
y_errors = sem(high_anxious[:, 1])
axs[0].errorbar(high_anxious[:, 0], high_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')


# Plot the xz side
sns.regplot(x=high_anxious[:, 0], y=high_anxious[:, 2], scatter_kws={'s': 50}, ax=axs[1])
axs[1].set_xlabel('Learning rate (alpha+)')
axs[1].set_ylabel('Inverse temperature (beta)')
axs[1].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 0], high_anxious[:, 2])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[1].legend([legend], loc="upper right", frameon=False)

x_errors = sem(high_anxious[:, 0])
y_errors = sem(high_anxious[:, 2])
axs[1].errorbar(high_anxious[:, 0], high_anxious[:, 2], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

# Plot the xz side
sns.regplot(x=high_anxious[:, 1], y=high_anxious[:, 2], scatter_kws={'s': 50}, ax=axs[2])
axs[2].set_xlabel('Aversaive Learning rate (alpha-)')
axs[2].set_ylabel('Inverse temperature (beta)')
axs[2].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(high_anxious[:, 1], high_anxious[:, 2])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[2].legend([legend], loc="upper right", frameon=False)

x_errors = sem(high_anxious[:, 1])
y_errors = sem(high_anxious[:, 2])
axs[2].errorbar(high_anxious[:, 1], high_anxious[:, 2], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()




fig, axs = plt.subplots(1, 3,figsize=(15, 5))

# Plot the Alpha vs Beta 
sns.regplot(x=low_anxious[:, 0], y=low_anxious[:, 1], scatter_kws={'s': 50}, ax=axs[0])
axs[0].set_xlabel('Learning rate (alpha+)')
axs[0].set_ylabel('Aversaive Learning rate (alpha-)')
axs[0].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 0], low_anxious[:, 1])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[0].legend([legend], loc="upper right", frameon=False)

x_errors = sem(low_anxious[:, 0])
y_errors = sem(low_anxious[:, 1])
axs[0].errorbar(low_anxious[:, 0], low_anxious[:, 1], xerr=x_errors, yerr=y_errors, fmt='none', c='k')


# Plot the alpha verses A
sns.regplot(x=low_anxious[:, 0], y=low_anxious[:, 2], scatter_kws={'s': 50}, ax=axs[1])
axs[1].set_xlabel('Learning rate (alpha+)')
axs[1].set_ylabel('Inverse temperature (beta)')
axs[1].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 0], low_anxious[:, 2])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[1].legend([legend], loc="upper right", frameon=False)

x_errors = sem(low_anxious[:, 0])
y_errors = sem(low_anxious[:, 2])
axs[1].errorbar(low_anxious[:, 0], low_anxious[:, 2], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

# Plot Beta versus A 
sns.regplot(x=low_anxious[:, 1], y=low_anxious[:, 2], scatter_kws={'s': 50}, ax=axs[2])
axs[2].set_xlabel('Aversaive Learning rate (alpha-)')
axs[2].set_ylabel('Inverse temperature (beta)')
axs[2].set_title('Fitted Parameters Correlation')

slope, intercept, r_value, p_value, std_err = linregress(low_anxious[:, 1], low_anxious[:, 2])
legend = f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, Pearson: {r_value:.2f}"
axs[2].legend([legend], loc="upper right", frameon=False)

x_errors = sem(low_anxious[:, 1])
y_errors = sem(low_anxious[:, 2])
axs[2].errorbar(low_anxious[:, 1], low_anxious[:, 2], xerr=x_errors, yerr=y_errors, fmt='none', c='k')

plt.show()
'''













'''
#####################
# Group comparision #
#####################

# Model One #
#############

alpha_high = fitted_params_after[:25, 0]
alpha_low = fitted_params_after[25:, 0]
beta_high = fitted_params_after[:25, 1]
beta_low = fitted_params_after[25:, 1]

t_stat_alpha, p_value_alpha = stats.ttest_ind(alpha_high, alpha_low)
t_stat_beta, p_value_beta = stats.ttest_ind(beta_high, beta_low)

df = len(fitted_params) - 2

print("T-statistic for alpha: ", t_stat_alpha)
print("P-value for alpha: ", p_value_alpha)
print("Degrees of freedom for alpha: ", df)
print("\nT-statistic for beta: ", t_stat_beta)
print("P-value for beta: ", p_value_beta)
print("Degrees of freedom for beta: ", df)

# Model Two #
#############

alpha_high = fitted_params_model2[:25, 0]
alpha_low = fitted_params_model2[25:, 0]
beta_high = fitted_params_model2[:25, 1]
beta_low = fitted_params_model2[25:, 1]
A_high= fitted_params_model2[:25, 2]
A_low= fitted_params_model2[25:, 2]

t_stat_alpha, p_value_alpha = stats.ttest_ind(alpha_high, alpha_low)
t_stat_beta, p_value_beta = stats.ttest_ind(beta_high, beta_low)
t_stat_A, p_value_A = stats.ttest_ind(A_high, A_low)
df = len(fitted_params_model2) - 2

print("T-statistic for alpha: ", t_stat_alpha)
print("P-value for alpha: ", p_value_alpha)
print("Degrees of freedom for alpha: ", df)
print("\nT-statistic for beta: ", t_stat_beta)
print("P-value for beta: ", p_value_beta)
print("Degrees of freedom for beta: ", df)
print("\nT-statistic for A: ", t_stat_A)
print("P-value for A: ", p_value_A)
print("Degrees of freedom for A: ", df)



# Model Three #
###############

alpha_pos_high = fitted_params_model3[:25, 0]
alpha_pos_low = fitted_params_model3[25:, 0]
alpha_neg_high = fitted_params_model3[:25, 1]
alpha_neg_low  = fitted_params_model3[25:, 1]
beta_high= fitted_params_model3[:25, 2]
beta_low= fitted_params_model3[25:, 2]

t_stat_alphapos, p_value_alphapos = stats.ttest_ind(alpha_pos_high, alpha_pos_low)
t_stat_alphaneg, p_value_alphaneg = stats.ttest_ind(alpha_neg_high, alpha_neg_low)
t_stat_beta, p_value_beta = stats.ttest_ind(beta_high, beta_low)
df = len(fitted_params_model3) - 2

print("T-statistic for alpha+: ", t_stat_alphapos)
print("P-value for alpha+: ", p_value_alphapos)
print("Degrees of freedom for alpha+: ", df)
print("\nT-statistic for alpha-: ", t_stat_alphaneg)
print("P-value for alpha-: ", p_value_alphaneg)
print("Degrees of freedom for alpha-: ", df)
print("\nT-statistic for beta: ", t_stat_beta)
print("P-value for beta: ", p_value_beta)
print("Degrees of freedom for beta: ", df)
'''












ddd


###################################
# NLL Fitting to Bheaviour Data   #
###################################

# NLL Model One #
#################
alpha = 0.4
beta = 7
V0 = 0.5
NLLs = calculate_NLL_for_all_participants_MODEL1(Behaviour_Data, alpha, beta, V0)
Behaviour_Data["NLL_Model1"]=NLLs
participants = [i for i in range(1, len(NLLs)+1)]
plot_NLL(NLLs, participants)

#Fit to  model 
parameters=[0.4,7]
V0=0.5
NLLs=[]

for i,row in Behaviour_Data.iterrows():
        alpha,beta=fitted_params[i]
        NLLS=calculate_NLL_for_participant_model1_fitting(i,Behaviour_Data, alpha, beta, V0)
        NLLs.append(NLLS)

Behaviour_Data["NLL_Model1"]=NLLs


#NLL Model Two #
################
alpha = 0.4
beta = 5
V0 = 0.5
A=0.5
NLLs = calculate_NLL_for_all_participants_model2(Behaviour_Data, alpha, beta,A, V0)
Behaviour_Data["NLL_Model2"]=NLLs
participants = [i for i in range(1, len(NLLs)+1)]
plot_NLL_model2(NLLs, participants)

#Fit to  model 
parameters=[0.5,7,0.5]
V0=0.5
NLLs=[]
for i,row in Behaviour_Data.iterrows():
    alpha,beta,A=fitted_params_model2[i]
    NLLS=calculate_NLL_for_participant_model2_fitted(i,Behaviour_Data, alpha, beta,A, V0)
    NLLs.append(NLLS)

Behaviour_Data["NLL_Model2"]=NLLs


# NLL Model Three #
###################


alpha_pos = 0.25
alpha_neg=0.5
beta = 6
V0 = 0.5
NLLs = calculate_NLL_for_all_participants_model3(Behaviour_Data, alpha_pos, beta,alpha_neg, V0)
Behaviour_Data["NLL_Model3"]=NLLs
participants = [i for i in range(1, len(NLLs)+1)]
plot_NLL_model3(NLLs, participants)

#Fit to  model 
parameters=[0.25,0.5,6]
V0=0.5
NLLs=[]
for i,row in Behaviour_Data.iterrows():
    alpha_pos,alpha_neg,beta=fitted_params_model3[i]
    NLLS=calculate_NLL_for_participant_model3_fitted(i,Behaviour_Data, alpha_pos, alpha_neg,beta, V0)
    NLLs.append(NLLS)

Behaviour_Data["NLL_Model3"]=NLLs



















######################
# Paremters Settings #
######################
def plot_outcomes(outcomes, title):
    outcomes_avg = np.mean(outcomes, axis=1)
    trials = np.arange(len(outcomes_avg))
    plt.plot(trials, outcomes_avg)
    plt.title(title)
    plt.xlabel('Trials')
    plt.ylabel('Average Outcome (Aversive Sound)')
    plt.show()


def plot_V_A_V_B(V_A_sims, V_B_sims, title):
    V_A_avg = np.mean(V_A_sims, axis=0)
    V_B_avg = np.mean(V_B_sims, axis=0)
    trials = np.arange(len(V_A_avg))
    plt.plot(trials, V_A_avg, label='V(A)')
    plt.plot(trials, V_B_avg, label='V(B)')
    plt.title(title)
    plt.xlabel('Trials')
    plt.ylabel('Average Value')
    plt.legend(loc="upper right", frameon=False)
    plt.show()


# Model One  #
##############

'''
# Plotting VA and V B  for diff alpha 
# Constants
V0 = 0.5
num_sims = 1000
num_trials=160
alphas = [0.1, 0.4, 0.9]
betas = [5]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for j, alpha in enumerate(alphas):
        V_A_sims, V_B_sims, _, _, _ = simulate(alpha, betas[0], V0, num_trials, num_sims)
        axs[0].plot(np.mean(V_A_sims, axis=1), label=f'Alpha = {alpha}')
        axs[1].plot(np.mean(V_B_sims, axis=1), label=f'Alpha = {alpha}')

axs[0].set_xlabel('Trials')
axs[0].set_ylabel('V(A)')
axs[0].legend(loc="upper right", frameon=False)

axs[1].set_xlabel('Trials')
axs[1].set_ylabel('V(B)')
axs[1].legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()


# Plotting VA and V B  for diff beta 
# Constants
V0 = 0.5
num_sims = 1000
num_trials=160
alphas = [0.4]
betas = [1,5,9]

fig, axs = plt.subplots(1, 2, figsize=(15, 15))
for j, beta in enumerate(betas):
        V_A_sims, V_B_sims, _, _, _ = simulate(alphas[0], beta, V0, num_trials, num_sims)
        axs[0].plot(np.mean(V_A_sims, axis=1), label=f'Beta = {beta}')
        axs[1].plot(np.mean(V_B_sims, axis=1), label=f'Beta = {beta}')

axs[0].set_xlabel('Trials')
axs[0].set_ylabel('V(A)')
axs[0].legend(loc="upper right", frameon=False)

axs[1].set_xlabel('Trials')
axs[1].set_ylabel('V(B)')
axs[1].legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()




#VA-VB

# Constants
V0 = 0.5
num_sims = 1000
num_trials=160
alphas = [0.1, 0.4, 0.9]
betas = [1, 5, 9]

# Calculate the outcomes for different alpha and beta values
Vs_alpha_beta_varied = []
for alpha in alphas:
    Vs_beta_varied = []
    for beta in betas:
        _, _, diff, _, _ = simulate(alpha, beta, V0, num_trials, num_sims)
        Vs_beta_varied.append(np.mean(diff, axis=1))
    Vs_alpha_beta_varied.append(Vs_beta_varied)

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# Plot the outcomes for different alpha values with beta held constant
for i in range(len(alphas)):
    ax1.plot(Vs_alpha_beta_varied[i][1], label=f'Alpha = {alphas[i]}')
ax1.set_title('Value with Beta = 5')
ax1.set_xlabel('Trials')
ax1.set_ylabel('V(A)-V(B)')
ax1.legend(loc="upper right", frameon=False)

# Plot the outcomes for different beta values with alpha held constant
for i in range(len(betas)):
    ax2.plot(Vs_alpha_beta_varied[1][i], label=f'Beta = {betas[i]}')
ax2.set_title('Value with Alpha = 0.4')
ax2.set_xlabel('Trials')
ax2.set_ylabel('V(A)-V(B)')
ax2.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()


# outcomes 

# Constants
V0 = 0.5
num_sims = 1000
num_trials=160
alphas = [0.1, 0.4, 0.9]
betas = [1, 5, 9]

# Calculate the outcomes for different alpha and beta values
outcomes_alpha_beta_varied = []
for alpha in alphas:
    outcomes_beta_varied = []
    for beta in betas:
        _, _, _, _, outcomes = simulate(alpha, beta, V0, num_trials, num_sims)
        outcomes_beta_varied.append(np.mean(outcomes, axis=1)*160)
    outcomes_alpha_beta_varied.append(outcomes_beta_varied)

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# Plot the outcomes for different alpha values with beta held constant
for i in range(len(alphas)):
    ax1.plot(outcomes_alpha_beta_varied[i][1], label=f'Alpha = {alphas[i]}')
ax1.set_title('Outcome with Beta = 5')
ax1.set_xlabel('Trials')
ax1.set_ylabel('Average Outcome (Aversive Sound)')
ax1.legend(loc="upper right", frameon=False)

# Plot the outcomes for different beta values with alpha held constant
for i in range(len(betas)):
    ax2.plot(outcomes_alpha_beta_varied[1][i], label=f'Beta = {betas[i]}')
ax2.set_title('Outcome with Alpha = 0.4')
ax2.set_xlabel('Trials')
ax2.set_ylabel('Average Outcome (Aversive Sound)')
ax2.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()


'''
'''

# Model Two #
#############

# Constants
V0 = 0.5
num_sims = 1000
num_trials=160

#Plotting VA and V B  for diff A 
alphas = [0.4]
betas = [5]
As=[0,0.1,0.3,0.7,1]
V_A_sims_A_varied = []
V_B_sims_A_varied = []


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for j, A in enumerate(As):
        V_A_sims, V_B_sims, _, _, _ = simulate_model2(alphas[0], betas[0], V0,A, num_trials, num_sims)
        axs[0].plot(np.mean(V_A_sims, axis=1), label=f'A = {A}')
        axs[1].plot(np.mean(V_B_sims, axis=1), label=f'A = {A}')

axs[0].set_xlabel('Trials')
axs[0].set_ylabel('V(A)')
axs[0].legend(loc="upper right", frameon=False)

axs[1].set_xlabel('Trials')
axs[1].set_ylabel('V(B)')
axs[1].legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()


#Plotting V(A)-V(B)
# Calculate the outcomes for different alpha and beta values
Vs_alpha_beta_varied = []
for alpha in alphas:
    Vs_beta_varied = []
    for beta in betas:
        _, _, diff, _, _ = simulate(alpha, beta, V0, num_trials, num_sims)
        Vs_beta_varied.append(np.mean(diff, axis=1))
    Vs_alpha_beta_varied.append(Vs_beta_varied)

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# Plot the outcomes for different alpha values with beta held constant
for i in range(len(alphas)):
    ax1.plot(Vs_alpha_beta_varied[i][1], label=f'Alpha = {alphas[i]}')
ax1.set_title('Value with Beta = 5')
ax1.set_xlabel('Trials')
ax1.set_ylabel('V(A)-V(B)')
ax1.legend(loc="upper right", frameon=False)

# Plot the outcomes for different beta values with alpha held constant
for i in range(len(betas)):
    ax2.plot(Vs_alpha_beta_varied[1][i], label=f'Beta = {betas[i]}')
ax2.set_title('Value with Alpha = 0.4')
ax2.set_xlabel('Trials')
ax2.set_ylabel('V(A)-V(B)')
ax2.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()

# A varied 
alphas = [0.4]
betas = [5]
As=[0,0.1,0.3,0.7,1]

fig, axs = plt.subplots(1, 1, figsize=(10, 5))
for A in As:
    _, _, _, _, outcomes= simulate_model2(alphas[0], betas[0], A,V0, num_trials, num_sims)
    axs.plot(np.mean(outcomes,axis=1)*160, label=f'A = {A}')

axs.set_xlabel('Trials')
axs.set_ylabel('Average Outcome (Aversive Sound)')
axs.legend(loc="upper right", frameon=False)
plt.tight_layout()
plt.show()
'''


#  Model Three  #                  
#################
'''
# Plotting VA and V B  for diff alpha pos  
# Constants
V0 = 0.5
num_sims = 1000
num_trials=160
alphas_pos = [0.1, 0.25, 0.9]
alphas_neg = [0.1,0.5,0.9]
betas = [6]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for j, alpha in enumerate(alphas_pos):
        V_A_sims, V_B_sims, _, _, _ = simulate_model3(alpha, alphas_neg[1], betas[0], V0, num_trials, num_sims)
        axs[0].plot(np.mean(V_A_sims, axis=1), label=f'Alpha + = {alpha}')
        axs[1].plot(np.mean(V_B_sims, axis=1), label=f'Alpha + = {alpha}')

for j, alpha in enumerate(alphas_neg):
        V_A_sims, V_B_sims, _, _, _ = simulate_model3(alphas_pos[1], alpha, betas[0], V0, num_trials, num_sims)
        axs[0].plot(np.mean(V_A_sims, axis=1), label=f'Alpha - = {alpha}')
        axs[1].plot(np.mean(V_B_sims, axis=1), label=f'Alpha - = {alpha}')

axs[0].set_xlabel('Trials')
axs[0].set_ylabel('V(A)')
axs[0].legend(loc="upper right", frameon=False)

axs[1].set_xlabel('Trials')
axs[1].set_ylabel('V(B)')
axs[1].legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()
'''



'''
#V(A)-V(B)
V0 = 0.5
num_sims = 1000
num_trials=160
alphas_pos = [0.1, 0.25, 0.9]
alphas_neg = [0.1,0.5,0.9]
betas = [6]

Vs_alpha_pos_varied = []
for alphap in alphas_pos:
    Vs_alpha_neg_varied = []
    for alphan in alphas_neg:
        _, _, diff, _, _ = simulate_model3(alphap, alphan, betas[0], V0, num_trials, num_sims)
        Vs_alpha_neg_varied.append(np.mean(diff, axis=1))
    Vs_alpha_pos_varied.append(Vs_alpha_neg_varied)

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# Plot the outcomes for different alpha values with beta held constant
for i in range(len(alphas_pos)):
    ax1.plot(Vs_alpha_pos_varied[i][1], label=f'Alpha + = {alphas_pos[i]}')
ax1.set_title('Value with Alpha- = 0.5 and Beta = 6')
ax1.set_xlabel('Trials')
ax1.set_ylabel('V(A)-V(B)')
ax1.legend(loc="upper right", frameon=False)

# Plot the outcomes for different beta values with alpha held constant
for i in range(len(alphas_neg)):
    ax2.plot(Vs_alpha_pos_varied[1][i], label=f'Alpha - = {alphas_neg[i]}')
ax2.set_title('Value with Alpha+ = 0.25 and Beta = 6')
ax2.set_xlabel('Trials')
ax2.set_ylabel('V(A)-V(B)')
ax2.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()
'''

'''
#Outcomes

V0 = 0.5
num_sims = 1000
num_trials=160
alphas_pos = [0.1, 0.25, 0.9]
alphas_neg = [0.1,0.5,0.9]
betas = [6]

Vs_alpha_pos_varied = []
for alphap in alphas_pos:
    Vs_alpha_neg_varied = []
    for alphan in alphas_neg:
        _, _, _, _, outcomes = simulate_model3(alphap, alphan, betas[0], V0, num_trials, num_sims)
        Vs_alpha_neg_varied.append(np.mean(outcomes,axis=1)*160)
    Vs_alpha_pos_varied.append(Vs_alpha_neg_varied)

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# Plot the outcomes for different alpha values with beta held constant
for i in range(len(alphas_pos)):
    ax1.plot(Vs_alpha_pos_varied[i][1], label=f'Alpha + = {alphas_pos[i]}')
ax1.set_title('Value with Alpha- = 0.5 and Beta = 6')
ax1.set_xlabel('Trials')
ax1.set_ylabel('Average Outcome (Aversive Sound)')
ax1.legend(loc="upper right", frameon=False)

# Plot the outcomes for different beta values with alpha held constant
for i in range(len(alphas_neg)):
    ax2.plot(Vs_alpha_pos_varied[1][i], label=f'Alpha - = {alphas_neg[i]}')
ax2.set_title('Value with Alpha+ = 0.25 and Beta = 6')
ax2.set_xlabel('Trials')
ax2.set_ylabel('Average Outcome (Aversive Sound)')
ax2.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()
'''

#######################
# Parameters Recovery #
#######################
'''
# Model One #
#############

mean = [0.474, 5.163]
variance = [0.0286, 0.0500]
n_samples = 50
n_trials = 160
v0 = 0.5
n_simulations = 1

# Original parameter values from data
original_params = [0.4682618, 5.08712691]

params = simulate_parameters_model1(mean, variance, n_samples)

# Plot alpha and beta in different plots
sns.distplot(params[:, 0], hist=True, rug=False, label='alpha')
sns.distplot(params[:, 1], hist=True, rug=False, label='beta')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title("Distribution of Sampled Alpha's and Beta's")
plt.legend(loc="upper right", frameon=False)
xticks = np.arange(0, 7, 0.5)
plt.xticks(xticks, xticks)

plt.show()


correlation_coeffs_alpha = []
correlation_coeffs_beta = []
fitted_params_all = []
simulated_params_all=[] 
for i in range(5):
    simulated_choices = []
    simulated_outcomes = []
    simulated_params=simulate_parameters_model1(mean, variance, n_samples)
    simulated_params_all.append(simulated_params)
    for j in range(n_samples):
        choices, outcomes = simulate_data_model1(simulated_params[j],v0, n_trials, 1)
        simulated_choices.append(choices)
        simulated_outcomes.append(outcomes)
    fitted_paramstemp=[]
    for k in range(n_samples):
        fitted_params_temp=fit_params_model1(simulated_choices[k], simulated_outcomes[k], mean[0], mean[1], v0)
        fitted_paramstemp.append(fitted_params_temp)

    fitted_params = np.array(fitted_paramstemp)
    fitted_params_all.append(fitted_params)




for i, arr in enumerate(fitted_params_all):
    arr[:, 1] = (arr[:, 1] - np.min(arr[:, 1])) / (np.max(arr[:, 1]) - np.min(arr[:, 1])) * (25 - 4) + 4
    fitted_params_all[i] = arr

alphas_fitted = []
betas_fitted= []

for array in fitted_params_all:
    inner_alphas=[]
    inner_beta=[]
    for inner_array in array:
        inner_alphas.append(inner_array[0])
        inner_beta.append(inner_array[1])
    alphas_fitted.append(inner_alphas)
    betas_fitted.append(inner_beta)



alphas_sim = []
betas_sim= []
for array in simulated_params_all:
    inner_alphas=[]
    inner_beta=[]
    for inner_array in array:
        inner_alphas.append(inner_array[0])
        inner_beta.append(inner_array[1])
    alphas_sim.append(inner_alphas)
    betas_sim.append(inner_beta)




for sublist1, sublist2 in zip(alphas_sim, alphas_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_alpha.append(r)

for sublist1, sublist2 in zip(betas_sim, betas_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_beta.append(r)


print(correlation_coeffs_beta)



fig, ax = plt.subplots(figsize=(10,10))

data=[correlation_coeffs_alpha,correlation_coeffs_beta]
sns.boxplot(data=data, ax=ax)

ax.set_xticklabels(['Alpha', 'Beta'])
ax.set_ylabel('Correlation')
ax.legend(['Alpha', 'Beta'],loc="upper right", frameon=False)
plt.show()




def ganarate_parameters( v0, n_trials, n_simulated_data):
    fitted_data_params = []
    simualated_data_params=[]
    for i in range(n_simulated_data):
        simulated_params = np.random.multivariate_normal(mean, np.diag(variance))
        simulated_choices, simulated_outcomes = simulate_data_model1(simulated_params, v0, n_trials,1)
        fitted_params = fit_params_model1(simulated_choices, simulated_outcomes,original_params[0],original_params[1], v0)
        simualated_data_params.append(simulated_params)
        fitted_data_params.append(fitted_params)
    return simualated_data_params,fitted_data_params

def plot_correlation(sampled_params, fitted_param, n, m,str):
    correlaiton_alpha = []
    correlation_beta=[]
    if str == "m":
        indices_to_remove = [i for i, x in enumerate(fitted_param) if x[1] > 100]
        fitted_param = [x for i, x in enumerate(fitted_param) if i not in indices_to_remove]
        sampled_params = [x for i, x in enumerate(sampled_params) if i not in indices_to_remove]


        alpha_sampled = [subarray[0] for subarray in sampled_params]
        alpha_fitted = [subarray[0] for subarray in fitted_param]
        beta_sampled = [subarray[1] for subarray in sampled_params]
        beta_fitted = [subarray[1] for subarray in fitted_param]           
        corr_a  = pearsonr(alpha_sampled,alpha_fitted)
        corr_b  = pearsonr(beta_sampled,beta_fitted)
        correlation_beta.append(corr_b[0])
        correlaiton_alpha.append(corr_a[0])

    elif str == "n":
        indices_to_remove = [i for i, x in enumerate(fitted_param) if x[1] > 100]
        fitted_param = [x for i, x in enumerate(fitted_param) if i not in indices_to_remove]
        sampled_params = [x for i, x in enumerate(sampled_params) if i not in indices_to_remove]

        alpha_sampled = [subarray[0] for subarray in sampled_params]
        alpha_fitted = [subarray[0] for subarray in fitted_param]
        beta_sampled = [subarray[1] for subarray in sampled_params]
        beta_fitted = [subarray[1] for subarray in fitted_param]           
        corr_a  = pearsonr(alpha_sampled,alpha_fitted)
        corr_b  = pearsonr(beta_sampled,beta_fitted)

        correlation_beta.append(corr_b[0])
        correlaiton_alpha.append(corr_a[0])  


    return correlaiton_alpha,correlation_beta


sns.set(style="whitegrid")
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

# Test the effect of n trials
n_simulated_data = 50
n_trials = [10, 50,100,250,500,1000]
corr_a=[]
corr_b=[]
for n in n_trials:
    simualated_data_params,fitted_data_params = ganarate_parameters(V0, n, n_simulated_data)
    correlaiton_alpha,correlation_beta=plot_correlation(simualated_data_params, fitted_data_params, n, n_simulated_data,"n")
    corr_a.append(correlaiton_alpha)
    corr_b.append(correlation_beta)

corr_a=np.array(corr_a).flatten()
corr_b=np.array(corr_b).flatten()
corr_a=pd.DataFrame({"Pearson Correlation":corr_a,
                     "Trial":n_trials})
corr_b=pd.DataFrame({"Pearson Correlation":corr_b,
                     "Trial":n_trials})


sns.lineplot(x="Trial",y="Pearson Correlation",data=corr_a, label="Alpha",ax=ax1)
sns.lineplot(x="Trial",y="Pearson Correlation", data=corr_b,label="Beta",ax=ax1)
ax1.set_xlabel('Number of trials')
ax1.set_ylabel('Pearson\'s correlation')
ax1.set_title(f'Effect of number of trials on Correlation , Number of Simulations ={n_simulated_data}')
ax1.legend(loc="upper right", frameon=False)

# Test the effect of n simulated data
n_trials = 100
n_simulated_data = [10, 50, 100,250,500,1000]
corr_a=[]
corr_b=[]
for m in n_simulated_data:
    simualated_data_params,fitted_data_params = ganarate_parameters(V0, n_trials, m)
    correlaiton_alpha,correlation_beta=plot_correlation(simualated_data_params, fitted_data_params, n_trials, m,"m")
    corr_a.append(correlaiton_alpha)
    corr_b.append(correlation_beta)

#fix plotting turn into fatafrmae 
corr_a=np.array(corr_a).flatten()
corr_b=np.array(corr_b).flatten()
corr_a=pd.DataFrame({"PearsonCorrelation":corr_a,
                     "NSimul":n_simulated_data})
corr_b=pd.DataFrame({"PearsonCorrelation":corr_b,
                     "NSimul":n_simulated_data})
sns.lineplot(x="NSimul",y="PearsonCorrelation",data=corr_a, label="Alpha",ax=ax2)
sns.lineplot(x="NSimul",y="PearsonCorrelation", data=corr_b,label="Beta",ax=ax2)
ax2.set_xlabel('Number of Simulations')
ax2.set_ylabel('Pearson\'s correlation')
ax2.set_title(f'Effect of number of simulated data on Correlation , Number of trials={n_trials}')
ax2.legend(loc="upper right", frameon=False)

plt.show()
'''

'''
# Model Two #
#############

mean = [0.657, 4.756,1.197]
variance = [0.0101,0.711, 0.108]
n_samples = 50
n_trials = 160
v0 = 0.5
n_simulations = 1

# Original parameter values from data
original_params = [0.657, 4.756,1.197]

params = simulate_parameters_model2(mean, variance, n_samples)

# Plot alpha and beta in different plots
sns.distplot(params[:, 0], hist=True, rug=False, label='alpha')
sns.distplot(params[:, 1], hist=True, rug=False, label='beta')
sns.distplot(params[:, 2], hist=True, rug=False, label='beta')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title("Distribution of Sampled Alpha's, Beta's and A's")
plt.legend(loc="upper right", frameon=False)
xticks = np.arange(0, 7.5, 0.5)
plt.xticks(xticks, xticks)

plt.show()


correlation_coeffs_alpha = []
correlation_coeffs_beta = []
correlation_coeffs_A = []

fitted_params_all = []
simulated_params_all=[] 

for i in range(5):
    simulated_choices = []
    simulated_outcomes = []
    simulated_params=simulate_parameters_model2(mean, variance, n_samples)
    simulated_params_all.append(simulated_params)
    for j in range(n_samples):
        choices, outcomes = simulate_data_model2(simulated_params[j],v0, n_trials, 1)
        simulated_choices.append(choices)
        simulated_outcomes.append(outcomes)
    fitted_paramstemp=[]
    for k in range(n_samples):
        fitted_params_temp=fit_params_model2(simulated_choices[k], simulated_outcomes[k], mean[0], mean[1],mean[2], v0)
        fitted_paramstemp.append(fitted_params_temp)

    fitted_params = np.array(fitted_paramstemp)
    fitted_params_all.append(fitted_params)




for i, arr in enumerate(fitted_params_all):
    arr[:, 1] = (arr[:, 1] - np.min(arr[:, 1])) / (np.max(arr[:, 1]) - np.min(arr[:, 1])) * (25 - 4) + 4
    fitted_params_all[i] = arr

alphas_fitted = []
betas_fitted= []
As_fitted=[]

for array in fitted_params_all:
    inner_alphas=[]
    inner_beta=[]
    inner_A=[]
    for inner_array in array:
        inner_alphas.append(inner_array[0])
        inner_beta.append(inner_array[1])
        inner_A.append(inner_array[2])
    alphas_fitted.append(inner_alphas)
    betas_fitted.append(inner_beta)
    As_fitted.append(inner_A)



alphas_sim = []
betas_sim= []
As_sim=[]
for array in simulated_params_all:
    inner_alphas=[]
    inner_beta=[]
    inner_A=[]    
    for inner_array in array:
        inner_alphas.append(inner_array[0])
        inner_beta.append(inner_array[1])
        inner_A.append(inner_array[2])       
    alphas_sim.append(inner_alphas)
    betas_sim.append(inner_beta)
    As_sim.append(inner_A)




for sublist1, sublist2 in zip(alphas_sim, alphas_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_alpha.append(r)

for sublist1, sublist2 in zip(betas_sim, betas_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_beta.append(r)

for sublist1, sublist2 in zip(As_sim, As_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_A.append(r)





fig, ax = plt.subplots(figsize=(10,10))

data=[correlation_coeffs_alpha,correlation_coeffs_beta,correlation_coeffs_A]
sns.boxplot(data=data, ax=ax)

ax.set_xticklabels(['Alpha', 'Beta','A'])
ax.set_ylabel('Correlation')
ax.legend(['Alpha', 'Beta','A'],loc="upper right", frameon=False)
plt.show()


def ganarate_parameters( v0, n_trials, n_simulated_data):
    fitted_data_params = []
    simualated_data_params=[]
    for i in range(n_simulated_data):
        simulated_params = np.random.multivariate_normal(mean, np.diag(variance))
        simulated_choices, simulated_outcomes = simulate_data_model2(simulated_params, v0, n_trials,1)
        fitted_params = fit_params_model2(simulated_choices, simulated_outcomes,original_params[0],original_params[1],original_params[2], v0)
        simualated_data_params.append(simulated_params)
        fitted_data_params.append(fitted_params)
    return simualated_data_params,fitted_data_params

def plot_correlation(sampled_params, fitted_param, n, m,str):
    correlaiton_alpha = []
    correlation_beta=[]
    correlation_A=[]
    if str == "m":
        indices_to_remove = [i for i, x in enumerate(fitted_param) if x[1] > 100]
        fitted_param = [x for i, x in enumerate(fitted_param) if i not in indices_to_remove]
        sampled_params = [x for i, x in enumerate(sampled_params) if i not in indices_to_remove]


        alpha_sampled = [subarray[0] for subarray in sampled_params]
        alpha_fitted = [subarray[0] for subarray in fitted_param]
        beta_sampled = [subarray[1] for subarray in sampled_params]
        beta_fitted = [subarray[1] for subarray in fitted_param]       
        A_sampled  = [subarray[2] for subarray in sampled_params]
        A_fitted =  [subarray[2] for subarray in fitted_param] 
        corr_a  = pearsonr(alpha_sampled,alpha_fitted)
        corr_b  = pearsonr(beta_sampled,beta_fitted)
        corr_A = pearsonr(A_sampled,A_fitted)
        correlation_beta.append(corr_b[0])
        correlaiton_alpha.append(corr_a[0])
        correlation_A.append(corr_A[0])

    elif str == "n":
        indices_to_remove = [i for i, x in enumerate(fitted_param) if x[1] > 100]
        fitted_param = [x for i, x in enumerate(fitted_param) if i not in indices_to_remove]
        sampled_params = [x for i, x in enumerate(sampled_params) if i not in indices_to_remove]

        alpha_sampled = [subarray[0] for subarray in sampled_params]
        alpha_fitted = [subarray[0] for subarray in fitted_param]
        beta_sampled = [subarray[1] for subarray in sampled_params]
        beta_fitted = [subarray[1] for subarray in fitted_param]           
        A_sampled  = [subarray[2] for subarray in sampled_params]
        A_fitted =  [subarray[2] for subarray in fitted_param] 
        corr_a  = pearsonr(alpha_sampled,alpha_fitted)
        corr_b  = pearsonr(beta_sampled,beta_fitted)
        corr_A = pearsonr(A_sampled,A_fitted)
        correlation_beta.append(corr_b[0])
        correlaiton_alpha.append(corr_a[0])
        correlation_A.append(corr_A[0])  


    return correlaiton_alpha,correlation_beta,correlation_A


sns.set(style="whitegrid")
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

# Test the effect of n trials
n_simulated_data = 50
n_trials = [10, 50,100,250,500,1000,1500,2000]
corr_a=[]
corr_b=[]
corr_A=[]
for n in n_trials:
    simualated_data_params,fitted_data_params = ganarate_parameters(V0, n, n_simulated_data)
    correlaiton_alpha,correlation_beta,correlation_A=plot_correlation(simualated_data_params, fitted_data_params, n, n_simulated_data,"n")
    corr_a.append(correlaiton_alpha)
    corr_b.append(correlation_beta)
    corr_A.append(correlation_A)

corr_a=np.array(corr_a).flatten()
corr_b=np.array(corr_b).flatten()
corr_A=np.array(corr_A).flatten()

corr_a=pd.DataFrame({"Pearson Correlation":corr_a,
                     "Trial":n_trials})
corr_b=pd.DataFrame({"Pearson Correlation":corr_b,
                     "Trial":n_trials})
corr_A=pd.DataFrame({"Pearson Correlation":corr_A,
                     "Trial":n_trials})

sns.lineplot(x="Trial",y="Pearson Correlation",data=corr_a, label="Alpha",ax=ax1)
sns.lineplot(x="Trial",y="Pearson Correlation", data=corr_b,label="Beta",ax=ax1)
sns.lineplot(x="Trial",y="Pearson Correlation", data=corr_A,label="A",ax=ax1)
ax1.set_xlabel('Number of trials')
ax1.set_ylabel('Pearson\'s correlation')
ax1.set_title(f'Effect of number of trials on Correlation , Number of Simulations ={n_simulated_data}')
ax1.legend(loc="upper right", frameon=False)

# Test the effect of n simulated data
n_trials = 100
n_simulated_data = [10, 50, 100,250,500,1000,1500,2000]
corr_a=[]
corr_b=[]
corr_A=[]
for m in n_simulated_data:
    simualated_data_params,fitted_data_params = ganarate_parameters(V0, n_trials, m)
    correlaiton_alpha,correlation_beta,correlation_A=plot_correlation(simualated_data_params, fitted_data_params, n_trials, m,"m")
    corr_a.append(correlaiton_alpha)
    corr_b.append(correlation_beta)
    corr_A.append(correlation_A)

#fix plotting turn into fatafrmae 
corr_a=np.array(corr_a).flatten()
corr_b=np.array(corr_b).flatten()
corr_A=np.array(corr_A).flatten()
corr_a=pd.DataFrame({"PearsonCorrelation":corr_a,
                     "NSimul":n_simulated_data})
corr_b=pd.DataFrame({"PearsonCorrelation":corr_b,
                     "NSimul":n_simulated_data})

corr_A=pd.DataFrame({"PearsonCorrelation":corr_A,
                     "NSimul":n_simulated_data})

sns.lineplot(x="NSimul",y="PearsonCorrelation",data=corr_a, label="Alpha",ax=ax2)
sns.lineplot(x="NSimul",y="PearsonCorrelation", data=corr_b,label="Beta",ax=ax2)
sns.lineplot(x="NSimul",y="PearsonCorrelation", data=corr_A,label="A",ax=ax2)
ax2.set_xlabel('Number of Simulations')
ax2.set_ylabel('Pearson\'s correlation')
ax2.set_title(f'Effect of number of simulated data on Correlation , Number of trials={n_trials}')
ax2.legend(loc="upper right", frameon=False)

plt.show()
'''


 
# Model Three #
###############

mean = [0.337, 0.54,7.171]
variance = [0.053,0.033, 0.5166]
n_samples = 50
n_trials = 160
v0 = 0.5
n_simulations = 1


# Original parameter values from data
original_params = [0.337, 0.54,7.171]

params = simulate_parameters_model2(mean, variance, n_samples)

# Plot alpha and beta in different plots
sns.distplot(params[:, 0], hist=True, rug=False, label='Alpha+')
sns.distplot(params[:, 1], hist=True, rug=False, label='Alpha-')
sns.distplot(params[:, 2], hist=True, rug=False, label='Beta')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title("Distribution of Sampled Alpha's and Beta's")
plt.legend(loc="upper right", frameon=False)
xticks = np.arange(0, 10, 1)
plt.xticks(xticks, xticks)

plt.show()

'''
correlation_coeffs_alphapos = []
correlation_coeffs_alphaneg = []
correlation_coeffs_beta = []

fitted_params_all = []
simulated_params_all=[] 

for i in range(5):
    simulated_choices = []
    simulated_outcomes = []
    simulated_params=simulate_parameters_model3(mean, variance, n_samples)
    simulated_params_all.append(simulated_params)
    for j in range(n_samples):
        choices, outcomes = simulate_data_model3(simulated_params[j],v0, n_trials, 1)
        simulated_choices.append(choices)
        simulated_outcomes.append(outcomes)
    fitted_paramstemp=[]
    for k in range(n_samples):
        fitted_params_temp=fit_params_model3(simulated_choices[k], simulated_outcomes[k], mean[0], mean[2],mean[1], v0)
        fitted_paramstemp.append(fitted_params_temp)

    fitted_params = np.array(fitted_paramstemp)
    fitted_params_all.append(fitted_params)




for i, arr in enumerate(fitted_params_all):
    arr[:, 1] = (arr[:, 1] - np.min(arr[:, 1])) / (np.max(arr[:, 1]) - np.min(arr[:, 1])) * (25 - 4) + 4
    fitted_params_all[i] = arr

alphaspos_fitted = []
alphasneg_fitted=[]
betas_fitted= []


for array in fitted_params_all:
    inner_alphaspos=[]
    inner_alphasneg=[]
    inner_beta=[]
    for inner_array in array:
        inner_alphaspos.append(inner_array[0])
        inner_alphasneg.append(inner_array[1])
        inner_beta.append(inner_array[2])
    alphaspos_fitted.append(inner_alphaspos)
    alphasneg_fitted.append(inner_alphasneg)
    betas_fitted.append(inner_beta)



alphaspos_sim = []
alphasneg_sim=[]
betas_sim= []

for array in simulated_params_all:
    inner_alphaspos=[]
    inner_alphasneg=[]
    inner_beta=[]
    for inner_array in array:
        inner_alphaspos.append(inner_array[0])
        inner_alphasneg.append(inner_array[1])
        inner_beta.append(inner_array[2])
    alphaspos_sim.append(inner_alphaspos)
    alphasneg_sim.append(inner_alphasneg)
    betas_sim.append(inner_beta)



for sublist1, sublist2 in zip(alphaspos_sim, alphaspos_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_alphapos.append(r)

for sublist1, sublist2 in zip(alphasneg_sim, alphasneg_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_alphaneg.append(r)

for sublist1, sublist2 in zip(betas_sim, betas_fitted):
    r, p = pearsonr(sublist1, sublist2)
    correlation_coeffs_beta.append(r)





fig, ax = plt.subplots(figsize=(10,10))

data=[correlation_coeffs_alphapos,correlation_coeffs_alphaneg,correlation_coeffs_beta]
sns.boxplot(data=data, ax=ax)

ax.set_xticklabels(['Alpha+', 'Alpha-','Beta'])
ax.set_ylabel('Correlation')
ax.legend(['Alpha+', 'Alpha-','Beta'],loc="upper right", frameon=False)
plt.show()

def ganarate_parameters( v0, n_trials, n_simulated_data):
    fitted_data_params = []
    simualated_data_params=[]
    for i in range(n_simulated_data):
        simulated_params = np.random.multivariate_normal(mean, np.diag(variance))
        simulated_choices, simulated_outcomes = simulate_data_model3(simulated_params, v0, n_trials,1)
        fitted_params = fit_params_model3(simulated_choices, simulated_outcomes,original_params[0],original_params[2],original_params[1], v0)
        simualated_data_params.append(simulated_params)
        fitted_data_params.append(fitted_params)
    return simualated_data_params,fitted_data_params

def plot_correlation(sampled_params, fitted_param, n, m,str):
    correlaiton_alphapos = []
    correlation_alphaneg=[]
    correlation_beta=[]
    if str == "m":
        indices_to_remove = [i for i, x in enumerate(fitted_param) if x[1] > 100]
        fitted_param = [x for i, x in enumerate(fitted_param) if i not in indices_to_remove]
        sampled_params = [x for i, x in enumerate(sampled_params) if i not in indices_to_remove]


        alphapos_sampled = [subarray[0] for subarray in sampled_params]
        alphapos_fitted = [subarray[0] for subarray in fitted_param]
        alphaneg_sampled  = [subarray[1] for subarray in sampled_params]
        alphaneg_fitted =  [subarray[1] for subarray in fitted_param] 
        beta_sampled = [subarray[2] for subarray in sampled_params]
        beta_fitted = [subarray[2] for subarray in fitted_param]       
        corr_apos  = pearsonr(alphapos_sampled,alphapos_fitted)
        corr_b  = pearsonr(beta_sampled,beta_fitted)
        corr_aneg= pearsonr(alphaneg_sampled,alphaneg_fitted)
        correlation_beta.append(corr_b[0])
        correlaiton_alphapos.append(corr_apos[0])
        correlation_alphaneg.append(corr_aneg[0])

    elif str == "n":
        indices_to_remove = [i for i, x in enumerate(fitted_param) if x[1] > 100]
        fitted_param = [x for i, x in enumerate(fitted_param) if i not in indices_to_remove]
        sampled_params = [x for i, x in enumerate(sampled_params) if i not in indices_to_remove]

        alphapos_sampled = [subarray[0] for subarray in sampled_params]
        alphapos_fitted = [subarray[0] for subarray in fitted_param]
        alphaneg_sampled  = [subarray[1] for subarray in sampled_params]
        alphaneg_fitted =  [subarray[1] for subarray in fitted_param] 
        beta_sampled = [subarray[2] for subarray in sampled_params]
        beta_fitted = [subarray[2] for subarray in fitted_param]       

        corr_apos  = pearsonr(alphapos_sampled,alphapos_fitted)
        corr_b  = pearsonr(beta_sampled,beta_fitted)
        corr_aneg= pearsonr(alphaneg_sampled,alphaneg_fitted)
        correlation_beta.append(corr_b[0])
        correlaiton_alphapos.append(corr_apos[0])
        correlation_alphaneg.append(corr_aneg[0])


    return correlaiton_alphapos,correlation_alphaneg,correlation_beta


sns.set(style="whitegrid")
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

# Test the effect of n trials
n_simulated_data = 50
n_trials = [10,50,100]#,250,500,1000,1250]
corr_apos=[]
corr_b=[]
corr_aneg=[]
for n in n_trials:
    simualated_data_params,fitted_data_params = ganarate_parameters(V0, n, n_simulated_data)
    correlaiton_alphapos,correlaiton_alphaneg,correlation_beta=plot_correlation(simualated_data_params, fitted_data_params, n, n_simulated_data,"n")
    corr_apos.append(correlaiton_alphapos)
    corr_b.append(correlation_beta)
    corr_aneg.append(correlaiton_alphaneg)

corr_apos=np.array(corr_apos).flatten()
corr_b=np.array(corr_b).flatten()
corr_aneg=np.array(corr_aneg).flatten()

corr_apos=pd.DataFrame({"Pearson Correlation":corr_apos,
                     "Trial":n_trials})
corr_b=pd.DataFrame({"Pearson Correlation":corr_b,
                     "Trial":n_trials})
corr_aneg=pd.DataFrame({"Pearson Correlation":corr_aneg,
                     "Trial":n_trials})

sns.lineplot(x="Trial",y="Pearson Correlation",data=corr_apos, label="Alpha+",ax=ax1)
sns.lineplot(x="Trial",y="Pearson Correlation", data=corr_aneg,label="Alpha-",ax=ax1)
sns.lineplot(x="Trial",y="Pearson Correlation", data=corr_b,label="Beta",ax=ax1)
ax1.set_xlabel('Number of trials')
ax1.set_ylabel('Pearson\'s correlation')
ax1.set_title(f'Effect of number of trials on Correlation , Number of Simulations ={n_simulated_data}')
ax1.legend(loc="upper right", frameon=False)

# Test the effect of n simulated data
n_trials = 100
n_simulated_data = [10, 50, 100]#,250,500,1000,1250]
corr_apos=[]
corr_b=[]
corr_aneg=[]
for m in n_simulated_data:
    simualated_data_params,fitted_data_params = ganarate_parameters(V0, n_trials, m)
    correlaiton_alphapos,correlaiton_alphaneg,correlation_beta=plot_correlation(simualated_data_params, fitted_data_params, n_trials, m,"m")
    corr_apos.append(correlaiton_alphapos)
    corr_b.append(correlation_beta)
    corr_aneg.append(correlaiton_alphaneg)

corr_apos=np.array(corr_apos).flatten()
corr_b=np.array(corr_b).flatten()
corr_aneg=np.array(corr_aneg).flatten()

corr_apos=pd.DataFrame({"Pearson Correlation":corr_apos,
                     "NSimul":n_trials})
corr_b=pd.DataFrame({"Pearson Correlation":corr_b,
                     "NSimul":n_trials})
corr_aneg=pd.DataFrame({"Pearson Correlation":corr_aneg,
                     "NSimul":n_trials})

sns.lineplot(x="NSimul",y="Pearson Correlation",data=corr_apos, label="Alpha+",ax=ax2)
sns.lineplot(x="NSimul",y="Pearson Correlation", data=corr_aneg,label="Alpha-",ax=ax2)
sns.lineplot(x="NSimul",y="Pearson Correlation", data=corr_b,label="Beta",ax=ax2)
ax2.set_xlabel('Number of simulations')
ax2.set_ylabel('Pearson\'s correlation')
ax2.set_title(f'Effect of number of simulations on Correlation , Number of Trials ={n_trials}')
ax2.legend(loc="upper right", frameon=False)

plt.show()
'''












#est how samples and simulations and tiralas affect AIC scores FOR EACH MODEL
##################
# Model Recovery #
##################

# Model One #
#############


# Set paremters 
alpha = 0.4
beta = 7
V0 = 0.5
n_trials= 160
n_simulations= 1
n_samples=50


params=[0.4,7]
datasets=[]
for i in range(n_simulations):
    simulated_choices = []
    simulated_outcomes = []
    dataframe=[]
    for j in range(n_samples):
        choices, outcomes = simulate_data_model1(params,V0, n_trials, 1)
        simulated_choices.append(choices)
        simulated_outcomes.append(outcomes)

    # Construct the column names for each trial
    trial_cols = [f'trial_{t}' for t in range(1, n_trials+1)]
    # Append simulated data to a dictionary
    simulated_data = {'Participant': range(1, n_samples+1)}
    for t in range(n_trials):
        trial_data = []
        for p in range(n_samples):
            choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
            trial_data.append(choice_outcome)
        simulated_data[trial_cols[t]] = trial_data
    # Convert the dictionary to a dataframe
    df = pd.DataFrame(simulated_data)
    datasets.append(df)

#Fit to  model 


NLLs=[]
for d in datasets:
    fitted_paremters_model1=fit_params_for_all_participants_model1_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alpha,beta=fitted_paremters_model1[i]
        NLLS=calculate_NLL_for_participant_model1_recovery(i,d, alpha, beta, V0)
        NLLs.append(NLLS)

data1=datasets[0]
data1["NLL"]=NLLs


'''
sns.set(style="whitegrid")
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

# Test the effect of n trials

n_simulations= 1
n_samples=50
n_trials = [10,50,100,250,500,1000]
datasets_model1=[]
for n in n_trials:
    for i in range(n_simulations):
        simulated_choices = []
        simulated_outcomes = []
        dataframe=[]
        for j in range(n_samples):
            choices, outcomes = simulate_data_model1(params,V0, n, 1)
            simulated_choices.append(choices)
            simulated_outcomes.append(outcomes)

        # Construct the column names for each trial
        trial_cols = [f'trial_{t}' for t in range(1, n+1)]
        # Append simulated data to a dictionary
        simulated_data = {'Participant': range(1, n_samples+1)}
        for t in range(n):
            trial_data = []
            for p in range(n_samples):
                choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
                trial_data.append(choice_outcome)
            simulated_data[trial_cols[t]] = trial_data
        # Convert the dictionary to a dataframe
        df = pd.DataFrame(simulated_data)
        datasets_model1.append(df)
 


for d in datasets_model1:

    fitted_paremters_model1=fit_params_for_all_participants_model1_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alpha,beta=fitted_paremters_model1[i]
        NLLS=calculate_NLL_for_participant_model1_recovery(i,d, alpha, beta, V0)
        inner_NLL.append(NLLS)

    d['NLL']=inner_NLL

#Compute AIC and BIC then plot 

# number of parameters for each model
AIC=[]
BIC=[]
for d in datasets_model1:
    rows,columns=d.shape
    # number of observations
    particpants= rows 
    n= columns-2
    p=2
    # arrays to store  AIC and BIC scores for each model

    d["AIC"]=np.zeros(particpants)
    d['BIC']=np.zeros(particpants)


    # loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
    for i in range(0,particpants):

        d.loc[i,"AIC"] = (2 * d.loc[i,'NLL']) + (2 * p)
        d.loc[i,'BIC'] = (2 * d.loc[i,'NLL']) + (p * np.log(n))
      
    # sum up the scores for data set
    AIC.append(np.sum(d.loc[:,"AIC"]))
    BIC.append(np.sum(d.loc[:,"BIC"]))






Data_AIC=pd.DataFrame({"AIC":AIC,
                     "Trial":n_trials})
Data_BIC=pd.DataFrame({"BIC":BIC,
                     "Trial":n_trials})


sns.lineplot(x="Trial",y="AIC",data=Data_AIC, label="AIC",ax=ax1)
sns.lineplot(x="Trial",y="BIC", data=Data_BIC,label="BIC",ax=ax1)
ax1.set_xlabel('Number of trials')
ax1.set_ylabel('Model Recovery Accuracy')
ax1.set_title(f'Effect of number of trials on Accuracy')
ax1.legend(loc="upper right", frameon=False)



# Test the effect of n samples

n_simulations= 1
n_samples=[10,50,100,250,500,1000]
n_trials = 100
datasets_model1=[]
for m in n_samples:
    for i in range(n_simulations):
        simulated_choices = []
        simulated_outcomes = []
        dataframe=[]
        for j in range(m):
            choices, outcomes = simulate_data_model1(params,V0, n_trials, 1)
            simulated_choices.append(choices)
            simulated_outcomes.append(outcomes)

        # Construct the column names for each trial
        trial_cols = [f'trial_{t}' for t in range(1, n_trials+1)]
        # Append simulated data to a dictionary
        simulated_data = {'Participant': range(1, m+1)}
        for t in range(n_trials):
            trial_data = []
            for p in range(m):
                choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
                trial_data.append(choice_outcome)
            simulated_data[trial_cols[t]] = trial_data
        # Convert the dictionary to a dataframe
        df = pd.DataFrame(simulated_data)
        datasets_model1.append(df)
 

for d in datasets_model1:
    fitted_paremters_model1=fit_params_for_all_participants_model1_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alpha,beta=fitted_paremters_model1[i]
        NLLS=calculate_NLL_for_participant_model1_recovery(i,d, alpha, beta, V0)
        inner_NLL.append(NLLS)
    d['NLL']=inner_NLL


#Compute AIC and BIC then plot 

# number of parameters for each model
AIC=[]
BIC=[]
for d in datasets_model1:
    rows,columns=d.shape
    # number of observations
    particpants= rows 
    n= columns-2
    p=2
    # arrays to store  AIC and BIC scores for each model

    d["AIC"]=np.zeros(particpants)
    d['BIC']=np.zeros(particpants)


    # loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
    for i in range(0,particpants):

        d.loc[i,"AIC"] = (2 * d.loc[i,'NLL']) + (2 * p)
        d.loc[i,'BIC'] = (2 * d.loc[i,'NLL']) + (p * np.log(n))
      
    # sum up the scores for each model
    AIC.append(np.sum(d.loc[:,"AIC"]))
    BIC.append(np.sum(d.loc[:,"BIC"]))






Data_AIC=pd.DataFrame({"AIC":AIC,
                     "Samples":n_samples})
Data_BIC=pd.DataFrame({"BIC":BIC,
                     "Samples":n_samples})


sns.lineplot(x="Samples",y="AIC",data=Data_AIC, label="AIC",ax=ax2)
sns.lineplot(x="Samples",y="BIC", data=Data_BIC,label="BIC",ax=ax2)
ax2.set_xlabel('Number of Particapnts')
ax2.set_ylabel('Model Recovery Accuracy')
ax2.set_title(f'Effect of number of Particpants on Accuracy')
ax2.legend(loc="upper right", frameon=False)

plt.show()
'''


# Model Two #
#############
# Set paremters 
alpha = 0.4
beta = 5
V0 = 0.5
n_trials= 160
n_simulations= 1
n_samples=50

params=[0.4 ,5,0.5 ]
datasets_model2=[]


for i in range(n_simulations):
    simulated_choices = []
    simulated_outcomes = []
    dataframe=[]
    for j in range(n_samples):
        choices, outcomes = simulate_data_model2(params,V0, n_trials, 1)
        simulated_choices.append(choices)
        simulated_outcomes.append(outcomes)

    # Construct the column names for each trial
    trial_cols = [f'trial_{t}' for t in range(1, n_trials+1)]
    # Append simulated data to a dictionary
    simulated_data = {'Participant': range(1, n_samples+1)}
    for t in range(n_trials):
        trial_data = []
        for p in range(n_samples):
            choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
            trial_data.append(choice_outcome)
        simulated_data[trial_cols[t]] = trial_data
    # Convert the dictionary to a dataframe
    df = pd.DataFrame(simulated_data)
    datasets_model2.append(df)

#Fit to  model 
parameters=[0.4,5,0.5]
V0=0.5
NLLs=[]
for d in datasets_model2:
    fitted_paremters_model2=fit_params_for_all_participants_model2_recovery(d,V0,parameters)

    for i,row in d.iterrows():
        alpha,beta,A=fitted_paremters_model2[i]
        NLLS=calculate_NLL_for_participant_model2_recovery(i,d, alpha, beta,A, V0)
        NLLs.append(NLLS)

data2=datasets_model2[0]
data2["NLL"]=NLLs

'''
sns.set(style="whitegrid")
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

# Test the effect of n trials

n_simulations= 1
n_samples=50
n_trials = [10,50,100,250,500,1000]
datasets_model2=[]
for n in n_trials:
    for i in range(n_simulations):
        simulated_choices = []
        simulated_outcomes = []
        dataframe=[]
        for j in range(n_samples):
            choices, outcomes = simulate_data_model2(params,V0, n, 1)
            simulated_choices.append(choices)
            simulated_outcomes.append(outcomes)

        # Construct the column names for each trial
        trial_cols = [f'trial_{t}' for t in range(1, n+1)]
        # Append simulated data to a dictionary
        simulated_data = {'Participant': range(1, n_samples+1)}
        for t in range(n):
            trial_data = []
            for p in range(n_samples):
                choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
                trial_data.append(choice_outcome)
            simulated_data[trial_cols[t]] = trial_data
        # Convert the dictionary to a dataframe
        df = pd.DataFrame(simulated_data)
        datasets_model2.append(df)
 


for d in datasets_model2:

    fitted_paremters_model2=fit_params_for_all_participants_model2_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alpha,beta,A=fitted_paremters_model2[i]
        NLLS=calculate_NLL_for_participant_model2_recovery(i,d, alpha, beta,A, V0)
        inner_NLL.append(NLLS)

    d['NLL']=inner_NLL

#Compute AIC and BIC then plot 

# number of parameters for each model
AIC=[]
BIC=[]
for d in datasets_model2:
    rows,columns=d.shape
    # number of observations
    particpants= rows 
    n= columns-2
    p=3
    # arrays to store  AIC and BIC scores for each model

    d["AIC"]=np.zeros(particpants)
    d['BIC']=np.zeros(particpants)


    # loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
    for i in range(0,particpants):

        d.loc[i,"AIC"] = (2 * d.loc[i,'NLL']) + (2 * p)
        d.loc[i,'BIC'] = (2 * d.loc[i,'NLL']) + (p * np.log(n))
      
    # sum up the scores for data set
    AIC.append(np.sum(d.loc[:,"AIC"]))
    BIC.append(np.sum(d.loc[:,"BIC"]))






Data_AIC=pd.DataFrame({"AIC":AIC,
                     "Trial":n_trials})
Data_BIC=pd.DataFrame({"BIC":BIC,
                     "Trial":n_trials})


sns.lineplot(x="Trial",y="AIC",data=Data_AIC, label="AIC",ax=ax1)
sns.lineplot(x="Trial",y="BIC", data=Data_BIC,label="BIC",ax=ax1)
ax1.set_xlabel('Number of trials')
ax1.set_ylabel('Model Recovery Accuracy')
ax1.set_title(f'Effect of number of trials on Accuracy')
ax1.legend(loc="upper right", frameon=False)



# Test the effect of n samples

n_simulations= 1
n_samples=[10,50,100,250,500,1000]
n_trials = 100
datasets_model2=[]
for m in n_samples:
    for i in range(n_simulations):
        simulated_choices = []
        simulated_outcomes = []
        dataframe=[]
        for j in range(m):
            choices, outcomes = simulate_data_model2(params,V0, n_trials, 1)
            simulated_choices.append(choices)
            simulated_outcomes.append(outcomes)

        # Construct the column names for each trial
        trial_cols = [f'trial_{t}' for t in range(1, n_trials+1)]
        # Append simulated data to a dictionary
        simulated_data = {'Participant': range(1, m+1)}
        for t in range(n_trials):
            trial_data = []
            for p in range(m):
                choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
                trial_data.append(choice_outcome)
            simulated_data[trial_cols[t]] = trial_data
        # Convert the dictionary to a dataframe
        df = pd.DataFrame(simulated_data)
        datasets_model2.append(df)
 

for d in datasets_model2:
    fitted_paremters_model1=fit_params_for_all_participants_model2_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alpha,beta,A=fitted_paremters_model1[i]
        NLLS=calculate_NLL_for_participant_model2_recovery(i,d, alpha, beta,A, V0)
        inner_NLL.append(NLLS)
    d['NLL']=inner_NLL


#Compute AIC and BIC then plot 

# number of parameters for each model
AIC=[]
BIC=[]
for d in datasets_model2:
    rows,columns=d.shape
    # number of observations
    particpants= rows 
    n= columns-2
    p=3
    # arrays to store  AIC and BIC scores for each model

    d["AIC"]=np.zeros(particpants)
    d['BIC']=np.zeros(particpants)


    # loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
    for i in range(0,particpants):

        d.loc[i,"AIC"] = (2 * d.loc[i,'NLL']) + (2 * p)
        d.loc[i,'BIC'] = (2 * d.loc[i,'NLL']) + (p * np.log(n))
      
    # sum up the scores for each model
    AIC.append(np.sum(d.loc[:,"AIC"]))
    BIC.append(np.sum(d.loc[:,"BIC"]))






Data_AIC=pd.DataFrame({"AIC":AIC,
                     "Samples":n_samples})
Data_BIC=pd.DataFrame({"BIC":BIC,
                     "Samples":n_samples})


sns.lineplot(x="Samples",y="AIC",data=Data_AIC, label="AIC",ax=ax2)
sns.lineplot(x="Samples",y="BIC", data=Data_BIC,label="BIC",ax=ax2)
ax2.set_xlabel('Number of Particapnts')
ax2.set_ylabel('Model Recovery Accuracy')
ax2.set_title(f'Effect of number of Particpants on Accuracy')
ax2.legend(loc="upper right", frameon=False)

plt.show()
'''

# Model Three #
###############
# Set parameters 
alpha_pos=0.25
beta=6
alpha_neg=0.5
V0 = 0.5
n_trials=160
n_simulations= 1
n_samples=50

params=[0.25 ,6,0.5]
datasets_model3=[]


for i in range(n_simulations):
    simulated_choices = []
    simulated_outcomes = []
    dataframe=[]
    for j in range(n_samples):
        choices, outcomes = simulate_data_model3(params,V0, n_trials, 1)
        simulated_choices.append(choices)
        simulated_outcomes.append(outcomes)

    # Construct the column names for each trial
    trial_cols = [f'trial_{t}' for t in range(1, n_trials+1)]
    # Append simulated data to a dictionary
    simulated_data = {'Participant': range(1, n_samples+1)}
    for t in range(n_trials):
        trial_data = []
        for p in range(n_samples):
            choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
            trial_data.append(choice_outcome)
        simulated_data[trial_cols[t]] = trial_data
    # Convert the dictionary to a dataframe
    df = pd.DataFrame(simulated_data)
    datasets_model3.append(df)



#Fit to  model 
parameters=[0.25,6,0.5]
V0=0.5
NLLs=[]
for d in datasets_model3:
    fitted_paremters_model3=fit_params_for_all_participants_model3_recovery(d,V0,parameters)

    for i,row in d.iterrows():
        alpha_pos,beta,alpha_neg=fitted_paremters_model3[i]
        NLLS=calculate_NLL_for_participant_model3_recvoery(i,d, alpha_pos, beta,alpha_neg, V0)
        NLLs.append(NLLS)

data3=datasets_model3[0]
data3["NLL"]=NLLs

'''
sns.set(style="whitegrid")
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

# Test the effect of n trials

n_simulations= 1
n_samples=50
n_trials = [10,50,100,250,500,1000]
datasets_model3=[]
for n in n_trials:
    for i in range(n_simulations):
        simulated_choices = []
        simulated_outcomes = []
        dataframe=[]
        for j in range(n_samples):
            choices, outcomes = simulate_data_model3(params,V0, n, 1)
            simulated_choices.append(choices)
            simulated_outcomes.append(outcomes)

        # Construct the column names for each trial
        trial_cols = [f'trial_{t}' for t in range(1, n+1)]
        # Append simulated data to a dictionary
        simulated_data = {'Participant': range(1, n_samples+1)}
        for t in range(n):
            trial_data = []
            for p in range(n_samples):
                choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
                trial_data.append(choice_outcome)
            simulated_data[trial_cols[t]] = trial_data
        # Convert the dictionary to a dataframe
        df = pd.DataFrame(simulated_data)
        datasets_model3.append(df)
 


for d in datasets_model3:

    fitted_paremters_model3=fit_params_for_all_participants_model3_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alphapos,beta,alphaneg=fitted_paremters_model3[i]
        NLLS=calculate_NLL_for_participant_model3_recvoery(i,d, alphapos, beta,alphaneg, V0)
        inner_NLL.append(NLLS)

    d['NLL']=inner_NLL

#Compute AIC and BIC then plot 

# number of parameters for each model
AIC=[]
BIC=[]
for d in datasets_model3:
    rows,columns=d.shape
    # number of observations
    particpants= rows 
    n= columns-2
    p=3
    # arrays to store  AIC and BIC scores for each model

    d["AIC"]=np.zeros(particpants)
    d['BIC']=np.zeros(particpants)


    # loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
    for i in range(0,particpants):

        d.loc[i,"AIC"] = (2 * d.loc[i,'NLL']) + (2 * p)
        d.loc[i,'BIC'] = (2 * d.loc[i,'NLL']) + (p * np.log(n))
      
    # sum up the scores for data set
    AIC.append(np.sum(d.loc[:,"AIC"]))
    BIC.append(np.sum(d.loc[:,"BIC"]))






Data_AIC=pd.DataFrame({"AIC":AIC,
                     "Trial":n_trials})
Data_BIC=pd.DataFrame({"BIC":BIC,
                     "Trial":n_trials})


sns.lineplot(x="Trial",y="AIC",data=Data_AIC, label="AIC",ax=ax1)
sns.lineplot(x="Trial",y="BIC", data=Data_BIC,label="BIC",ax=ax1)
ax1.set_xlabel('Number of trials')
ax1.set_ylabel('Model Recovery Accuracy')
ax1.set_title(f'Effect of number of trials on Accuracy')
ax1.legend(loc="upper right", frameon=False)



# Test the effect of n samples

n_simulations= 1
n_samples=[10,50,100,250,500,1000]
n_trials = 100
datasets_model3=[]
for m in n_samples:
    for i in range(n_simulations):
        simulated_choices = []
        simulated_outcomes = []
        dataframe=[]
        for j in range(m):
            choices, outcomes = simulate_data_model3(params,V0, n_trials, 1)
            simulated_choices.append(choices)
            simulated_outcomes.append(outcomes)

        # Construct the column names for each trial
        trial_cols = [f'trial_{t}' for t in range(1, n_trials+1)]
        # Append simulated data to a dictionary
        simulated_data = {'Participant': range(1, m+1)}
        for t in range(n_trials):
            trial_data = []
            for p in range(m):
                choice_outcome = [int(simulated_choices[p][t]), int(simulated_outcomes[p][t])]
                trial_data.append(choice_outcome)
            simulated_data[trial_cols[t]] = trial_data
        # Convert the dictionary to a dataframe
        df = pd.DataFrame(simulated_data)
        datasets_model3.append(df)
 

for d in datasets_model3:
    fitted_paremters_model3=fit_params_for_all_participants_model3_recovery(d,V0,params)
    inner_NLL=[]
    for i,row in d.iterrows():
        alphapos,beta,alphaneg=fitted_paremters_model3[i]
        NLLS=calculate_NLL_for_participant_model3_recvoery(i,d, alphapos, beta,alphaneg, V0)
        inner_NLL.append(NLLS)
    d['NLL']=inner_NLL


#Compute AIC and BIC then plot 

# number of parameters for each model
AIC=[]
BIC=[]
for d in datasets_model3:
    rows,columns=d.shape
    # number of observations
    particpants= rows 
    n= columns-2
    p=3
    # arrays to store  AIC and BIC scores for each model

    d["AIC"]=np.zeros(particpants)
    d['BIC']=np.zeros(particpants)


    # loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
    for i in range(0,particpants):

        d.loc[i,"AIC"] = (2 * d.loc[i,'NLL']) + (2 * p)
        d.loc[i,'BIC'] = (2 * d.loc[i,'NLL']) + (p * np.log(n))
      
    # sum up the scores for each model
    AIC.append(np.sum(d.loc[:,"AIC"]))
    BIC.append(np.sum(d.loc[:,"BIC"]))






Data_AIC=pd.DataFrame({"AIC":AIC,
                     "Samples":n_samples})
Data_BIC=pd.DataFrame({"BIC":BIC,
                     "Samples":n_samples})


sns.lineplot(x="Samples",y="AIC",data=Data_AIC, label="AIC",ax=ax2)
sns.lineplot(x="Samples",y="BIC", data=Data_BIC,label="BIC",ax=ax2)
ax2.set_xlabel('Number of Particapnts')
ax2.set_ylabel('Model Recovery Accuracy')
ax2.set_title(f'Effect of number of Particpants on Accuracy')
ax2.legend(loc="upper right", frameon=False)

plt.show()
'''



##########################################
# Model Comparision and AIC ,BIC Scores  #
##########################################

# Fitted Data # 
################

# number of parameters for each model
p1 = 2
p2 = 3

# number of observations
n = 160

particpants=50
# arrays to store  AIC and BIC scores for each model

Behaviour_Data["AIC1"]=np.zeros(particpants)
Behaviour_Data['AIC2']=np.zeros(particpants)
Behaviour_Data['BIC1']=np.zeros(particpants)
Behaviour_Data['BIC2']=np.zeros(particpants)
Behaviour_Data['AIC3']=np.zeros(particpants)
Behaviour_Data['BIC3']=np.zeros(particpants)

# loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
for i in range(0,particpants):

    Behaviour_Data.loc[i,"AIC1"] = (2 * Behaviour_Data.loc[i,'NLL_Model1']) + (2 * p1)
    Behaviour_Data.loc[i,'AIC2'] = (2 * Behaviour_Data.loc[i,'NLL_Model2']) + (2 * p2)
    Behaviour_Data.loc[i,'AIC3'] = (2 * Behaviour_Data.loc[i,'NLL_Model3']) + (2 * p2)
    Behaviour_Data.loc[i,'BIC1'] = (2 * Behaviour_Data.loc[i,'NLL_Model1']) + (p1 * np.log(n))
    Behaviour_Data.loc[i,'BIC2'] = (2 * Behaviour_Data.loc[i,'NLL_Model2']) + (p2 * np.log(n))
    Behaviour_Data.loc[i,'BIC3'] = (2 * Behaviour_Data.loc[i,'NLL_Model3']) + (p2 * np.log(n))

# sum up the scores for each model
sum_AIC1 = np.sum(Behaviour_Data.loc[:,"AIC1"])
sum_AIC2 = np.sum(Behaviour_Data.loc[:,"AIC2"])
sum_AIC3 = np.sum(Behaviour_Data.loc[:,"AIC3"])

sum_BIC1 = np.sum(Behaviour_Data.loc[:,"BIC1"])
sum_BIC2 = np.sum(Behaviour_Data.loc[:,"BIC2"])
sum_BIC3 = np.sum(Behaviour_Data.loc[:,"BIC3"])

# report the results
print("Model 1 AIC score:", sum_AIC1)
print("Model 2 AIC score:", sum_AIC2)
print("Model 3 AIC score:", sum_AIC3)

print("Model 1 BIC score:", sum_BIC1)
print("Model 2 BIC score:", sum_BIC2)
print("Model 3 BIC score:", sum_BIC3)

# choose the "best" model
if sum_AIC1 < sum_AIC2:
    if sum_AIC1< sum_AIC3:
        print("Model 1 is the best model according to AIC")
    else:
        print("Model 3 is the best model according to AIC")
elif sum_AIC2< sum_AIC3:
        print("Model 2 is the best model according to AIC")
else:
        print("Model 3 is the best model according to AIC")

if sum_BIC1 < sum_BIC2:
    if sum_BIC1< sum_BIC3:
        print("Model 1 is the best model according to BIC")
    else:
        print("Model 3 is the best model according to BIC")
elif sum_BIC2< sum_BIC3:
        print("Model 2 is the best model according to BIC")
else:
        print("Model 3 is the best model according to BIC")

# split the data into high anxious and low anxious groups
high_anxious = Behaviour_Data.iloc[:25,:]
low_anxious = Behaviour_Data.iloc[26:,:]

# create the figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot the AIC scores
ax[0].scatter(np.arange(1,26), high_anxious['AIC1'], c='red', label='High Anxious, Model 1')
ax[0].scatter(np.arange(1,26), high_anxious['AIC2'], c='red', marker='x', label='High Anxious, Model 2')
ax[0].scatter(np.arange(1,26), high_anxious['AIC3'], c='red', marker='v', label='High Anxious, Model 3')

ax[0].scatter(np.arange(26,50), low_anxious['AIC1'], c='blue', label='Low Anxious, Model 1')
ax[0].scatter(np.arange(26,50) ,low_anxious['AIC2'], c='blue', marker='x', label='Low Anxious, Model 2')
ax[0].scatter(np.arange(26,50) ,low_anxious['AIC3'], c='blue', marker='v', label='Low Anxious, Model 3')
ax[0].set_xlabel('Particpants')
ax[0].set_ylabel('AIC Score')
ax[0].set_title('AIC Scores')
ax[0].legend()

ax[1].scatter(np.arange(1,26), high_anxious['BIC1'], c='red', label='High Anxious, Model 1')
ax[1].scatter(np.arange(1,26), high_anxious['BIC2'], c='red', marker='x', label='High Anxious, Model 2')
ax[1].scatter(np.arange(1,26), high_anxious['BIC3'], c='red', marker='v', label='High Anxious, Model 3')

ax[1].scatter(np.arange(26,50), low_anxious['BIC1'], c='blue', label='Low Anxious, Model 1')
ax[1].scatter(np.arange(26,50), low_anxious['BIC2'], c='blue', marker='x', label='Low Anxious, Model 2')
ax[1].scatter(np.arange(26,50) ,low_anxious['BIC3'], c='blue', marker='v', label='Low Anxious, Model 3')

ax[1].set_xlabel('Partcipants ')
ax[1].set_ylabel('BIC Score')
ax[1].set_title('BIC Scores')
ax[1].legend()

plt.show()



sum_AIC1 = np.sum(Behaviour_Data.loc[:, "AIC1"])
sum_AIC2 = np.sum(Behaviour_Data.loc[:, "AIC2"])
sum_AIC3 = np.sum(Behaviour_Data.loc[:, "AIC3"])
sum_BIC1 = np.sum(Behaviour_Data.loc[:, "BIC1"])
sum_BIC2 = np.sum(Behaviour_Data.loc[:, "BIC2"])
sum_BIC3 = np.sum(Behaviour_Data.loc[:, "BIC3"])

sum_AIC1_high = np.sum(Behaviour_Data.loc[:25, "AIC1"])
sum_AIC2_high = np.sum(Behaviour_Data.loc[:25, "AIC2"])
sum_AIC3_high = np.sum(Behaviour_Data.loc[:25, "AIC3"])
sum_BIC1_high = np.sum(Behaviour_Data.loc[:25, "BIC1"])
sum_BIC2_high = np.sum(Behaviour_Data.loc[:25, "BIC2"])
sum_BIC3_high = np.sum(Behaviour_Data.loc[:25, "BIC3"])

sum_AIC1_low = np.sum(Behaviour_Data.loc[26:, "AIC1"])
sum_AIC2_low = np.sum(Behaviour_Data.loc[26:, "AIC2"])
sum_AIC3_low = np.sum(Behaviour_Data.loc[26:, "AIC3"])
sum_BIC1_low = np.sum(Behaviour_Data.loc[26:, "BIC1"])
sum_BIC2_low = np.sum(Behaviour_Data.loc[26:, "BIC2"])
sum_BIC3_low = np.sum(Behaviour_Data.loc[26:, "BIC3"])

print("Sum of AIC1 for all participants: ", sum_AIC1)
print("Sum of AIC2 for all participants: ", sum_AIC2)
print("Sum of AIC3 for all participants: ", sum_AIC3)
print("Sum of BIC1 for all participants: ", sum_BIC1)
print("Sum of BIC2 for all participants: ", sum_BIC2)
print("Sum of BIC3 for all participants: ", sum_BIC3)

print("Sum of AIC1 for high anxious participants: ", sum_AIC1_high)
print("Sum of AIC2 for high anxious participants: ", sum_AIC2_high)
print("Sum of AIC3 for high anxious participants: ", sum_AIC3_high)
print("Sum of BIC1 for high anxious participants: ", sum_BIC1_high)
print("Sum of BIC2 for high anxious participants: ", sum_BIC2_high)
print("Sum of BIC3 for high anxious participants: ", sum_BIC3_high)

print("Sum of AIC1 for low anxious participants: ", sum_AIC1_low)
print("Sum of AIC2 for low anxious participants: ", sum_AIC2_low)
print("Sum of AIC3 for low anxious participants: ", sum_AIC3_low)

print("Sum of BIC1 for low anxious participants: ", sum_BIC1_low)
print("Sum of BIC2 for low anxious participants: ", sum_BIC2_low)
print("Sum of BIC3 for low anxious participants: ", sum_BIC3_low)




# Simulated Data # 
##################


# number of parameters for each model
p1 = 2
p2 = 3

# number of observations
n = 160

particpants=50
# arrays to store  AIC and BIC scores for each model

data1["AIC"]=np.zeros(particpants)
data2['AIC']=np.zeros(particpants)
data3['AIC']=np.zeros(particpants)

data1['BIC']=np.zeros(particpants)
data2['BIC']=np.zeros(particpants)
data3['BIC']=np.zeros(particpants)

# loop through the data to calculate the NLL, AIC and BIC scores for each model and each participant
for i in range(0,particpants):

    data1.loc[i,"AIC"] = (2 * data1.loc[i,'NLL']) + (2 * p1)
    data2.loc[i,'AIC'] = (2 * data2.loc[i,'NLL']) + (2 * p2)
    data3.loc[i,'AIC'] = (2 * data3.loc[i,'NLL']) + (2 * p2)
    data1.loc[i,'BIC'] = (2 * data1.loc[i,'NLL']) + (p1 * np.log(n))
    data2.loc[i,'BIC'] = (2 * data2.loc[i,'NLL']) + (p2 * np.log(n))
    data3.loc[i,'BIC'] = (2 * data3.loc[i,'NLL']) + (p2 * np.log(n))

# sum up the scores for each model
sum_AIC1 = np.sum(data1.loc[:,"AIC"])
sum_AIC2 = np.sum(data2.loc[:,"AIC"])
sum_AIC3 = np.sum(data3.loc[:,"AIC"])
sum_BIC1 = np.sum(data1.loc[:,"BIC"])
sum_BIC2 = np.sum(data2.loc[:,"BIC"])
sum_BIC3 = np.sum(data3.loc[:,"BIC"])

# report the results
print("Model 1 AIC score:", sum_AIC1)
print("Model 2 AIC score:", sum_AIC2)
print("Model 3 AIC score:", sum_AIC3)
print("Model 1 BIC score:", sum_BIC1)
print("Model 2 BIC score:", sum_BIC2)
print("Model 3 BIC score:", sum_BIC3)




# choose the "best" model
if sum_AIC1 < sum_AIC2:
    if sum_AIC1< sum_AIC3:
        print("For simulated data Model 1 is the best model according to AIC")
    else:
        print("For simulated data Model 3 is the best model according to AIC")
elif sum_AIC2< sum_AIC3:
        print("For simulated datanModel 2 is the best model according to AIC")
else:
        print("For simulated data Model 3 is the best model according to AIC")

if sum_BIC1 < sum_BIC2:
    if sum_BIC1< sum_BIC3:
        print("For simulated data Model 1 is the best model according to BIC")
    else:
        print("For simulated data Model 3 is the best model according to BIC")
elif sum_BIC2< sum_BIC3:
        print("For simulated data Model 2 is the best model according to BIC")
else:
        print("For simulated data Model 3 is the best model according to BIC")

# split the data into high anxious and low anxious groups
high_anxious_m1 = data1.iloc[:25,:]
low_anxious_m1 = data1.iloc[26:,:]
high_anxious_m2 = data2.iloc[:25,:]
low_anxious_m2 = data2.iloc[26:,:]
high_anxious_m3 = data3.iloc[:25,:]
low_anxious_m3 = data3.iloc[26:,:]
# create the figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# plot the AIC scores
ax[0].scatter(np.arange(1,26), high_anxious_m1['AIC'], c='red', label='High Anxious, Model 1')
ax[0].scatter(np.arange(1,26), high_anxious_m2['AIC'], c='red', marker='x', label='High Anxious, Model 2')
ax[0].scatter(np.arange(1,26), high_anxious_m3['AIC'], c='red', marker='v', label='High Anxious, Model 3')
ax[0].scatter(np.arange(26,50), low_anxious_m1['AIC'], c='blue', label='Low Anxious, Model 1')
ax[0].scatter(np.arange(26,50) ,low_anxious_m2['AIC'], c='blue', marker='x', label='Low Anxious, Model 2')
ax[0].scatter(np.arange(26,50) ,low_anxious_m3['AIC'], c='blue', marker='v', label='Low Anxious, Model 3')
ax[0].set_xlabel('Particpants')
ax[0].set_ylabel('AIC Score')
ax[0].set_title('AIC Scores')
ax[0].legend()

ax[1].scatter(np.arange(1,26), high_anxious_m1['BIC'], c='red', label='High Anxious, Model 1')
ax[1].scatter(np.arange(1,26), high_anxious_m2['BIC'], c='red', marker='x', label='High Anxious, Model 2')
ax[1].scatter(np.arange(1,26), high_anxious_m3['BIC'], c='red', marker='v', label='High Anxious, Model 3')
ax[1].scatter(np.arange(26,50), low_anxious_m1['BIC'], c='blue', label='Low Anxious, Model 1')
ax[1].scatter(np.arange(26,50), low_anxious_m2['BIC'], c='blue', marker='x', label='Low Anxious, Model 2')
ax[1].scatter(np.arange(26,50), low_anxious_m3['BIC'], c='blue', marker='v', label='Low Anxious, Model 3')

ax[1].set_xlabel('Partcipants ')
ax[1].set_ylabel('BIC Score')
ax[1].set_title('BIC Scores')
ax[1].legend()

plt.show()




sum_AIC1_high = np.sum(high_anxious_m1.loc[:25, "AIC"])
sum_AIC2_high = np.sum(high_anxious_m2.loc[:25, "AIC"])
sum_AIC3_high = np.sum(high_anxious_m3.loc[:25, "AIC"])
sum_BIC1_high = np.sum(high_anxious_m1.loc[:25, "BIC"])
sum_BIC2_high = np.sum(high_anxious_m2.loc[:25, "BIC"])
sum_BIC3_high = np.sum(high_anxious_m3.loc[:25, "BIC"])

sum_AIC1_low = np.sum(low_anxious_m1.loc[26:, "AIC"])
sum_AIC2_low = np.sum(low_anxious_m2.loc[26:, "AIC"])
sum_AIC3_low = np.sum(low_anxious_m3.loc[26:, "AIC"])
sum_BIC1_low = np.sum(low_anxious_m1.loc[26:, "BIC"])
sum_BIC2_low = np.sum(low_anxious_m2.loc[26:, "BIC"])
sum_BIC3_low = np.sum(low_anxious_m3.loc[26:, "BIC"])

print("Sum of AIC1 for all participants: ", sum_AIC1)
print("Sum of AIC2 for all participants: ", sum_AIC2)
print("Sum of AIC3 for all participants: ", sum_AIC3)
print("Sum of BIC1 for all participants: ", sum_BIC1)
print("Sum of BIC2 for all participants: ", sum_BIC2)
print("Sum of BIC3 for all participants: ", sum_BIC3)

print("Sum of AIC1 for high anxious participants: ", sum_AIC1_high)
print("Sum of AIC2 for high anxious participants: ", sum_AIC2_high)
print("Sum of AIC3 for high anxious participants: ", sum_AIC3_high)

print("Sum of BIC1 for high anxious participants: ", sum_BIC1_high)
print("Sum of BIC2 for high anxious participants: ", sum_BIC2_high)
print("Sum of BIC3 for high anxious participants: ", sum_BIC3_high)

print("Sum of AIC1 for low anxious participants: ", sum_AIC1_low)
print("Sum of AIC2 for low anxious participants: ", sum_AIC2_low)
print("Sum of AIC3 for low anxious participants: ", sum_AIC3_low)

print("Sum of BIC1 for low anxious participants: ", sum_BIC1_low)
print("Sum of BIC2 for low anxious participants: ", sum_BIC2_low)
print("Sum of BIC3 for low anxious participants: ", sum_BIC3_low)






###################
# confusion Matrix#
####################

from sklearn.metrics import confusion_matrix
actual_values = Behaviour_Data.iloc[:, 1:161].applymap(lambda x: x[1]).values.ravel()


predicted_values_m1 = data1.iloc[:, 1:161].applymap(lambda x: x[1]).values.ravel()
conf_matrix_model1=confusion_matrix(actual_values, predicted_values_m1)


predicted_values_m2 = data2.iloc[:, 1:161].applymap(lambda x: x[1]).values.ravel()
conf_matrix_model2=confusion_matrix(actual_values, predicted_values_m2)



predicted_values_m3 = data3.iloc[:, 1:161].applymap(lambda x: x[1]).values.ravel()
conf_matrix_model3=confusion_matrix(actual_values, predicted_values_m3)



sns.heatmap(conf_matrix_model1, annot=True, fmt='d')
plt.show()

sns.heatmap(conf_matrix_model2, annot=True, fmt='d')
plt.show()
sns.heatmap(conf_matrix_model3, annot=True, fmt='d')
plt.show()

def precision(conf_matrix):
    tp = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    return tp / (tp + fp)

def accuracy(conf_matrix):
    tp = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    tn = conf_matrix[1][1]
    fn = conf_matrix[1][0]
    return (tp + tn) / (tp + tn + fp + fn)

def f1_score(confusion_matrix):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1


# Model 1
conf_matrix1 = [[3285, 1638], [1965, 1112]]
precision1 = precision(conf_matrix1)
accuracy1 = accuracy(conf_matrix1)
F1_Model1= f1_score(conf_matrix1)

# Model 2
conf_matrix2 = [[3260, 1663], [1896, 1181]]
precision2 = precision(conf_matrix2)
accuracy2 = accuracy(conf_matrix2)
F1_Model2= f1_score(conf_matrix2)

# Model 3
conf_matrix3 = [[3327, 1596], [2019, 1058]]
precision3 = precision(conf_matrix3)
accuracy3 = accuracy(conf_matrix3)
F1_Model3= f1_score(conf_matrix3)

print("Model 1:")
print("Precision:", precision1)
print("Accuracy:", accuracy1)
print("F1:", F1_Model1)
print("\nModel 2:")
print("Precision:", precision2)
print("Accuracy:", accuracy2)
print("F1:", F1_Model2)
print("\nModel 3:")
print("Precision:", precision3)
print("Accuracy:", accuracy3)
print("F1:", F1_Model3)
