from functions import transition_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sympy as sp
from matplotlib import pyplot as plt

# Name of the column names of the csv file being read
CSV_COLUMN_NAMES = ['Date' 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Read a csv file using Pandas
data = pd.read_csv(r'C:\Users\Admin\Desktop\Data Sets\HMC.csv',
                    names=CSV_COLUMN_NAMES, header=0)

# Generate array for csv column Adjusted Closing price
data_array = data['Adj Close'].values
data_array = data_array.tolist()

# Create Test Data
data_test = []
for i in range(1, 8):
    data_test.append(data_array[-i])

data_array = data_array[:len(data_array)-7]

# Generate clean array of values from above
clean_data = []
for element in data_array:
    clean_data.append(element)

# Create a state matrix with 1 representing increase and 0 representing decrease
state_matrix = []
decrease_difference_list = []
increase_difference_list = []
for idx in range(0, len(clean_data)-1):
    if clean_data[idx + 1] > clean_data[idx]:
        state_matrix.append(1)
        increase_difference_list.append(clean_data[idx + 1] - clean_data[idx])
    elif clean_data[idx + 1] < clean_data[idx]:
        state_matrix.append(0)
        decrease_difference_list.append(clean_data[idx] - clean_data[idx + 1])

last_element = state_matrix[-1]

# Initial probability distribution
zero_sum = 0
one_sum = 0
for element in state_matrix:
    if element == 0:
        zero_sum += 1
    elif element == 1:
        one_sum += 1

zero_prob = zero_sum / len(state_matrix)
one_prob = one_sum / len(state_matrix)

# Transition matrix
t_matrix = transition_matrix(state_matrix)
print("Markov Probability (transition) matrix: ", t_matrix)

# Create a list of the values of the stocks after increasing or decreasing
increasing_states = []
decreasing_states = []
for idx in range(len(state_matrix)):
    if state_matrix[idx] == 0:
        decreasing_states.append(clean_data[idx + 1])
    elif state_matrix[idx] == 1:
        increasing_states.append(clean_data[idx + 1])

# Means and standard deviations
d_mean = -(np.mean(decrease_difference_list))
d_mean = d_mean.astype('float32')
d_std = np.std(decrease_difference_list)
d_std = d_std.astype('float32')
i_mean = np.mean(increase_difference_list)
i_mean = i_mean.astype('float32')
i_std = np.std(increase_difference_list)
i_std = i_std.astype('float32')

# Create matrix to be reduced in order to find steady state
pre_steady_matrix = sp.Matrix([[-(1 - t_matrix[0][0]), t_matrix[1][0], 0],
                               [t_matrix[0][1], -(1 - t_matrix[1][1]), 0],
                               [1, 1, 1]])
psm_reduced = pre_steady_matrix.rref()

# Steady State Matrix (IMPORTANT)
steady_state = sp.Matrix([[psm_reduced[0][2]],
                          [psm_reduced[0][5]]])
print("Steady state: ", steady_state)
print("Steady state proportion of decreases: ", psm_reduced[0][2])
print("Steady state proportion of increases: ", psm_reduced[0][5])

# Initial distribution calculation
initial_zero = t_matrix[last_element][0]
initial_one = t_matrix[last_element][1]
print("Initial probability of being zero: ", initial_zero)
print("Initial probability of being one: ", initial_one)

# Mean and Standard Deviation printout
print("Decreasing mean and std dev: ", d_mean, d_std)
print("Increasing mean and std dev: ", i_mean, i_std)

# Markov Model Prediction using TensorFlow by Google
tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[initial_zero, initial_one])
transition_distribution = tfd.Categorical(probs=[[t_matrix[0][0], t_matrix[0][1]],
                                                 [t_matrix[1][0], t_matrix[1][1]]])
observation_distribution = tfd.Normal(loc=[d_mean, i_mean], scale=[d_std, i_std])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor
# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
  print("Predicted increase (positive) or decrease (negative) for the next 7 time steps: ", mean.numpy())

last_data_element = clean_data[-1]
prediction_list = []
for element in mean.numpy():
    last_data_element += element
    prediction_list.append(last_data_element)

# Comparison of predictions to test data
rounded_predictions = [round(num, 2) for num in prediction_list]
rounded_test = [round(num, 2) for num in data_test]
print("Test Data: ", rounded_test)
print("Predictions: ", rounded_predictions)

# Evaluation of percent error of the prediction
percent_errors = []
for i in range(7):
    percent_errors.append(abs((rounded_predictions[i] - rounded_test[i]) / rounded_test[i]))
rounded_percent_errors = []
for i in range(7):
    rounded_percent_errors.append(round(percent_errors[i], 4))

print("Percent error at each time step: ", rounded_percent_errors)
average = round(sum(percent_errors) / len(percent_errors), 5) * 100
average = str(average)
print("Average percent error: " + average + "%")


# Plotting
x_values1 = []
for num in range(1, 8):
    x_values1.append(num)

x_values2 = []
for num in range(1, 8):
    x_values2.append(num)

for item in data_test:
    data_array.append(item)

plt.plot(x_values1, data_array[-7:], label = "Actual")
plt.plot(x_values2, rounded_predictions, label = "Predicted")
plt.ylim((28, 34))
plt.xlabel('Time Steps')
# Set the y axis label of the current axis.
plt.ylabel('Change in Price ($)')
# Set a title of the current axes.
plt.title('Actual vs. Predicted Change in Stock Price (Honda - 1 Year Daily)', y=1.07)
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()