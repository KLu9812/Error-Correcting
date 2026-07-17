import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from typing import Sequence

mapping = {'yes': 0, 'no': 1}

def reading_csv(model_list, m=1000):
    X = np.zeros((len(model_list)*5, m))
    for j,model in enumerate(model_list):
        for i in range(5):
            data = pd.read_csv(model + "cladderresponses" + str(i + 1) + "_filtered.csv")
            X[j*5+i] = data['answers'].map(mapping).to_numpy()

    file_path = "./cladder_info.csv"
    data = pd.read_csv(file_path)
    Y = data['correct answers'].map(mapping).to_numpy()
            
    return X.T, Y

def top_k_diversity_threshold(model_list, index_train, folder_number, k=5, tau=0):

    X,Y = reading_csv(model_list)
    X_train = X[index_train]
    Y_train = Y[index_train]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train)
    Xv = X_train[valid]
    Yv = Y_train[valid]

    m,N = Xv.shape
    selected_agents = []
    
    agent_correct_count = np.sum(Xv.T == Yv, axis=1)
    
    first_agent = np.argmax(agent_correct_count)
    selected_agents.append(first_agent)
    
    first_agent_tasks = set(np.where(Xv.T[first_agent] == Yv)[0])
    
    for _ in range(k-1):
        max_disagreement = -1
        next_agent = -1
        
        # Find the agent that has answered 1 on at least tau fraction of the tasks
        for i in range(N):
            if i in selected_agents:
                continue
            
            # Check if agent i answered 1 on at least tau fraction of tasks
            if np.mean(Xv.T[i] == Yv) < tau:
                continue
            
            # Calculate disagreement with already selected agents
            disagreement = 0
            for selected_agent in selected_agents:
                # Count the number of tasks where the labels disagree
                disagreement += np.sum(Xv.T[i] != Xv.T[selected_agent])
            
            # Select the agent with maximum disagreement
            if disagreement > max_disagreement:
                max_disagreement = disagreement
                next_agent = i
                
        # If no agent satisfies the condition, stop early
        if next_agent == -1:
            print('bar is too high')
        
        selected_agents.append(next_agent)

    agent_list = np.array([model+str(i) for model in model_list for i in range(1,6)])
    return agent_list[np.array(selected_agents)]

def accuracy(Y, X):
    mask = ~np.isnan(X) & ~np.isnan(Y)
    return float(np.mean(X[mask] == Y[mask]))

def MOA(model_list, index_train, fold_number, k = 5):
    X,Y = reading_csv(model_list)
    X_train = X[index_train]
    Y_train = Y[index_train]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train)
    Xv = X_train[valid]
    Yv = Y_train[valid]

    selected_agents = []
    for j,model in enumerate(model_list):
        acc_independent = []
        for i in range(5):
            acc_independent.append(accuracy(Yv, Xv[:,j*5+i]))
        s = np.argsort(acc_independent)[-1]
        selected_agents.append(j*5+s)

    agent_list = np.array([model+str(i) for model in model_list for i in range(1,6)])
    return agent_list[np.array(selected_agents)]

def evaluate_decision_tree(Xv, Yv, agent_indices, n_splits=5):

    if len(agent_indices) == 0:
        most_common_label = np.bincount(Yv).argmax()  # The most frequent label
        accuracy = np.mean(Yv == most_common_label)  # Accuracy of guessing the most frequent label
        return accuracy

    # Select the features corresponding to the agents
    agent_indices = np.array(agent_indices)
    X_selected = Xv[:, agent_indices]

    accuracies = []

    # Perform n_splits cross-validation
    for _ in range(n_splits):
        # Perform train-test split with 80% for training and 20% for testing
        X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Yv, test_size=0.2, random_state=None)

        # Train a Decision Tree classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)

    # Return the average accuracy across all splits
    average_accuracy = np.mean(accuracies)
    return average_accuracy

def compute_conditioned_shapley(Xv, Yv, S_cond, k, i, T=100):
    N = Xv.shape[1]

    # Select the features corresponding to the conditioned set S_cond
    X_cond = Xv[:, S_cond]

    # Initialize a list to store the marginal contributions
    contributions = []

    # Set of remaining agents (not in S_cond)
    remaining_agents = [idx for idx in range(N) if idx not in S_cond+[i]]

    # Iterate T times to simulate different permutations
    for _ in range(T):
        S_tmp = list(np.random.choice(remaining_agents, size=k-len(S_cond), replace=False))

        accuracy_tmp = evaluate_decision_tree(Xv, Yv, S_cond + S_tmp)

        S_tmp[-1] = i
        accuracy_with_i = evaluate_decision_tree(Xv, Yv, S_cond + S_tmp)

        # Compute the marginal contribution for this permutation
        s = len(S_cond + S_tmp)
        # weight = factorial(s) * factorial(N - s - 1) / factorial(N)
        marginal_contribution = accuracy_with_i - accuracy_tmp
        contributions.append(marginal_contribution)

    # Return the average marginal contribution as the Shapley value
    shapley_value = np.mean(contributions)
    return shapley_value

def Greedy_marginal_Shapley_DT(model_list, index_train, fold_number, k = 5, T=100):

    X,Y = reading_csv(model_list)
    X_train = X[index_train]
    Y_train = Y[index_train]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train)
    Xv = X_train[valid]
    Yv = Y_train[valid]

    # Initialize variables to track the agent with the largest Shapley value
    selected_agents = []
    marginal_shapley = []

    # Iterate over all agents (features)
    # for j in range(Xv.shape[1]):
    for j in range(k):
        remaining_agents = [idx for idx in range(Xv.shape[1]) if idx not in selected_agents]
        largest_shapley_value = -np.inf
        for i in remaining_agents:
            shapley_value = compute_conditioned_shapley(Xv, Yv, selected_agents, k, i, T)

            # Update if this agent has the largest Shapley value
            if shapley_value > largest_shapley_value:
                largest_shapley_value = shapley_value
                agent_with_largest_shapley = i
        selected_agents.append(agent_with_largest_shapley)
        marginal_shapley.append(largest_shapley_value)
        if largest_shapley_value < 0:
            break
    agent_list = np.array([model+str(i) for model in model_list for i in range(1,6)])
    return agent_list[np.array(selected_agents)]

"""Learn Q from data"""
# input m*k matrix of proposers' answers, m-length vector of summarizer's output, and m-length ground truth vector
# all should be training data, answers should be converted to int
def learn_Q_fine(proposer_output, summarizer_ouput, Y):
    m,k = proposer_output.shape
    correct = {}
    total = {}
    for i in range(m):
        ans_ct = np.zeros(5) # frequency of each answer
        for j in range(k):
            ans_ct[int(proposer_output[i][j])] += 1
        key = [int(ans_ct[Y[i]]), []]
        for t in range(5):
            if t != Y[i] and ans_ct[t] > 0:
                key[1].append(int(ans_ct[t]))
        key[1] = tuple(sorted([x for x in key[1] if x != 0], reverse=True))
        key = tuple(key)

        if key not in correct.keys():
            if summarizer_ouput[i] == Y[i]:
                correct[key] = 1
            else:
                correct[key] = 0
            total[key] = 1
        else:
            if summarizer_ouput[i] == Y[i]:
                correct[key] += 1
            total[key] += 1
    Q = {}
    for s in correct.keys():
        Q[s] = correct[s]/total[s]
    return Q


def random_flip(a, t, seed=None):
    """
    Flip each 0/1 entry of array `a` with probability `t`.
    Works for int/float 0-1 arrays and bool arrays.
    """
    a = np.asarray(a)
    rng = np.random.default_rng(seed)
    flips = rng.random(a.shape) < t
    if a.dtype == bool:
        return np.logical_xor(a, flips)
    else:
        return np.where(flips, 1 - a, a)

def partitions_by_size(k: int):
    """Yield all unique partitions of k with positive integers,
    ordered by partition size (1 part, 2 parts, ..., k parts)."""
    if k < 1:
        return
    def exact_parts(n, m, max_part):
        # partitions of n into exactly m parts, each ≤ max_part, non-increasing
        if m == 1:
            if 1 <= n <= max_part:
                yield (n,)
            return
        max_first = min(max_part, n - (m - 1))            # leave ≥1 for each remaining part
        min_first = max(1, (n + m - 1) // m)              # ceil(n/m) pruning
        for first in range(max_first, min_first - 1, -1):
            for rest in exact_parts(n - first, m - 1, first):
                yield (first,) + rest

    for m in range(1, k + 1):
        yield from exact_parts(k, m, k)

def frequency_of_partitions_fine(labels_matrix, Y, agent_indices):
    agent_indices = np.array(agent_indices)
    # Extract the labels for the specified agents
    selected_labels = labels_matrix[agent_indices, :]  # (n x m) matrix for the selected agents
    k,m = selected_labels.shape

    # For each task, count how many agents labeled it as 1
    frequency = {}
    for i in range(m):
        ans_ct = np.zeros(5) # frequency of each answer
        for j in range(k):
            ans_ct[int(selected_labels[j][i])] += 1
        key = [int(ans_ct[Y[i]]), []]
        for t in range(5):
            if t != Y[i] and ans_ct[t] > 0:
                key[1].append(int(ans_ct[t]))
        key[1] = tuple(sorted([x for x in key[1] if x != 0], reverse=True))
        key = tuple(key)

        if key not in frequency.keys():
            frequency[key] = 1
        else:
            frequency[key] += 1

    return frequency

found_Q = {(5, ()):0.872,
           (0, (5,)):0.186,
           (1, (4,)):0.358,
           (2, (3,)):0.552,
           (3, (2,)):0.65,
           (4, (1,)):0.754}

def compute_conditioned_shapley_fine_Q(Xv, Yv, Q, S_cond, i, k=5, T=10):
    X_cond = Xv[:, S_cond]

    # Initialize a list to store the marginal contributions
    contributions = []

    # Set of remaining agents (not in S_cond)
    remaining_agents = [idx for idx in range(Xv.shape[1]) if idx not in S_cond+[i]]

    # Iterate T times to simulate different permutations
    for _ in range(T):
        S_tmp = list(np.random.choice(remaining_agents, size=k-len(S_cond), replace=False))

        pi = frequency_of_partitions_fine(Xv.T, Yv, S_cond+S_tmp)
        accuracy_without_i = sum(pi[key] * Q[key] for key in pi)

        # Compute accuracy using S_cond + S_tmp + i (including agent i)
        S_tmp[-1] = i
        pi = frequency_of_partitions_fine(Xv.T, Yv, S_cond+S_tmp)
        accuracy_with_i = sum(pi[key] * Q[key] for key in pi)

        # Compute the marginal contribution for this permutation
        marginal_contribution = accuracy_with_i - accuracy_without_i
        contributions.append(marginal_contribution)

    # Return the average marginal contribution as the Shapley value
    shapley_value = np.mean(contributions)
    return shapley_value

def use_all(model_list, index_train, fold_number, k = 5):
    proposer_list = []
    for model in model_list:
        for i in range(1, 6):
            proposer_list.append(model + str(i))
    return proposer_list

def get_q(fold_number, CROSSES = 5, k =5):
    df = pd.read_csv("AceReason-Nemotron-14B" + str(k) + "cladderqdata.csv")
    full_length = len(df["answers"])
    train_range = list(range(0, fold_number * (full_length // CROSSES))) + list(range((fold_number + 1) * (full_length // CROSSES), full_length))
    train_range = train_range[len(train_range)//2:]
    comb_total = {}
    comb_correct = {}
    for index in train_range:
        try:
            comb_total[df["comb"][index]] += 1
        except Exception:
            comb_total[df["comb"][index]] = 1
            comb_correct[df["comb"][index]] = 0
        if df["answers"][index] == df["correct answers"][index]:
            comb_correct[df["comb"][index]] += 1
    return_Q = {}
    for key in comb_total.keys():
        acc = comb_correct[key] / comb_total[key]
        key_string = key.replace(" ", "")[1: - 1]
        values = key_string.split(",")
        if values[-1] == '':
            values = values[:-1]
        values = [int(v) for v in values]
        full_key = (values[0], tuple([]))
        if len(values) > 1:
            full_key = (values[0], tuple(values[1:]))
        return_Q[full_key] = acc
    return return_Q

def Greedy_marginal_Shapley_fine_Q(model_list, index_train, fold_number, k = 5, Q = found_Q, T=10):

    X,Y = reading_csv(model_list)
    X_train = X[index_train[:len(index_train)//2]]
    Y_train = Y[index_train[:len(index_train)//2]]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train)
    Xv = X_train[valid]
    Yv = Y_train[valid]

    selected_agents = []
    marginal_shapley = []
    
    Q = get_q(fold_number, k=k)

    # Iterate over all agents (features)
    # for j in range(Xv.shape[1]):
    for j in range(k):
        remaining_agents = [idx for idx in range(Xv.shape[1]) if idx not in selected_agents]
        largest_shapley_value = -np.inf
        for i in remaining_agents:
            shapley_value = compute_conditioned_shapley_fine_Q(Xv, Yv, Q, selected_agents, i, k, T)

            # Update if this agent has the largest Shapley value
            if shapley_value > largest_shapley_value:
                largest_shapley_value = shapley_value
                agent_with_largest_shapley = i
        selected_agents.append(agent_with_largest_shapley)
        marginal_shapley.append(largest_shapley_value)
        if largest_shapley_value < 0:
            break

    agent_list = np.array([model+str(i) for model in model_list for i in range(1,6)])

    return agent_list[np.array(selected_agents)]

def top_k_accuracy(model_list, index_train, fold_number, k=5):
    X,Y = reading_csv(model_list)
    X_train = X[index_train]
    Y_train = Y[index_train]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train)
    Xv = X_train[valid]
    Yv = Y_train[valid]

    
    acc_independent = []
    for i in range(Xv.shape[1]):
        acc_independent.append(accuracy(Yv, Xv[:,i]))

    agent_list = np.array([model+str(i) for model in model_list for i in range(1,6)])
    return agent_list[np.argsort(acc_independent)[-k:]]

def top_model(model_list, index_train, folder_number, k=5):
    X,Y = reading_csv(model_list)
    X_train = X[index_train]
    Y_train = Y[index_train]
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y_train)
    Xv = X_train[valid]
    Yv = Y_train[valid]

    acc_model = []
    for j,model in enumerate(model_list):
        acc_independent = []
        for i in range(5):
            acc_independent.append(accuracy(Yv, Xv[:,j*5+i]))
        acc_model.append(np.mean(acc_independent))
    selected_model = np.argsort(acc_model)[-1]
    selected_agents = [selected_model*5+i for i in range(5)]
    
    agent_list = np.array([model+str(i) for model in model_list for i in range(1,6)])
    return agent_list[np.array(selected_agents)]

def rateresponse(model_list, index_train, fold_number, k=5):
    df = pd.read_csv("gpt52claddernamedrates.csv")
    all_proposers = []
    for model in model_list:
        for i in range(1, 6):
            all_proposers.append(model + str(i))
    score_pairs = []
    for proposer in all_proposers:
        score_pairs.append((proposer, sum(df[proposer][index_train])))
    score_pairs.sort(key = lambda x: x[1])
    return_proposers = []
    for i in range(k):
        return_proposers.append(score_pairs[-i-1][0])
    return return_proposers
