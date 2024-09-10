# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.

### Actions
The agent can take two actions:

R: Move right.
L: Move left.

### Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

## POLICY ITERATION ALGORITHM
### Step 1: 
Start with a random policy and value function.
### Step 2: 
Compute the value of the current policy.
### Step 3:
Make the policy greedy with respect to the value function.
### Step 4: 
Refine the policy until it converges.
### Step 5: 
The final policy is the optimal solution.

## POLICY IMPROVEMENT FUNCTION
### Name: Ritika s
### Register Number: 212221240046
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] *  (not done))
                new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
# Finding the improved policy
pi_2 = policy_improvement(V1, P)
print('Name: Ritika S Register Number: 212221240046')
print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))
# Finding the value function for the improved policy
V2 = policy_evaluation(pi_2, P)
print('Name: Ritika S Register Number: 212221240046')
print_state_value_function(V2, P, n_cols=7, prec=5)
# comparing the initial and the improved policy
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```

## POLICY ITERATION FUNCTION
### Name: Ritika S
### Register Number: 212221240046
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
    while True:
        old_pi = {s:pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        if old_pi == {s:pi(s) for s in range(len(P))}:
          break
    return V, pi
optimal_V, optimal_pi = policy_iteration(P)
print('Name: Ritika S Register Number: 212221240046')
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', '>'), n_cols=7)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, optimal_pi, goal_state=goal_state)*100,
    mean_return(env, optimal_pi)))
print_state_value_function(optimal_V, P, n_cols=7, prec=5)
```

## OUTPUT:
### optimal policy
![a1](https://github.com/user-attachments/assets/a7dd61fb-2d63-499e-aea5-e43bb47b23cb)
### optimal value function
![a2](https://github.com/user-attachments/assets/31e38822-80fe-4a40-89e5-5bbcd7d30e17)
### success rate for the optimal policy
![a3](https://github.com/user-attachments/assets/080af0dd-12ed-41ee-ba44-ad7dd4cd7d3a)



## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.
