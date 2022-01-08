# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from os import stat
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        for i in range(self.iterations):
            newValues = self.values.copy()
            allStates = self.mdp.getStates()
            for j in range(len(allStates)):
                state = allStates[j]
                if (not self.mdp.isTerminal(state)):
                    possibleActions = self.mdp.getPossibleActions(state)
                    maxQ = float('-inf')
                    for k in range(len(possibleActions)):
                        action = possibleActions[k]
                        maxQ = max(self.getQValue(state, action), maxQ)
                    newValues[state] = maxQ
            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        q = 0

        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for i in range(len(transitionStatesAndProbs)):
            newState = transitionStatesAndProbs[i][0]
            transitionProbability = transitionStatesAndProbs[i][1]
            reward = self.mdp.getReward(state, action, newState)
            q += transitionProbability * (reward + self.discount * self.getValue(newState))

        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        allActions = self.mdp.getPossibleActions(state)
        maxQ = float('-inf')
        bestAction = None
        for i in range(len(allActions)):
            action = allActions[i]
            q = self.getQValue(state, action)

            if (q > maxQ):
                maxQ = q
                bestAction = action
        return bestAction
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()

        for i in range(self.iterations):
            indexToUpdate = i % len(allStates)
            state = allStates[indexToUpdate]
            if (not self.mdp.isTerminal(state)):
                possibleActions = self.mdp.getPossibleActions(state)
                maxQ = float('-inf')
                for j in range(len(possibleActions)):
                    action = possibleActions[j]
                    maxQ = max(maxQ, self.getQValue(state, action))
                self.values[state] = maxQ

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        allStates = self.mdp.getStates()

        for i in range(len(allStates)):
            state = allStates[i]
            if (not self.mdp.isTerminal(state)):
                possibleActions = self.mdp.getPossibleActions(state)
                for j in range(len(possibleActions)):
                    action = possibleActions[j]
                    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for k in range(len(transitionStatesAndProbs)):
                        newState = transitionStatesAndProbs[k][0]
                        if newState in predecessors:
                            predecessors[newState].add(state)
                        else:
                            predecessors[newState] = {state}
        
        queue = util.PriorityQueue()

        for i in range(len(allStates)):
            state = allStates[i]
            if (not self.mdp.isTerminal(state)):
                possibleActions = self.mdp.getPossibleActions(state)
                maxQ = float('-inf')
                for j in range(len(possibleActions)):
                    action = possibleActions[j]
                    maxQ = max(self.computeQValueFromValues(state, action), maxQ)
                diff = abs(self.values[state] - maxQ)
                queue.update(state, -diff)
        
        for i in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            if (not self.mdp.isTerminal(state)):
                possibleActions = self.mdp.getPossibleActions(state)
                maxQ = float('-inf')
                for j in range(len(possibleActions)):
                    action = possibleActions[j]
                    maxQ = max(self.computeQValueFromValues(state, action), maxQ)
                self.values[state] = maxQ
            
            for predecessor in predecessors[state]:
                if (not self.mdp.isTerminal(predecessor)):
                    possibleActions = self.mdp.getPossibleActions(predecessor)
                    maxQ = float('-inf')
                    for k in range(len(possibleActions)):
                        action = possibleActions[k]
                        maxQ = max(self.computeQValueFromValues(predecessor, action), maxQ)
                    diff = abs(self.values[predecessor] - maxQ)

                    if self.theta < diff:
                        queue.update(predecessor, -diff)