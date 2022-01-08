# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        foodToBeEaten = newFood.asList()
        isAGhostScared = min(newScaredTimes) > 0

        positionOfGhosts = []
        for ghostNumber in range(len(newGhostStates)):
            ghost = newGhostStates[ghostNumber]
            ghostPosition = ghost.getPosition()
            positionOfGhosts.append(ghostPosition)

        if newPos in positionOfGhosts and not isAGhostScared:
            return -1

        if newPos in currentGameState.getFood().asList():
            return 1

        minDistanceToGhost = float('inf')
        for positionOfGhost in positionOfGhosts:
            distanceToGhost = util.manhattanDistance(newPos, positionOfGhost)
            minDistanceToGhost = min(minDistanceToGhost, distanceToGhost)

        minDistanceToFood = float('inf')
        for positionOfFood in foodToBeEaten:
            distanceToFood = util.manhattanDistance(newPos, positionOfFood)
            minDistanceToFood = min(minDistanceToFood, distanceToFood)

        return 1 / (minDistanceToFood) - 1 / (minDistanceToGhost)
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        numOfGhosts = gameState.getNumAgents() - 1
        pacManNumber = 0
        
        def maxValue(state, depthOfState):
            v = float('-inf')

            if isTerminalStateTest(state, depthOfState):
                return self.evaluationFunction(state)

            availableActionsForPacman = state.getLegalActions(pacManNumber)

            for act in availableActionsForPacman:
                firstGhostNumber = 1
                stateOfEnvironment = state.generateSuccessor(pacManNumber, act)
                v = max(v, minValue(stateOfEnvironment, depthOfState, firstGhostNumber))

            return v

        def minValue(state, depthOfState, ghostNumber):
            v = float('inf')

            if isTerminalStateTest(state, depthOfState):
                return self.evaluationFunction(state)

            availableActionsForGhost = state.getLegalActions(ghostNumber)

            for act in availableActionsForGhost:
                stateOfEnvironment = state.generateSuccessor(ghostNumber, act)

                if ghostNumber != numOfGhosts:
                    v = min(v, minValue(stateOfEnvironment, depthOfState, ghostNumber + 1))
                else:
                    v = min(v, maxValue(stateOfEnvironment, 1 + depthOfState))

            return v

        def isTerminalStateTest(state, depthOfState):
            if depthOfState == self.depth:
                return True
            
            if state.isWin() or state.isLose():
                return True

            return False

        firstGhostNumber = 1
        availableActionsForPacman = gameState.getLegalActions(pacManNumber)

        maxUtility = float('-inf')
        maxUtilityAction = availableActionsForPacman[0]

        for act in availableActionsForPacman:
            stateOfEnvironment = gameState.generateSuccessor(pacManNumber, act)
            if (minValue(stateOfEnvironment, 0, firstGhostNumber) >= maxUtility):
                maxUtility = minValue(stateOfEnvironment, 0, firstGhostNumber)
                maxUtilityAction = act
        
        return maxUtilityAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numOfGhosts = gameState.getNumAgents() - 1
        pacManNumber = 0
        
        def maxValue(state, depthOfState, alpha, beta):
            v = float('-inf')

            if isTerminalStateTest(state, depthOfState):
                return self.evaluationFunction(state)

            availableActionsForPacman = state.getLegalActions(pacManNumber)

            for act in availableActionsForPacman:
                firstGhostNumber = 1
                stateOfEnvironment = state.generateSuccessor(pacManNumber, act)
                v = max(v, minValue(stateOfEnvironment, depthOfState, firstGhostNumber, alpha, beta))

                if (v > beta):
                    return v

                alpha = max(alpha, v)

            return v

        def minValue(state, depthOfState, ghostNumber, alpha, beta):
            v = float('inf')

            if isTerminalStateTest(state, depthOfState):
                return self.evaluationFunction(state)

            availableActionsForGhost = state.getLegalActions(ghostNumber)

            for act in availableActionsForGhost:
                stateOfEnvironment = state.generateSuccessor(ghostNumber, act)

                if ghostNumber != numOfGhosts:
                    v = min(v, minValue(stateOfEnvironment, depthOfState, ghostNumber + 1, alpha, beta))
                else:
                    v = min(v, maxValue(stateOfEnvironment, 1 + depthOfState, alpha, beta))
                
                if (v < alpha):
                    return v
                
                beta = min(beta, v)

            return v

        def isTerminalStateTest(state, depthOfState):
            if depthOfState == self.depth:
                return True
            
            if state.isWin() or state.isLose():
                return True

            return False

        firstGhostNumber = 1
        availableActionsForPacman = gameState.getLegalActions(pacManNumber)

        maxUtility = float('-inf')
        maxUtilityAction = availableActionsForPacman[0]

        alpha = float('-inf')
        beta = float('inf')
        for act in availableActionsForPacman:
            stateOfEnvironment = gameState.generateSuccessor(pacManNumber, act)
            if (minValue(stateOfEnvironment, 0, firstGhostNumber, alpha, beta) >= maxUtility):
                maxUtility = minValue(stateOfEnvironment, 0, firstGhostNumber, alpha, beta)
                maxUtilityAction = act
            
            if maxUtility > beta:
                break

            alpha = max(alpha, minValue(stateOfEnvironment, 0, firstGhostNumber, alpha, beta))
        
        return maxUtilityAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numOfGhosts = gameState.getNumAgents() - 1
        pacManNumber = 0
        
        def maxValue(state, depthOfState):
            v = float('-inf')

            if isTerminalStateTest(state, depthOfState):
                return self.evaluationFunction(state)

            availableActionsForPacman = state.getLegalActions(pacManNumber)

            for act in availableActionsForPacman:
                firstGhostNumber = 1
                stateOfEnvironment = state.generateSuccessor(pacManNumber, act)
                v = max(v, expectedValue(stateOfEnvironment, depthOfState, firstGhostNumber))

            return v

        def expectedValue(state, depthOfState, ghostNumber):
            v = 0

            if isTerminalStateTest(state, depthOfState):
                return self.evaluationFunction(state)

            availableActionsForGhost = state.getLegalActions(ghostNumber)

            for act in availableActionsForGhost:
                stateOfEnvironment = state.generateSuccessor(ghostNumber, act)
                probabilityOfState = 1 / len(availableActionsForGhost) #assuming all actions have equal probabilities of occurring
                if ghostNumber != numOfGhosts:
                    v += probabilityOfState * expectedValue(stateOfEnvironment, depthOfState, ghostNumber + 1)
                else:
                    v += probabilityOfState * maxValue(stateOfEnvironment, 1 + depthOfState)

            return v

        def isTerminalStateTest(state, depthOfState):
            if depthOfState == self.depth:
                return True
            
            if state.isWin() or state.isLose():
                return True

            return False

        firstGhostNumber = 1
        availableActionsForPacman = gameState.getLegalActions(pacManNumber)

        maxUtility = float('-inf')
        maxUtilityAction = availableActionsForPacman[0]

        for act in availableActionsForPacman:
            stateOfEnvironment = gameState.generateSuccessor(pacManNumber, act)
            if (expectedValue(stateOfEnvironment, 0, firstGhostNumber) >= maxUtility):
                maxUtility = expectedValue(stateOfEnvironment, 0, firstGhostNumber)
                maxUtilityAction = act
        
        return maxUtilityAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    So, we need a linear evaluation of the form: f = w1*p1 + w2*p2 +...+wn*pn
    The parameters/variables chosen are: 1/minDistanceToFood, minDistanceToGhost, currentGameState.getScore(), 1/(len(foodToBeEaten) + 1), min(newScaredTimes)
    since we know that we want to maixmize all these parameters for pacman
    The weights have been generated based on trial and error. As we can see, the main/highest weight is on 1/minDistanceToFood since this is the main objective for pacman.
    Secondary goal is getting away from ghosts( 2nd largest weight is on minDistanceToGhost)
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    foodToBeEaten = newFood.asList()
    isAGhostScared = min(newScaredTimes) > 0

    positionOfGhosts = []
    for ghostNumber in range(len(newGhostStates)):
        ghost = newGhostStates[ghostNumber]
        ghostPosition = ghost.getPosition()
        positionOfGhosts.append(ghostPosition)

    if newPos in positionOfGhosts and not isAGhostScared:
        return -1

    if newPos in currentGameState.getFood().asList():
        return 1

    minDistanceToGhost = float('inf')
    for positionOfGhost in positionOfGhosts:
        distanceToGhost = util.manhattanDistance(newPos, positionOfGhost)
        minDistanceToGhost = min(minDistanceToGhost, distanceToGhost)
    
    minDistanceToFood = float('inf')
    for positionOfFood in foodToBeEaten:
        distanceToFood = util.manhattanDistance(newPos, positionOfFood)
        minDistanceToFood = min(minDistanceToFood, distanceToFood)

    coefficients = [ 95.2, 15.2, 12.3, 10.2, 5.2]
    parameters = [1/minDistanceToFood, minDistanceToGhost, currentGameState.getScore(), 1/(len(foodToBeEaten) + 1), min(newScaredTimes)]
    linearWeightedFunction = coefficients[0] * parameters[0] + coefficients[1] * parameters[1] + coefficients[2] * parameters[2] + coefficients[3] * parameters[3]  + coefficients[4] * parameters[4]
    return linearWeightedFunction
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
