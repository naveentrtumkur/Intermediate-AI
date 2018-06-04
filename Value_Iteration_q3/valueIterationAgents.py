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


import mdp, util

from learningAgents import ValueEstimationAgent

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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #print "Final state data"
        #print self.mdp.getPossibleActions((3,1))
        #print self.mdp.getTransitionStatesAndProbs((3,1),"exit")
        #print "Final state data"
        
        for i in range(self.iterations):
            
            newVal = self.values.copy()
            
            for state in self.mdp.getStates():
                #print "state is "+ str(state)
                if (mdp.isTerminal(state)):
                    continue
                    
                possibleActions = self.mdp.getPossibleActions(state)
                #print possibleActions
                maxVal=-999
                
                for action in possibleActions:
                    #print action
                    val = self.computeQValueFromValues(state,action)
                    #print val
                    
                    if val > maxVal:
                        maxVal = val
                newVal[state] = maxVal
            
            self.values = newVal.copy()
            #print self.values
        

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
        
        sum = 0 #Variable to calculte the max V-val
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
          #print nextState
          #print prob
          sum += prob * (self.mdp.getReward(state,action,nextState) + (self.discount*self.values[nextState]))
        return sum #Return the calculated sum value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
       
        if (self.mdp.isTerminal(state)): #Check for terminal condition
            return None
        else:
            possible_actions =self.mdp.getPossibleActions(state) #If not get the possible actions
            
            maxVal=-9999
            best_action="north" #Initialize best_action to north
            
            #Find the best action from all the possible actions and return that action
            for a in possible_actions:
                val = self.computeQValueFromValues(state,a)
                
                if val > maxVal:
                    maxVal = val
                    best_action = a
                    
            return best_action
        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
