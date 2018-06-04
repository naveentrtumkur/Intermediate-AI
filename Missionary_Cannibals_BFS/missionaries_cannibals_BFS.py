#This program solves the classical Missionaries and Cannibals problem using breadth-first search technique

#Import the operator and Queue library required to solve this problem
import operator
import Queue

#Define the start state of the boat
Start = [3,3,1]

#Enumerate the values for wrongside, missionary, cannibal and boat.
LeftSide = 1
Miss = 0
Cann = 1
Boat = 2
#print Miss,Cann,Boat

#Sets the flag towards right (1=set=towards right, 0=false=towards left)
Towards_Right = 1

#calculation based on number of times boat moved
boat_count =0

#The list of possible actions that can be taken
#Logic: Number of missionaries >= cannibals
# I have hardcoded the possible actions that can be taken
actions = [[1,0,1],[2,0,1],[0,1,1],[0,2,1],[1,1,1]]

#The end goal of our search problem
goal = [0,0,0]

#Define a class State with all getters/setters to work through the problem.

class State:
	def __init__(self, value):
		self.state = value
		self.action = []
		self.depth = 0
		self.children = []
		self.parent = None
		self.expanded = 0
                self.boat_count = 0

        # A list of useful getters/setters which will be called at suitable places
	def set_expanded(self, expanded):
		self.expanded = expanded

	def get_expanded(self):
		return self.expanded

	def set_parentNode(self, parent):
		self.parent = parent

	def get_parentNode(self):
		return self.parent

	def set_action(self, action):
		self.action = action

	def get_state(self):
		return self.state

	def get_action(self):
		return self.action

	def set_depth(self, depth):
		self.depth = depth

	def get_depth(self):
		return self.depth

	def get_children(self):
		return self.children

	def valid_childs(self):
		# Goes through possible children and discards invalid states
		if len(self.children) == 0:
			return
		valid_childs = []
		for child in self.children:
                        #Check if the state is valid and append it to valid_childs list.
			if self.valid(child.state):
				valid_childs.append(child)
		self.children = valid_childs
		return self.children

	def valid(self, state):
                #For a valid state, Number of Cannibals <= Number of Missionaries
		if state[Cann] <= state[Miss] and state[Cann] <= 3 and state[Miss] <= 3:
			return True
		return False

def print_state(state, action, count):
	# Concatenates into proper output
	# state [0,0,0], action [0,0,0], # of nodes expanded
        #Towards_Right = 0
        if count%2 == 0:
	    return 'Left Status=(' + ','.join(map(str,state)) + ')'+', Right to Left Boat Contains=(' + ','.join(map(str,action))+')' 
	else:
            return 'Left Status=(' + ','.join(map(str,state)) + ')'+', Left to right Boat Contains=(' + ','.join(map(str,action))+')' 

def print_path(root, leaf):
	# Prints the path from root node to leaf node
	stack = [] #nodes from leaf to parent pushed onto this list
	node = leaf
        count = 0
	stack.append(print_state(leaf.get_state(), [0,0,0],count))
        count += 1
	while node != root: #Recurse from bottom to up node.
		stack.append(print_state(node.get_parentNode().get_state(), node.get_action(), count))
		node = node.get_parentNode()
                count += 1
	while stack: #The stack holds contents from root -> leaf from top of stack to bottom. Just  print it out.
		print stack.pop()
	return



def BFS_search(root):
	#Implementation of BFS Search to solve the problem.
        print "Cannibals and Missionary problem: Breadth First search implementation"
	isDone = None
	q = Queue.Queue() #Declare a queue
	q.put(root) #Add root to queue
	explored_states = set() #Maintain a set of explored states
	cur_depth = 0
	expanded = 0
	children = [] #A list of children (temp)
	while not(q.empty()):
		state = q.get()	#Remove the state from queue and expand it.

		# Expansion of set by addition of valid children
		if not(cur_depth == state.get_depth()):
			for child in children:
				if state.valid(child.get_state()):
					if child.get_state() == goal: #If goal is reached then end there
						isDone = child
					child.parent.children.append(child)
					child.set_expanded(expanded)
			if not(isDone == None):
					print_path(root, isDone) #If goal is reached, print out path and return
					return 
			children = []
			cur_depth += 1 #Increment the depth of the queue
                        #print "Len=",q.qsize()
                        #print "cur==",cur_depth

		# If invalid state, just skip and ignore
		if not state.valid(state.get_state()):
			continue

		# Based on actions, find the new states which can be reached
		for act in actions:
			child = None

			# Finds the child depending on boat is left or right
			if state.get_state()[Boat] == LeftSide:
			    child = State(map(operator.sub, state.get_state(), act))
			elif state.get_state()[Boat] != LeftSide:
			    child = State(map(operator.add, state.get_state(), act))
			child.set_action(act) #Sets the action to be taken

			# Keep list of children that are from new states
			if not(child == state) and child not in explored_states:
				expanded += 1 #Increment expanded counter
				child.set_depth(state.depth+1) #Update depth, parent and add to explored states
				child.set_parentNode(state)
				children.append(child)
				explored_states.add(child)
				q.put(child)

#Call the main function/subroutine
if __name__ == '__main__':
	root = State(Start)
	BFS_search(root)
