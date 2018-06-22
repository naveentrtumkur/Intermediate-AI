# Implementation of Viterbi Algorithm for Hidden Markov Model
# I have used the example figure(right) given in concept check question-4 as a reference.

# Define the states required.
states = ('1', '2','3')
 
# Define the start probability 
start_probability = {'1': 0.5, '2': 0.2, '3': 0.3}

# Define the transition probability 
transition_probability = {
   '1' : {'1': 0.6, '2': 0.2,'3': 0.2 },
   '2' :  {'1': 0.5, '2': 0.3, '3': 0.2 },
   '3' : {'1': 0.4,'2': 0.1 ,'3': 0.5 }
   }

# Define the observations 
observ = ('Up', 'Up', 'Up')
   
# Evidence probability 
evidence_prob = {
   '1' : {'Up': 0.7, 'Down': 0.1, 'Unchanged': 0.2},
   '2' : {'Up': 0.1, 'Down': 0.6, 'Unchanged': 0.3},
   '3' : {'Up': 0.3, 'Down': 0.3, 'Unchanged': 0.4}
   }


# Function which implements viterbi_algorithm given the observ, states, start_prob, trans_prob and evidence.
def Viterbi_algorithm(observations, states, start_prob, trans_prob, evidence):
	path = { state:[] for state in states} #Creating a dictionary and initialising it.
	cur_prob = {} # Initializing current probability to empty dictionary.
	for state in states:
		cur_prob[state] = start_prob[state]*evidence[state][observations[0]] #Calculate the current probability values.
		#print cur_prob[state]
	for i in xrange(1, len(observations)):
		last_prob = cur_prob
		cur_prob = {}
		for curr_state in states:
			# below is the recurrence relation to compute max_prob
			maxProb, last_state = max(((last_prob[last_state]*trans_prob[last_state][curr_state]*evidence[curr_state][observations[i]], last_state) 
				                       for last_state in states))
			cur_prob[curr_state] = maxProb
			path[curr_state].append(last_state)

	#Find the maximum Probability value.
	maxProb = -1
	maxPath = None
	# In each state find the max prob and return that.
	for state in states:
		path[state].append(state)
		if cur_prob[state] > maxProb:
			#print maxPath
			maxPath = path[state]
			maxProb = cur_prob[state]
			#print maxProb
	return maxPath

# Call the main function.
if __name__ == '__main__':
	# Print the result after calling the viterbi algorithm.
	# i.e a list woukd be returned you can print that.
	viterbi_res = Viterbi_algorithm(observ, states, start_probability, transition_probability, evidence_prob)
    	print "Predicted Hidden States are:",viterbi_res
