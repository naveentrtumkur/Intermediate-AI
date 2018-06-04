# This is a script used to computed the V-values (V*)
# Specifically to question, we will be computing V18 and V19.
# 18 and 19 can be passed from the command-line


import numpy as np
import sys

values = np.zeros(3) #This is used to store the value in three states.

loop_count = int(sys.argv[1]) #Accept till where to loop from the user Example: 18 or 19
#print loop_count

# I will be harcoding the possible rewards and transition probability values
reward = np.array([[[1,0,0],[2,2,0]],[[1,1,0],[0,0,-10]],[[0,0,0],[0,0,0]]])
#print reward

#Below would be the transisition probability values
trans_prob = np.array([[[1.0,0,0],[0.5,0.5,0]],[[0.5,0.5,0],[0,0,1.0]],[[0,0,0],[0,0,0]]])
#print trans_prob
#print (trans_prob.shape)

#print trans_prob[0][:][:]
#print np.dot(trans_prob[0][:][:],np.transpose(reward[0][:][:]))

count = 0 #Initialize count to zero and loop till 18 to get V18 & loop till 19 to get V19

while count < loop_count:
    old_val = np.copy(values)
    for i in range(0,3):
        #print "old==",old_val
        vals = [] 
        for j in range(0,2):
            sum = 0
            for k in range(0,3):
                #print np.max(np.sum(np.dot(trans_prob[i][j][:],np.transpose(reward[i][j][:]))))
                sum += (trans_prob[i][j][k]*((reward[i][j][k])+ old_val[k]))
                #print "Sum using K=",trans_prob[i][j][k]*((reward[i][j][k]))
            #vals.append(np.sum(trans_prob[i][j][:]*((reward[i][j][:])+old_val[i])))
            vals.append(sum)
        values[i] = max(vals)
        #print "max=",max(vals) #You can print every max value
    count += 1 #Increment the counter value

print values
#End of the Program
