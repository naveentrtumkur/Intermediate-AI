# Implementation of CNN which will compute the application of the 3x3 vertical and horizontal sobel masks
# to an input image of size 5x5x3 with zero-padding of size 1.
# We need to apply zero-padding of size 1 so that entire image's sobel masks are taken into consideration.

# Use numpy array to store and work with computation
import numpy as np
import matplotlib

# Function defintion to perform the padding required.
def pad_with(vect, p_width, iaxis, kwargs):
	pad_val = kwargs.get('padder',0)
	#Padding the outer layer of the vector with zeroes.
	vect[:p_width[0]] = pad_val
	vect[-p_width[1]:] = pad_val
	#vect = np.array[1:,3:]
	np.delete(vect,0,0)
	#print("vect=",vect)
	return vect


# I will be harcoding the input volume values (i.e of the image masks)
input_vol = np.array([[[3, 1, 3, 8, 2],[4,1,5,7,9],[2,1,4,5,0],[4,1,5,8,3],[3,1,4,7,2]],
					[[5, 4, 1, 3, 8],[4,9,1,4,7],[7,3,1,4,6],[8,4,1,5,2],[2,3,1,8,2]],
					[[2, 2, 3, 7, 3],[6,9,4,4,5],[1,1,1,1,1],[8,3,4,5,5],[7,2,3,1,4]]])

#print(input_vol)

#Apply padding. You can either hard code or use padding in numpy arrays to achieve this.
# Apply a padding of size 1 with zeroes around the input vector.
#input_pad = []
input_pad = np.pad(input_vol, 1, mode='constant')
#print(input_pad3[0][:])

# Remove the first and last vector.
input_pad = np.delete(input_pad,0,0)
input_pad = np.delete(input_pad,3,0)

#print("vector1: \n",input_pad[0][:]) 
#print("vector2: \n",input_pad[1][:])
#print("vector3: \n",input_pad[2][:])
#print("After padding applied \n", input_pad)


# Implementation of convolutional neural nets.

#Initialize the vertical mask
vertical_mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
print("Vertical mask= \n",vertical_mask) 

#Initialize teh horizontal mask
horizontal_mask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
print("Horizontal mask= \n",horizontal_mask)
	
# Compute the output vector by traversing the filter over input repeatedly.
# The depth is 2, so we should get the output of size-2.

# Apply one pass of convolution by doing the following:-
# 1. Take input image pixels
# 2. Use vertical_filter_mask to compute output1_values (size 2x2)
# 3. Use horizontal_filter_mask to compute output2_values (size 2x2)
# Stride is 1, Padding of 1 zero mask around input. 

out1 = np.multiply(input_pad[0][0][0:3], horizontal_mask)

#print("Input val",input_pad[0][0][0:3])
#print(input_pad[0][0][0])

#ex = np.array([1,2,3])
#ex = np.vstack([ex,[4,5,6]])
#ex = np.vstack([ex,[0,1,1]])
#print(ex)
#print(out1)



#print("Output of horizontal pass=",res_vals)
def cnn_one_pass_horizontal(inp, horiz_mask, stride):
	print("\nOutput value after one pass horizontal calculation..")
	mult_res = []
	res_vals = []
	for k in range(0,5):
		row_list = []
		for i in range(0,5):
			inter_sum = 0
			for j in range(0,3):

				first_inp = input_pad[j][k][i:i+3]
				first_inp = np.vstack([first_inp,input_pad[j][k+1][i:i+3]])
				first_inp = np.vstack([first_inp,input_pad[j][k+2][i:i+3]])

		#print("Sliced input array:\n", first_inp)
				inter_sum += np.sum(np.multiply(first_inp,horiz_mask))
		#mult_res.append()
		#print("\nMultiplied result=\n",inter_sum)
			row_list.append(inter_sum)
		print(row_list)
		res_vals.append(row_list)
	#final_horiz = res_vals
	return res_vals
	#print(res_vals)


def cnn_one_pass_vertical(inp, vert_mask, stride):
	print("\nOutput value after one pass vertical calculation..")

	mult_res = []
	res_vals = []
	for i in range(0,5):
		row_list = []
		for k in range(0,5):
			inter_sum = 0
			for j in range(0,3):

				first_inp = input_pad[j][k][i:i+3]
				first_inp = np.vstack([first_inp,input_pad[j][k+1][i:i+3]])
				first_inp = np.vstack([first_inp,input_pad[j][k+2][i:i+3]])

			#print("Sliced input array:\n", first_inp)
				inter_sum += np.sum(np.multiply(first_inp,vert_mask))
			#mult_res.append()
			#print("\nMultiplied result=\n",inter_sum)
			row_list.append(inter_sum)
		print(row_list)
		res_vals.append(row_list)	
	return res_vals
	#print(res_vals)


# Call the method to calculate the horizontal filtering.
final_horizontal = cnn_one_pass_horizontal(input_vol,horizontal_mask,1)

print("Final horizontal=",final_horizontal)

# call the method to calculate the vertical filtering.
final_vertical = cnn_one_pass_vertical(input_vol,vertical_mask,1)

print("Final vertical=",final_vertical)

# I have summed the output which we get after horizontal filtering and vertical filtering.
# Ideally it can be treated as 5x5x2 output.
