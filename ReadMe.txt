There are two programs and one written homework.

Steps/Command to execute the programming assignment.
- Make sure all the libraries are installed. Use python2 or python3 to test the code.

Program-1) Execute the first program with the following commad:-

bash-3.2$ python3 coding_1_cnn.py 

Sample Output:

Vertical mask= 
 [[-1  0  1]
 [-2  0  2]
 [-1  0  1]]
Horizontal mask= 
 [[ 1  2  1]
 [ 0  0  0]
 [-1 -2 -1]]

Output value after one pass horizontal calculation..
[-47, -62, -54, -61, -57]
[2, 5, 12, 23, 20]
[-1, 16, 8, 5, 19]
[-5, -6, -11, -15, -8]
[48, 46, 46, 56, 38]
Final horizontal= [[-47, -62, -54, -61, -57], [2, 5, 12, 23, 20], [-1, 16, 8, 5, 19], [-5, -6, -11, -15, -8], [48, 46, 46, 56, 38]]

Output value after one pass vertical calculation..
[33, 50, 37, 27, 20]
[-10, -15, -22, -28, -18]
[18, 8, 16, 35, 30]
[23, 29, 13, 1, 0]
[-51, -58, -53, -62, -50]
Final vertical= [[33, 50, 37, 27, 20], [-10, -15, -22, -28, -18], [18, 8, 16, 35, 30], [23, 29, 13, 1, 0], [-51, -58, -53, -62, -50]]
bash-3.2$ 


Program-2) Execute the second program with the following command.

bash-3.2$ python coding_2_train.py
train= [-0.18  0.18  0.09]
bash-3.2$ 

- The output can be clearly viewed from the graph.