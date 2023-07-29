'''Define x as a variable with the initial value x0.
Set the loss function, y, equal to x multiplied by x. Do not make use of operator overloading.
Set the function to return the gradient of y with respect to x.'''


import tensorflow as tf

def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = tf.Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = x*x
    # Return the gradient of y with respect to x
	return tape.gradient(y,x).numpy()

# Compute and print gradien	ts at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
