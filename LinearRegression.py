import tensorflow as tf

# Model parameters
theta1 = tf.Variable([.3], dtype=tf.float32)
theta2 = tf.Variable([-.3], dtype=tf.float32)

# input and output
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

# hypothesis for linear Model
hypothesis = theta1 * x + theta2

# calculate loss
squared_difference = tf.square(hypothesis - y)
loss = tf.reduce_sum(squared_difference)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

print(session.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

for i in range(1000):
    session.run(train, {x:x_train, y:y_train})

currentTheta1, currentTheta2, currentLoss = session.run([theta1, theta2, loss], {x:x_train, y:y_train})
print("currentTheta1: %s currentTheta2: %s currentLoss: %s" %(currentTheta1, currentTheta2, currentLoss))

# calculate for new value of X
x_Current = [5, 6, 7]
print(session.run(hypothesis, {x:x_Current}))
