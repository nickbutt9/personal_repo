from TMNT import NnOps, PretrainedModel
import numpy as np

import pickle

# Load test paramters
with open('../Assignment_1/assignment-one-test-parameters.pkl', 'rb') as f:
    test_params = pickle.load(f)

w1, w2, w3, b1, b2, b3, inputs, targets = test_params['w1'], test_params['w2'], test_params['w3'], test_params['b1'], test_params['b2'], test_params['b3'], test_params['inputs'], test_params['targets']


# Training loop
EPOCHS = 5
batch_size = 1
num_batches = int(len(inputs)/batch_size)
lr = 0.01

train_loss = []
val_loss = []

# Indices for training data
idx = np.arange(len(inputs))

# Initialize model and gradient tape
net = PretrainedModel(w1, w2, w3, b1, b2, b3)
NnOps.tape = []

epoch_losses = []
train_loss = []

# Run model training for set number of epochs
for epoch in range(EPOCHS):
    np.random.shuffle(idx)
    net.train()
    NnOps.train()
    net.zero_grad()
    epoch_losses = []

    # Iterate through dataset
    for i in idx:
        input, target = inputs[i], targets[i]
        output = net.forward(input)
        loss = NnOps.mse(target, output)
        epoch_losses.append(loss)

        NnOps.backward()
        
        # Print gradients for first layer weights and biases
        if epoch == 0 and i == 0:
            print("First Layer Weight Gradients")
            print(net.fc1.A.grad)
            print("First Layer Bias Gradients")
            print(net.fc1.b.grad)

    train_loss.append(np.mean(epoch_losses))

    # Update parameters
    NnOps.step(net, lr, num_batches)
    net.zero_grad()
    NnOps.tape = []
    

    print("Epoch " + str(i + epoch) + " Training Loss: " + str(loss))

# Evaluate model performance
net.eval()
NnOps.val()

epoch_losses = []

for i in idx:
    input, target = inputs[i], targets[i]
    output = net.forward(input)
    loss = NnOps.mse(target, output)
    epoch_losses.append(loss)

    NnOps.backward()

train_loss.append(np.mean(epoch_losses))


# Plot Training Curve
import matplotlib.pyplot as plt

train_step = np.arange(EPOCHS + 1) + 1

plt.style.use('seaborn')
plt.plot(train_step, train_loss, label = 'Training Loss')
#plt.plot(val_loss, label = 'Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Training Curve for Model')
plt.legend()
plt.show()
