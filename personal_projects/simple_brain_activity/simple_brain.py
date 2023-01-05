# Simple brain for nocturnal animal using Nengo
# To view functionality, open file in Nengo and change input sliders to view outputted behaviour

# Inputs: 1) brightness, 2) location of food
# Outputs: 1) direction of movement
# Behaviour: if dark, goes to food, otherwise go to starting location

import nengo

model = nengo.Network()
with model:
    brightness = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim_brightness = nengo.Node(0)
    nengo.Connection(stim_brightness, brightness)
    
    # food position relative to current position
    food = nengo.Ensemble(n_neurons=200, dimensions=2, radius=1)
    stim_food = nengo.Node([0,0])
    nengo.Connection(stim_food, food)
    
    motor = nengo.Ensemble(n_neurons=200, dimensions=2)
    
    move_food = nengo.Ensemble(n_neurons=200, dimensions=3, radius=1.5)
    nengo.Connection(brightness, move_food[0])
    nengo.Connection(food, move_food[1:])
    
    def food_gathering(x):
        brightness, food_x, food_y = x
        if brightness < 0:
            return food_x, food_y
        else:
            return 0, 0
    nengo.Connection(move_food, motor, function=food_gathering)
    
    position = nengo.Ensemble(n_neurons=500, dimensions=2)
    tau = 0.1
    nengo.Connection(position, position, synapse=tau)
    nengo.Connection(motor, position, transform=tau, synapse=tau)
    
    move_home = nengo.Ensemble(n_neurons=300, dimensions=3, radius=1.5)
    nengo.Connection(brightness, move_home[0])
    nengo.Connection(position, move_home[1:])

    def go_home(x):
        brightness, pos_x, pos_y = x
        if brightness >= 0:
            return -pos_x, -pos_y
        else:
            return 0, 0
    nengo.Connection(move_home, motor, function=go_home)