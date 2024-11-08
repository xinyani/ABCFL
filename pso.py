import numpy as np
from fun import fu
n_particles = 5
n_iterations = 5

# Defines the cognitive and social constants
c1 = 2
c2 = 2


# Define the boundaries of the search space (min and max values for each parameter)
bounds = [(0.00001, 0.01), (4, 256),(1,10), (0.00001, 1)]

# Initializes the particles at random positions within the search space
particles = [np.random.uniform(low=bounds[i][0], high=bounds[i][1], size=n_particles) for i in range(len(bounds))]

# Initializes the velocities of the particles to zero
velocities = [np.zeros(n_particles) for _ in range(len(bounds))]


def position_to_params(particles, i):
    params_keys = ['lr', 'bb', 'ee', 'mm']

    params = {}
    for j, key in enumerate(params_keys):
        # Values are already initialized in the correct range, no need to scale
        value = particles[j][i]
        # If this parameter should be an integer, round the value
        if key in ['bb', 'ee']:
            value = int(round(value))

        params[key] = value
    return params

pbest_positions = [position_to_params(particles, i) for i in range(n_particles)]

#pbest_positions = particles
pbest_fitnesses = [0] * n_particles
gbest_position = None
gbest_fitness = None


def evaluate_fitness(data):
    y = fu(data['lr'],data['bb'],data['ee'],data['mm'])
    return y

cc=[]
params_keys = ['lr', 'bb', 'ee', 'mm']

for iteration in range(n_iterations):
    for i in range(n_particles):
        # Converts the position of the particle to a set of parameters
        params = position_to_params(particles, i)

        # Evaluates the fitness of the particle
        fitness = evaluate_fitness(params)

        # Updates the personal best position of the particle
        if fitness > pbest_fitnesses[i]:
            pbest_fitnesses[i] = fitness
            pbest_positions[i] = params  # Store the parameters, not the position

        # Updates the global best position
        if gbest_fitness is None or fitness > gbest_fitness:
            gbest_fitness = fitness
            gbest_position = params  # Store the parameters, not the position

    # Updates the velocities and positions of the particles
    for i in range(n_particles):
        for j, key in enumerate(params_keys):  # params_keys should be the list of parameter names
            r1, r2 = np.random.random(), np.random.random()
            velocities[j][i] = 0.5 * velocities[j][i] + c1 * r1 * (
                        pbest_positions[i][key] - particles[j][i]) + c2 * r2 * (gbest_position[key] - particles[j][i])
            particles[j][i] += velocities[j][i]
            # Clip the updated parameters to be within their bounds
            particles[j][i] = np.clip(particles[j][i], bounds[j][0], bounds[j][1])
    cc.append(gbest_fitness)
    print("gbest_fitness",cc)



