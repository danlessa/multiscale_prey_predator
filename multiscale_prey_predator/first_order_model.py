import numpy as np

# Behaviors
def reproduce_prey(params, substep, state_history, prev_state):
    born_population = prev_state['prey_population']
    born_population *= params['prey_reproduction_rate']
    born_population *= params['dt']
    return {'add_prey': born_population}


def reproduce_predators(params, substep, state_history, prev_state):
    born_population = prev_state['predator_population']
    born_population *= prev_state['prey_population']  
    born_population *= params['predator_interaction_factor']
    born_population *= params['dt']
    return {'add_predators': born_population}


def eliminate_prey(params, substep, state_history, prev_state):
    population = prev_state['prey_population']
    natural_elimination = population * params['prey_death_rate']
    
    interaction_elimination = population * prev_state['predator_population']
    interaction_elimination *= params['prey_interaction_factor']
    
    eliminated_population = natural_elimination + interaction_elimination
    eliminated_population *= params['dt']
    return {'add_prey': -1.0 * eliminated_population}


def eliminate_predators(params, substep, state_history, prev_state):
    population = prev_state['predator_population']
    eliminated_population = population * params['predator_death_rate']
    eliminated_population *= params['dt']
    return {'add_predators': -1.0 * eliminated_population}


# Mechanisms
def prey_population(params, substep, state_history, prev_state, policy_input):
    updated_prey_population = np.ceil(prev_state['prey_population'] + policy_input['add_prey'])
    return ('prey_population', max(updated_prey_population, 0))


def predator_population(params, substep, state_history, prev_state, policy_input):
    updated_predator_population = np.ceil(prev_state['predator_population'] + policy_input['add_predators'])
    return ('predator_population', max(updated_predator_population, 0))




initial_state = {
    'prey_population': 100,
    'predator_population': 50
}

params = {
    'prey_reproduction_rate': [0.3],
    'predator_interaction_factor': [0.0001],
    'prey_interaction_factor': [0.001],
    'prey_death_rate': [0.01],
    'predator_death_rate': [0.5],
    'dt': [0.1]
}

partial_state_update_block = [
    {
        # lotka_volterra.py
        'policies': {
            'reproduce_prey': reproduce_prey,
            'reproduce_predators': reproduce_predators,
            'eliminate_prey': eliminate_prey,
            'eliminate_predators': eliminate_predators
        },
        'variables': {
            'prey_population': prey_population,
            'predator_population': predator_population            
        }
    }
]

N_samples = 1
N_timesteps = 100