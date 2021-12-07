
import random
from uuid import UUID, uuid4
import numpy as np
from typing import Sequence, Union

# Initialization


def new_agent(agent_type: str, location: tuple[int, int],
              food: int = 10, age: int = 0) -> dict:
    agent = {'type': agent_type,
             'location': location,
             'food': food,
             'age': age}
    return agent


def generate_agents(N: int, M: int, initial_sites: int,
                    n_predators: int,
                    n_prey: int) -> dict[UUID, dict]:

    available_locations = [(n, m) for n in range(N) for m in range(M)]
    initial_agents = {}
    type_queue = ['prey'] * n_prey
    type_queue += ['predator'] * n_predators
    random.shuffle(type_queue)
    for agent_type in type_queue:
        location = random.choice(available_locations)
        available_locations.remove(location)
        created_agent = new_agent(agent_type, location)
        initial_agents[uuid4()] = created_agent
    return initial_agents


# Environment
@np.vectorize
def calculate_increment(value, growth_rate, max_value):
    new_value = (value + growth_rate
                 if value + growth_rate < max_value
                 else max_value)
    return new_value

# Location heper


def check_location(position: tuple,
                   all_sites: np.matrix,
                   busy_locations: list[tuple]) -> Sequence[tuple]:
    """
    Returns an list of available location tuples neighboring an given
    position location.
    """
    N, M = all_sites.shape
    potential_sites = [(position[0], position[1] + 1),
                       (position[0], position[1] - 1),
                       (position[0] + 1, position[1]),
                       (position[0] - 1, position[1])]
    potential_sites = [(site[0] % N, site[1] % M) for site in potential_sites]
    valid_sites = [
        site for site in potential_sites if site not in busy_locations]
    return valid_sites


def get_free_location(position: tuple,
                      all_sites: np.matrix,
                      used_sites: list[tuple]) -> Union[tuple, bool]:
    """
    Gets an random free location neighboring an position. Returns False if
    there aren't any location available.
    """
    available_locations = check_location(position, all_sites, used_sites)
    if len(available_locations) > 0:
        return random.choice(available_locations)
    else:
        return False


def nearby_agents(location: tuple, agents: dict[str, dict]) -> dict[str, dict]:
    """
    Filter the non-nearby agents.
    """
    neighbors = {label: agent for label, agent in agents.items()
                 if is_neighbor(agent['location'], location)}
    return neighbors


def is_neighbor(location_1: tuple, location_2: tuple) -> bool:
    dx = np.abs(location_1[0] - location_2[0])
    dy = (location_1[1] - location_2[0])
    distance = dx + dy
    if distance == 1:
        return True
    else:
        return False

# Behaviors


def digest_and_olden(params, substep, state_history, prev_state):
    agents = prev_state['agents']
    delta_food = {agent: -1 for agent in agents.keys()}
    delta_age = {agent: +1 for agent in agents.keys()}
    return {'agent_delta_food': delta_food,
            'agent_delta_age': delta_age}


def move_agents(params, substep, state_history, prev_state):
    """
    Move agents.
    """
    sites = prev_state['sites']
    agents = prev_state['agents']
    busy_locations = [agent['location'] for agent in agents.values()]
    new_locations = {}
    for label, properties in agents.items():
        new_location = get_free_location(
            properties['location'], sites, busy_locations)
        if new_location is not False:
            new_locations[label] = new_location
            busy_locations.append(new_location)
        else:
            continue
    return {'update_agent_location': new_locations}


def reproduce_agents(params, substep, state_history, prev_state):
    """
    Generates an new agent through an nearby agent pair, subject to rules.
    Not done.
    """
    agents = prev_state['agents']
    sites = prev_state['sites']
    food_threshold = params['reproduction_food_threshold']
    reproduction_food = params['reproduction_food']
    reproduction_probability = params['reproduction_probability']
    busy_locations = [agent['location'] for agent in agents.values()]
    already_reproduced = []
    new_agents = {}
    agent_delta_food = {}
    for agent_type in set(agent['type'] for agent in agents.values()):
        # Only reproduce agents of the same type
        specific_agents = {label: agent for label, agent in agents.items()
                           if agent['type'] == agent_type}
        for agent_label, agent_properties in specific_agents.items():
            location = agent_properties['location']
            if (agent_properties['food'] < food_threshold or agent_label in already_reproduced):
                continue
            kind_neighbors = nearby_agents(location, specific_agents)
            available_partners = [label for label, agent in kind_neighbors.items()
                                  if agent['food'] >= food_threshold
                                  and label not in already_reproduced]
            reproduction_location = get_free_location(
                location, sites, busy_locations)

            if reproduction_location is not False and len(available_partners) > 0:
                reproduction_partner_label = random.choice(available_partners)
                reproduction_partner = agents[reproduction_partner_label]
                already_reproduced += [agent_label, reproduction_partner_label]

                agent_delta_food[agent_label] = -1.0 * reproduction_food
                agent_delta_food[reproduction_partner_label] = - \
                    1.0 * reproduction_food
                new_agent_properties = {'type': agent_type,
                                        'location': reproduction_location,
                                        'food': 2.0 * reproduction_food,
                                        'age': 0}
                new_agents[uuid4()] = new_agent_properties
                busy_locations.append(reproduction_location)
    return {'agent_delta_food': agent_delta_food, 'agent_create': new_agents}


def feed_prey(params, substep, state_history, prev_state):
    """
    Feeds the hungry prey with all food located on its site.
    """
    agents = prev_state['agents']
    sites = prev_state['sites']
    preys = {k: v for k, v in agents.items() if v['type'] == 'prey'}
    hungry_preys = {label: properties for label, properties in preys.items()
                    if properties['food'] < params['hunger_threshold']}

    agent_delta_food = {}
    site_delta_food = {}
    for label, properties in hungry_preys.items():
        location = properties['location']
        available_food = sites[location]
        agent_delta_food[label] = available_food
        site_delta_food[location] = -available_food

    return {'agent_delta_food': agent_delta_food,
            'site_delta_food': site_delta_food}


def hunt_prey(params, substep, state_history, prev_state):
    """
    Feeds the hungry predators with an random nearby prey.
    """
    agents = prev_state['agents']
    sites = prev_state['sites']
    hungry_threshold = params['hunger_threshold']
    preys = {k: v for k, v in agents.items()
             if v['type'] == 'prey'}
    predators = {k: v for k, v in agents.items()
                 if v['type'] == 'predator'}
    hungry_predators = {k: v for k, v in predators.items()
                        if v['food'] < hungry_threshold}
    agent_delta_food = {}
    for predator_label, predator_properties in hungry_predators.items():
        location = predator_properties['location']
        nearby_preys = nearby_agents(location, preys)
        if len(nearby_preys) > 0:
            eaten_prey_label = random.choice(list(nearby_preys.keys()))
            delta_food = preys.pop(eaten_prey_label)['food']
            agent_delta_food[predator_label] = delta_food
            agent_delta_food[eaten_prey_label] = -1 * delta_food
        else:
            continue

    return {'agent_delta_food': agent_delta_food}


def natural_death(params, substep, state_history, prev_state):
    """
    Remove agents which are old or hungry enough.
    """
    agents = prev_state['agents']
    maximum_age = params['agent_lifespan']
    agents_to_remove = []
    for agent_label, agent_properties in agents.items():
        to_remove = agent_properties['age'] > maximum_age
        to_remove |= (agent_properties['food'] <= 0)
        if to_remove:
            agents_to_remove.append(agent_label)
    return {'remove_agents': agents_to_remove}


# Mechanisms
def agent_food_age(params, substep, state_history, prev_state, policy_input):
    delta_food_by_agent = policy_input['agent_delta_food']
    delta_age_by_agent = policy_input['agent_delta_age']
    updated_agents = prev_state['agents'].copy()

    for agent, delta_food in delta_food_by_agent.items():
        updated_agents[agent]['food'] += delta_food
    for agent, delta_age in delta_age_by_agent.items():
        updated_agents[agent]['age'] += delta_age
    return ('agents', updated_agents)


def agent_location(params, substep, state_history, prev_state, policy_input):
    updated_agents = prev_state['agents'].copy()
    for label, location in policy_input['update_agent_location'].items():
        updated_agents[label]['location'] = location
    return ('agents', updated_agents)


def agent_create(params, substep, state_history, prev_state, policy_input):
    updated_agents = prev_state['agents'].copy()
    for label, food in policy_input['agent_delta_food'].items():
        updated_agents[label]['food'] += food
    for label, properties in policy_input['agent_create'].items():
        updated_agents[label] = properties
    return ('agents', updated_agents)


def site_food(params, substep, state_history, prev_state, policy_input):
    updated_sites = prev_state['sites'].copy()
    for label, delta_food in policy_input['site_delta_food'].items():
        updated_sites[label] += delta_food
    return ('sites', updated_sites)


def agent_food(params, substep, state_history, prev_state, policy_input):
    updated_agents = prev_state['agents'].copy()
    for label, delta_food in policy_input['agent_delta_food'].items():
        updated_agents[label]['food'] += delta_food
    return ('agents', updated_agents)


def agent_remove(params, substep, state_history, prev_state, policy_input):
    agents_to_remove = policy_input['remove_agents']
    surviving_agents = {k: v for k, v in prev_state['agents'].items()
                        if k not in agents_to_remove}
    return ('agents', surviving_agents)

# Behaviors


def grow_food(params, substep, state_history, prev_state):
    """
    Increases the food supply in all sites, subject to an maximum.
    """
    regenerated_sites = calculate_increment(prev_state['sites'],
                                            params['food_growth_rate'],
                                            params['maximum_food_per_site'])
    return {'update_food': regenerated_sites}


initial_values = {
    'world_size_n_dimension': 20,
    'world_size_m_dimension': 20,
    'initial_food_sites': 5,
    'initial_predator_count': 20,
    'initial_prey_count': 20
}

params = {
    'food_growth_rate': [3],
    'maximum_food_per_site': [10],
    'reproduction_probability': [1.0],
    'agent_lifespan': [35],
    'reproduction_food_threshold': [6],  # When the agents start reproducing
    'hunger_threshold': [15],  # When the agents start eating
    'reproduction_food': [1]  # How much food is needed for reproducing
}

initial_state = {
    'agents': generate_agents(initial_values['world_size_n_dimension'], initial_values['world_size_m_dimension'],
                              initial_values['initial_food_sites'], initial_values['initial_predator_count'],
                              initial_values['initial_prey_count']),
    'sites': np.ones((initial_values['world_size_n_dimension'], initial_values['world_size_m_dimension'])) * initial_values['initial_food_sites']
}

# Mechanisms


def update_food(params, substep, state_history, prev_state, policy_input):
    key = 'sites'
    value = policy_input['update_food']
    return (key, value)


partial_state_update_block = [
    {
        # environment.py
        'policies': {
            'grow_food': grow_food
        },
        'variables': {
            'sites': update_food
        }
    },
    {
        # agents.py
        'policies': {
            'increase_agent_age': digest_and_olden
        },
        'variables': {
            'agents': agent_food_age

        }
    },
    {
        # agents.py
        'policies': {
            'move_agent': move_agents
        },
        'variables': {
            'agents': agent_location

        }
    },
    {
        # agents.py
        'policies': {
            'reproduce_agents': reproduce_agents

        },
        'variables': {
            'agents': agent_create

        }
    },
    {
        # agents.py
        'policies': {
            'feed_prey': feed_prey
        },
        'variables': {
            'agents': agent_food,
            'sites': site_food
        }
    },
    {
        # agents.py
        'policies': {
            'hunt_prey': hunt_prey
        },
        'variables': {
            'agents': agent_food
        }
    },
    {
        # agents.py
        'policies': {
            'natural_death': natural_death
        },
        'variables': {
            'agents': agent_remove
        }
    }
]

N_samples = 3
N_timesteps = 150