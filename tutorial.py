# general imports 
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import sys

# created by us 
import gridworld 

# ----------------------------------------------------------------------- #
#   This is a very simple script to play with RL params in a gridworld    #
# ----------------------------------------------------------------------- #
#
# The first thing you need to do is to understand the code
# below. Then, fill in the update_Q_SARSA() and update_Q_Learning()
# functions. You can use execute_configuration() to ensure your
# algorithm runs and produces plots before exploring with the help of
# run_parameter_sweep().
#
# Later, you can explore Monte Carlo/etc. Note that adding
# Monte Carlo, supporting n-step TD, or Q(sigma) require additional
# changes to the per-episode processing, including storing
# history. For simplicity, we currently use a simple if-else to toggle
# between single-step methods.
#
# The Config section allows you to tweak default parameters as well as
# run parameter sweeps to explore the effects on value functions and
# policies.
#
# Next, there are several different domains below.  Here are some
# things to explore to get you started, and then you can create your
# own experiments!  (Feel free to skip around.) 
#
# Exploration #1: Try the short_hallway and the long_hallway - is
# there a difference in how long it takes to learn each?  What happens
# as you adjust the discount factor? 
#
# Exploration #2: Try the test_maze.  What happens as you adjust the
# discount factor?  What about the learning rate?  Does it help to
# adjust the learning rate with time?  The epsilon with time?  Explore
# both using Q-learning and SARSA. Changing the learning rate or
# epsilon with time will require code changes in
# execute_configuration().
#
# Exploration #3: Try the simple_grid with different pit_rewards and
# action_error_prob: how does the policy change?  
#
# Exploration #4: Try the cliff_grid with a large pit_reward and
# different values of action_error_prob and epsilon (constant and
# decreasing).  How does the policy change?  Does it matter whether
# you use Q-learning or SARSA?
#
# Exploration #5: Adjust the code in gridworld.py to stop after a
# fixed number of iterations rather than when the goal is reached.
# Set the number of iterations to be large.  Does this change affect
# how quickly the policy converges?  How quickly the value function
# converges?  Explain what you see.

# -------------------- #
#   Different Tasks    #
# -------------------- #
# You can also create your own!
TASK_MAP = {
'short_hallway' : [
    '###', # '#' = wall
    '#o#', # 'o' = origin grid cell
    '#.#', # '.' = empty grid cell
    '#*#', # '*' = goal
    '###'],

'long_hallway' : [
    '###', # '#' = wall
    '#o#', # 'o' = origin grid cell
    '#.#', # '.' = empty grid cell
    '#.#', # '.' = empty grid cell
    '#.#', # '.' = empty grid cell
    '#.#', # '.' = empty grid cell
    '#.#', # '.' = empty grid cell
    '#.#', # '.' = empty grid cell
    '#.#', # '.' = empty grid cell
    '#*#', # '*' = goal
    '###'],

'test_maze' : [
    '#########',
    '#..#....#',
    '#..#..#.#',
    '#..#..#.#',
    '#..#.##.#',
    '#....*#.#',
    '#######.#',
    '#o......#',
    '#########'],

'simple_grid' : [
    '#######', 
    '#o....#', 
    '#..X..#', 
    '#....*#', 
    '#######'],

'cliff_grid' : [
    '#######', 
    '#.....#', 
    '#.##..#', 
    '#o...*#',
    '#XXXXX#', 
    '#######'],
}

# ----------------- #
#   Key Functions   # 
# ----------------- #
# The policy outputs the action for each states 
def policy( state , Q_table , action_count , epsilon ):
    if np.random.random() < epsilon:
        action = np.random.choice( action_count ) 
    else: 
        action = np.argmax( Q_table[ state , : ] ) 
    return action 

# Update the Q table.
def update_Q_SARSA( Q_table , alpha , gamma , state , action , reward , new_state , new_action ):
    sys.exit("TODO: Implement update_Q_SARSA")
    return Q_table 

def update_Q_Learning( Q_table , alpha , gamma , state , action , reward , new_state ):
    sys.exit("TODO: Implement update_Q_Learning!")
    return Q_table

# -------------------- #
# Config               #
# -------------------- #
#
# Change any values that you want applied to all configurations in the
# DEFAULT_CONFIG.
#
# Use sweep_params_row and sweep_params_column to run a 1- or 2-dimensional
# parameter sweep on any of the fields in the DEFAULT_CONFIG to
# observe effects.
DEFAULT_CONFIG = {
    # Task Parameters
    'task_name' : 'short_hallway', # See TASK_MAP keys above.
    'action_error_prob' : 0.1, # [0,1]
    'pit_reward' : -50, # Go nuts!
    # Algorithm Parameters
    'method' : 'sarsa', # Current Values: 'sarsa', 'qlearning'
    'alpha' : .5, # [0,1]
    'epsilon' : .1, # [0,1]
    'gamma' : .99, # [0,1]
    'episode_count' : 250, # Go nuts!
    'rep_count' : 10, # Go nuts!
    'episode_max_length' : 300, # Save yourself from infinite loops.
}

sweep_params_row = {
    'key' : 'gamma',
    'values' : [0, .1, .5, .9, .99]
}
sweep_params_column = {
    'key' : 'method',
    'values' : ['sarsa', 'qlearning']
}

def run_parameter_sweep(sweep_params_row = None, sweep_params_column = None):
    if not sweep_params_row:
        execute_configuration()
    else:
        param_base = DEFAULT_CONFIG
        for row, param_row_value in enumerate(sweep_params_row['values']):
            param_base[sweep_params_row['key']] = param_row_value
            if not sweep_params_column:
                execute_configuration(param_base, row_index=row, column_index=0, width=1, height=len(sweep_params_row['values']))
            else:
                for column, param_column_value in enumerate(sweep_params_column['values']):
                    param_base[sweep_params_column['key']] = param_column_value
                    execute_configuration(param_base, row_index=row, column_index=column, width=len(sweep_params_column['values']), height=len(sweep_params_row['values']))

def execute_configuration(config=DEFAULT_CONFIG, row_index=0, column_index=0, height=1, width=1):
    task = gridworld.GridWorld(TASK_MAP[config['task_name']] ,
                            action_error_prob=config['action_error_prob'],
                            rewards={'*': 50, 'moved': -1, 'hit-wall': -1,'X':config['pit_reward']} )
    task.get_max_reward()

    # Loop over some number of episodes
    episode_reward_set = np.zeros( ( config['rep_count'] , config['episode_count'] ) )
    for rep_iter in range( config['rep_count'] ):

        # Initialize the Q table
        Q_table = np.zeros( ( task.num_states , task.num_actions ) )

        # Loop until the episode is done
        for episode_iter in range( config['episode_count'] ):
        
            # Start the task
            task.reset()
            state = task.observe()
            action = policy( state , Q_table , task.num_actions , config['epsilon'] )
            episode_reward_list = []
            task_iter = 0

            # Loop until done -- check when do we get the final state reward?
            while True:
                task_iter = task_iter + 1
                new_state, reward = task.perform_action( action )
                new_action = policy( new_state , Q_table , task.num_actions , config['epsilon'] )
            
                # Update the Q_table.
                if config['method'] == 'sarsa':
                    Q_table = update_Q_SARSA( Q_table , config['alpha'] , config['gamma'] ,
                                              state , action , reward , new_state , new_action ) 
                elif config['method'] == 'qlearning':
                    Q_table = update_Q_Learning( Q_table , config['alpha'] , config['gamma'] ,
                                                 state , action , reward , new_state )
                else:
                    sys.exit("Unrecognized algorithm %s. Consider adding support?" % config['method'])

                # store the data
                episode_reward_list.append( reward )

                # stop if at goal/else update for the next iteration
                if task.is_terminal( state ) or task_iter > config['episode_max_length']:
                    episode_reward_set[ rep_iter , episode_iter ] = np.sum( episode_reward_list )
                    break
                else:
                    state = new_state
                    action = new_action

    add_plot(config, Q_table, episode_reward_set, row_index, column_index, width, height)

# -------------- #
#   Make Plots   #
# -------------- #
# Util to make an arrow 
# The directions are [ 'north' , 'south' , 'east' , 'west' ] 
def plot_arrow( location , direction , plot ):

    arrow = plt.arrow( location[0] , location[1] , dx , dy , fc="k", ec="k", head_width=0.05, head_length=0.1 )
    plot.add_patch(arrow) 

def add_plot(config, Q_table, episode_reward_set, row_index, column_index, width, height):
    # Useful stats for the plot
    task_map = TASK_MAP[config['task_name']]
    row_count = len( task_map )
    col_count = len( task_map[0] )
    value_function = np.reshape( np.max( Q_table , 1 ) , ( row_count , col_count ) )
    policy_function = np.reshape( np.argmax( Q_table , 1 ) , ( row_count , col_count ) )
    wall_info = .5 + np.zeros( ( row_count , col_count ) )
    wall_mask = np.zeros( ( row_count , col_count ) )
    for row in range( row_count ):
        for col in range( col_count ):
            if task_map[row][col] == '#':
                wall_mask[row,col] = 1
    wall_info = np.ma.masked_where( wall_mask==0 , wall_info )

    # Plot the rewards
    plt.subplot( height , width*2 , (row_index * (width*2) + column_index*2 + 1) )
    plt.plot( episode_reward_set.T )
    plt.title( 'Rewards per Episode (each line is a rep)' )
    plt.xlabel( 'Episode Number' )
    plt.ylabel( 'Sum of Rewards in Episode' )

    # value function plot
    plt.subplot( height , width*2 , (row_index * (width*2) + column_index*2 + 2) )
    plt.imshow( value_function , interpolation='none' , cmap=matplotlib.cm.jet )
    plt.colorbar()
    plt.imshow( wall_info , interpolation='none' , cmap=matplotlib.cm.gray )

    # policy plot
    for row in range( row_count ):
        for col in range( col_count ):
            if wall_mask[row][col] == 1:
                continue
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow( col , row , dx , dy , shape='full', fc='w' , ec='w' , lw=3, length_includes_head=True, head_width=.2 )
    plt.title( 'Value Function\n(Policy As Arrows)\nFor %s\nOn %s' % (config['method'], config['task_name']))


# -------------------- #
# Kick off the actual execution here.
# -------------------- #
plt.figure(figsize=(20*min(1, len(sweep_params_column)), 20*min(1, len(sweep_params_row))))

run_parameter_sweep(sweep_params_row, sweep_params_column)

plt.show( block=False ) 

# If you want to interact with it further...
# Note: If you ran more than one config, you will be interacting with
# the environment of the last one.
# Note: Type 'exit' to quit when you're done.
import pdb 
pdb.set_trace()
    


