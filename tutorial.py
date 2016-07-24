import matplotlib.pyplot as plt
import numpy as np
import gridworld 

# -------------------- #
#   Create the Task    #
# -------------------- #
action_list = [ 'north' , 'south' , 'east' , 'west' ] 

trivial_maze = [   
    '###', # '#' = wall
    '#o#', # 'o' = origin grid cell
    '#.#', # '.' = empty grid cell
    '#*#', # '*' = goal
    '###']

long_maze = [   
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
    '###']
            
simple_maze = [
    '#########',
    '#..#....#',
    '#..#..#.#',
    '#..#..#.#',
    '#..#.##.#',
    '#....*#.#',
    '#######.#',
    '#o......#',
    '#########']

maze = simple_maze 

task = gridworld.GridWorld( maze ,
                            action_error_prob=.1, 
                            rewards={'*': 50, 'moved': -1, 'hit-wall': -1 } )

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

# Update the Q table 
def update_SARSA( Q_table , alpha , gamma , state , action , reward , new_state , new_action ):
    old_Q = Q_table[ state , action ]
    next_Q = Q_table[ new_state , new_action ]
    new_Q = old_Q + alpha * ( reward + gamma * next_Q - old_Q )
    Q_table[ state , action ] = new_Q 
    return Q_table 
    
# Things to play with
# - changing the discount_factor
# - changing the initialization of the Q function (optimistic/shaping) 
# - changing the alpha
# - changing the epsilon in the epsilon greedy 
# - on vs. off policy 
# action error prob vs epsilon 

# Parameters 
alpha = .1
epsilon = .1 
gamma = .99
state_count = task.num_states  
action_count = task.num_actions 
episode_count = 100
rep_count = 10

# Loop over some number of episodes
episode_reward_set = np.zeros( ( rep_count , episode_count ) ) 
for rep_iter in range( rep_count ):

    # Initialize the Q table 
    Q_table = np.zeros( ( state_count , action_count ) )

    # Loop until the episode is done 
    for episode_iter in range( episode_count ):
        
        # Start the task 
        task.reset()
        state = task.observe() 
        action = policy( state , Q_table , action_count , epsilon ) 
        episode_reward_list = [] 

        # Loop until done -- check when do we get the final state reward? 
        while True: 
            new_state, reward = task.perform_action( action )
            new_action = policy( new_state , Q_table , action_count , epsilon ) 
            
            # update the Q_table
            Q_table = update_SARSA( Q_table , alpha , gamma , 
                                    state , action , reward , new_state , new_action ) 

            # store the data
            episode_reward_list.append( reward ) 
            
            # stop if at goal/else update for the next iteration 
            if task.is_terminal( state ):
                episode_reward_set[ rep_iter , episode_iter ] = np.sum( episode_reward_list )
                break
            else:
                state = new_state
                action = new_action 

# -------------- #
#   Make Plots   #
# -------------- #
def plot_value_and_q_function( Q_table , maze ):
    action_count = Q_table.shape[1]
    row_count = len( maze )
    col_count = len( maze[0] )
    Q_max = np.max( Q_table )
    Q_min = np.min( Q_table ) 
    for action_iter in range( action_count ): 
        plt.subplot( 1 , action_count + 1 , action_iter + 1 )
        plt.imshow( np.reshape( Q_table[:, action_iter] , ( row_count , col_count ) ) , interpolation='none' , vmin=Q_min , vmax=Q_max )
        plt.title( 'Q-value of ' + action_list[ action_iter ] ) 
        
    plt.subplot( 1 , action_count + 1 , action_count + 1 )
    plt.imshow( np.reshape( np.max( Q_table , 1 ) , ( row_count , col_count ) ) , interpolation='none' , vmin=Q_min , vmax=Q_max )
    plt.title( 'Value Function' )
    plt.colorbar() 
    plt.show() 

# Plot the rewards     
plt.figure(1) 
plt.plot( episode_reward_set.T )
plt.title( 'Rewards per Episode (each line is a rep)' ) 
plt.xlabel( 'Episode Number' )
plt.ylabel( 'Sum of Rewards in Episode' )
plt.show( block=False )

# Plot the Value functions 
plt.figure(2) 
plot_value_and_q_function( Q_table , maze )

# If you want to interact with it further... 
import pdb 
pdb.set_trace()
    


