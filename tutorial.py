# general imports 
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np

# created by us 
import gridworld 

# -------------------- #
#   Create the Task    #
# -------------------- #
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

simple_grid = [   
    '#######', 
    '#o....#', 
    '#..X..#', 
    '#....*#', 
    '#######']    

cliff_grid = [
    '#######', 
    '#.....#', 
    '#.##..#', 
    '#o...*#',
    '#XXXXX#', 
    '#######']    
                
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

maze = cliff_grid

task = gridworld.GridWorld( maze ,
                            action_error_prob=.1, 
                            rewards={'*': 50, 'moved': -1, 'hit-wall': -1,'X':-100} )

task.get_max_reward() 

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
# should the episode end when goal is reached? 

# Parameters 
alpha = .5
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
# Util to make an arrow 
# The directions are [ 'north' , 'south' , 'east' , 'west' ] 
def plot_arrow( location , direction , plot ):

    arrow = plt.arrow( location[0] , location[1] , dx , dy , fc="k", ec="k", head_width=0.05, head_length=0.1 )
    plot.add_patch(arrow) 

# Useful stats for the plot
row_count = len( maze )
col_count = len( maze[0] ) 
value_function = np.reshape( np.max( Q_table , 1 ) , ( row_count , col_count ) )
policy_function = np.reshape( np.argmax( Q_table , 1 ) , ( row_count , col_count ) )
wall_info = .5 + np.zeros( ( row_count , col_count ) )
wall_mask = np.zeros( ( row_count , col_count ) )
for row in range( row_count ):
    for col in range( col_count ):
        if maze[row][col] == '#':
            wall_mask[row,col] = 1     
wall_info = np.ma.masked_where( wall_mask==0 , wall_info )

# Plot the rewards
plt.subplot( 1 , 2 , 1 ) 
plt.plot( episode_reward_set.T )
plt.title( 'Rewards per Episode (each line is a rep)' ) 
plt.xlabel( 'Episode Number' )
plt.ylabel( 'Sum of Rewards in Episode' )

# value function plot 
plt.subplot( 1 , 2 , 2 ) 
plt.imshow( value_function , interpolation='none' , cmap=matplotlib.cm.jet )
plt.colorbar()
plt.imshow( wall_info , interpolation='none' , cmap=matplotlib.cm.gray )
plt.title( 'Value Function' )

# policy plot 
# plt.imshow( 1 - wall_mask , interpolation='none' , cmap=matplotlib.cm.gray )    
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
plt.title( 'Policy' )        
plt.show( block=False ) 

# If you want to interact with it further... 
import pdb 
pdb.set_trace()
    


