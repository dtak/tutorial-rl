# general imports 
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np

# created by us 
import gridworld 

# ----------------------------------------------------------------------- #
#   This is a very simple script to play with RL params in a gridworld    #
# ----------------------------------------------------------------------- #
#
# The first thing you need to do is to understand the code below and
# fill in the update_Q() function fill in the code so it can do either
# SARSA or Q-Learning (and later, you can explore Monte Carlo/etc).
# There's a section with task parameters and a section with 
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
# both using Q-learning and SARSA.  
#
# Exploration #3: Try the simple_grid with different pit_rewards and
# action_error_prob: how does the policy change?  
#
# Exploration #4: Try the cliff_grid with a large pit_reward and
# different values of action_error_prob and epsilon (constant and
# decreasing).  How does the policy change?  Does it matter whether
# you use Q-learning or SARSA?
#
# Exploration #5: Adjust the code in gridworld.py to stop a fixed
# number of iterations rather than when the goal is reached.  Set the
# number of iterations to be large.  Does this change affect how
# quickly the policy converges?  How quickly the value function
# converges?  Explain what you see.  

# -------------------- #
#   Different Tasks    #
# -------------------- #
# You can also create your own! 
short_hallway = [   
    '###', # '#' = wall
    '#o#', # 'o' = origin grid cell
    '#.#', # '.' = empty grid cell
    '#*#', # '*' = goal
    '###']

long_hallway = [   
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

test_maze = [
    '#########',
    '#..#....#',
    '#..#..#.#',
    '#..#..#.#',
    '#..#.##.#',
    '#....*#.#',
    '#######.#',
    '#o......#',
    '#########']

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
def update_Q( Q_table , alpha , gamma , state , action , reward , new_state , new_action ):
    # Fill in this function
    return Q_table 
    
# -------------------- #
#   Create the Task    #
# -------------------- #
# Task Parameters
task_name = short_hallway 
action_error_prob = .1 
pit_reward = -500
task = gridworld.GridWorld( task_name ,
                            action_error_prob=action_error_prob, 
                            rewards={'*': 50, 'moved': -1, 'hit-wall': -1,'X':pit_reward} )
task.get_max_reward() 

# ---------------- #
#   Run the Task   # 
# ---------------- #
# Algorithm Parameters 
alpha = .5
epsilon = .1
gamma = .99 
state_count = task.num_states  
action_count = task.num_actions 
episode_count = 250
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
        task_iter = 0 

        # Loop until done -- check when do we get the final state reward? 
        while True:
            task_iter = task_iter + 1 
            new_state, reward = task.perform_action( action )
            new_action = policy( new_state , Q_table , action_count , epsilon ) 
            
            # update the Q_table
            Q_table = update_Q( Q_table , alpha , gamma , 
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
row_count = len( task_name )
col_count = len( task_name[0] ) 
value_function = np.reshape( np.max( Q_table , 1 ) , ( row_count , col_count ) )
policy_function = np.reshape( np.argmax( Q_table , 1 ) , ( row_count , col_count ) )
wall_info = .5 + np.zeros( ( row_count , col_count ) )
wall_mask = np.zeros( ( row_count , col_count ) )
for row in range( row_count ):
    for col in range( col_count ):
        if task_name[row][col] == '#':
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
    


