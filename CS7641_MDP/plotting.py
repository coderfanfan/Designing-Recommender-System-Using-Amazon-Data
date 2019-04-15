from matplotlib import pyplot as plt
import numpy as np

def transform_policy(policy, map_nrow, map_ncol):
    """ 
    Transform policy from state -> all actions maps (in which the best action has value 1 and the rest actions have value 0)
    to states in the shape of the map in matrix (in which the best action is indicated by the index of the action)
    """ 
    return np.argmax(policy, axis=1).reshape((map_nrow, map_ncol))


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    
    # transform policy
    map_nrow = len(map_desc)
    map_ncol = len(map_desc[0])
    policy = transform_policy(policy, map_nrow, map_ncol)
    
    # plot policy
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)
            if map_desc[i, j] != b'H': # do not plot action on holes
                text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                               horizontalalignment='center', verticalalignment='center', color='w')


    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()

    return plt


def plot_value_map(title, value, map_desc, color_map):
    
    # transform value
    map_nrow = len(map_desc)
    map_ncol = len(map_desc[0])
    value = value.reshape(map_nrow, map_ncol)
    
    # plot value
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, xlim=(0, value.shape[1]), ylim=(0, value.shape[0]))
    font_size = 'x-large'
    if value.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            y = value.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)
            text = ax.text(x+0.5, y+0.5, round(value[i, j], 2), weight='bold', size=font_size,
                              horizontalalignment='center', verticalalignment='center', color='w')


    plt.axis('off')
    plt.xlim((0, value.shape[1]))
    plt.ylim((0, value.shape[0]))
    plt.tight_layout()

    return plt