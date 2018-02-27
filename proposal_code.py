import PIL
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import test_human
from collections import OrderedDict

tile_names = [
    'tile1_24',
    'tile2_24',
    'tile3_24',
    'tile1_25',
    'tile2_25',
    'tile3_25',
]
regions = [
    '"West" Region',
    '"Central" Region',
    '"East" Region',
    '"West" Region',
    '"Central" Region',
    '"East" Region',
]

tiles = {}
for name in tile_names:
    num = name[4:]
    tiles[num] = PIL.Image.open(name + 's.pgm')
    
def plot_tiles(tiles, tile_names, regions):
    """Plots the tiles for demonstration purposes."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
    axes = axes.reshape(6,)
    for i, ax in enumerate(axes):
        ax.tick_params(
            axis='both', 
            which='both', 
            bottom='off', 
            top='off', 
            labelbottom='off', 
            right='off', 
            left='off', 
            labelleft='off'  # labels along the bottom edge are off
            )
        tile = tile_names[i][4:]
        ax.set_title(tile + ', ' + regions[i])
        img = tiles[tile]
        ax.imshow(np.array(img), cmap='Greys')
        
        
        
all_craters = pd.DataFrame(columns = ['x', 'y', 'd', 'tile'])
for tile in tile_names:
    num = tile[4:]
    new_craters = pd.read_csv('./gt_labels/{}_gt.csv'.format(num), header=None)
    new_craters.index = range(len(all_craters), len(all_craters)+len(new_craters))
    new_craters.columns = ['x', 'y', 'd']
    new_craters['tile'] = num
    all_craters = pd.concat([all_craters, new_craters], axis=0)
    
def plot_craters(tile, craters, title=None, scale=None, colors=['r', 'y', 'cyan', 'o', 'g']):
    """Takes an input PIL image "tile" and a dictionary,
    with each key as a type of crater and its element a list
    of craters with form: (x, y, d) (xpos, ypos, diameter)
    """
    img = tile
    if not scale:
        scale=.35
    if not title:
        title = list(craters.keys())[0]
    size = (int(img.size[0]*scale/80), int(img.size[1]*scale/80))
    fig, ax = plt.subplots(figsize=size);
    ax.imshow(np.array(img), cmap='Greys');
    ax.set_title(title);
    ax.set_ylabel('N-S direction in pixels @12.5 meters/pixel')
    ax.set_xlabel('E-W direction in pixels @12.5 meters/pixel')
    handles = []
    for i, group in enumerate(craters):
        color = colors[i]
        handles.append(mpatches.Patch(color=color, label=group))
        for crater in craters[group]:
            x = crater[0]
            y = crater[1]
            r = crater[2]/2
            circle = plt.Circle((x, y), r, fill=False, color=color);
            ax.add_artist(circle);
    plt.legend(handles=handles);
    plt.show();
    return None

proposal_columns = all_craters.columns
true_proposals = pd.DataFrame(columns = proposal_columns)
for tile in tile_names:
    num = tile[4:]
    new_proposals = pd.read_csv('./bandiera2010_candidates/{}_tp.csv'.format(num), header=None)
    new_proposals.columns = ['x', 'y', 'd']
    new_proposals.index = range(len(true_proposals), len(true_proposals)+len(new_proposals))
    new_proposals['tile'] = num
    true_proposals = pd.concat([true_proposals, new_proposals], axis=0)
    
false_proposals = pd.DataFrame(columns = proposal_columns)
for tile in tile_names:
    num = tile[4:]
    new_proposals = pd.read_csv('./bandiera2010_candidates/{}_tn.csv'.format(num), header=None)
    new_proposals.columns = ['x', 'y', 'd']
    new_proposals.index = range(len(false_proposals), len(false_proposals)+len(new_proposals))
    new_proposals['tile'] = num
    false_proposals = pd.concat([false_proposals, new_proposals], axis=0)
    
proposals = OrderedDict()
proposals['true proposals'] = true_proposals[true_proposals.tile=='1_24'][['x', 'y', 'd']].values
proposals['false proposals'] = false_proposals[false_proposals.tile=='1_24'][['x', 'y', 'd']].values

def proposal_histogram(tp=true_proposals, fp=false_proposals):
    plt.hist(tp.d.astype(int), bins=30, alpha=.5, normed=True, color='blue');
    plt.axvline(x=tp.d.mean(), color='blue', label='mean true candidate diameter', linestyle='dotted');
    plt.hist(fp.d.astype(int), bins=30, alpha=.5, normed=True, color='red');
    plt.axvline(x=fp.d.mean(), color='red', label='mean false candidate diameter', linestyle='dotted');
    plt.title('Crater Proposal Diameter Distribution');
    plt.xlabel('Proposed Crater Diameter (pixels)');
    plt.ylabel('Number of Proposals');
    plt.legend();
    plt.show();
    
human_performance = pd.read_csv('first_attempt.csv')
tp = np.where((human_performance.crater==1) & (human_performance.prediction==1), True, False)
fp = np.where((human_performance.crater==0) & (human_performance.prediction==1), True, False)
tn = np.where((human_performance.crater==0) & (human_performance.prediction==0), True, False)
fn = np.where((human_performance.crater==1) & (human_performance.prediction==0), True, False)

def display_proposals(proposals=tp, title='title', num_imgs=5):
    fig, ax = plt.subplots(1, num_imgs, figsize=(num_imgs, 2));
    fig.suptitle(title);
    num = 0
    for axis in ax:
        img = test_human.get_image(human_performance[proposals]['id'].iloc[num])
        axis.imshow(img, cmap='Greys')
        axis = test_human.remove_ticks(axis)
        plt.tight_layout()
        num += 1