import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL


def invert_image(input_image):
    """inverts the input image color values. Returns image
    of same dim."""
    if not type(input_image) == type(np.array([0])):
        input_image = np.array(input_image)/256
    return (input_image - 256) * -1 

def normalize_image(input_image):
    """Takes a PIL image and "normalizes" its pixel values;
    returns a numpy array of same shape, with min value 0 and
    max value 256.
    """
    if not type(input_image) == type(np.array([0])):
        array = np.array(input_image)/256
    else:
        array = input_image/256
    min_ = np.min(np.min(array))
    array = array - min_
    max_ = np.max(np.max(array))
    array = array * 1/max_ * 256
    return array

def get_image(id_number):
    """Takes id number and returns an image."""
    path = './tp_images/' + id_number + '.bmp'
    try:
        img = PIL.Image.open(path)
    except:
        try:
            path = './fp_images/' + id_number + '.bmp'
            img = PIL.Image.open(path)
        except:
            raise Exception('Error: No file associated with ', id_number)
    return np.array(img)

def remove_ticks(ax_obj):
    """takes an ax object from matplotlib and removes ticks."""
    ax_obj.tick_params(
        axis='both', 
        which='both', 
        bottom='off', 
        top='off', 
        labelbottom='off', 
        right='off', 
        left='off', 
        labelleft='off'
        )
    return ax_obj

def show_example(id_number):
    """Takes example id number and shows it for user inspection."""
    img = get_image(id_number)
    #Plot "zoomed in"
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2, 56/80))
    ax[0].set_title('natural')
    ax[0].imshow(img, cmap='Greys')
    ax[0] = remove_ticks(ax[0])
    ax[1].set_title('inverted')
    ax[1].imshow(invert_image(img), cmap='Greys')
    ax[1] = remove_ticks(ax[1])
    plt.tight_layout()
    plt.show();
    
    #Plot "actual size"
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2, 28/80))
    ax[0].set_title('natural')
    ax[0].imshow(img, cmap='Greys')
    ax[0] = remove_ticks(ax[0])
    ax[1].set_title('inverted')
    ax[1].imshow(invert_image(img), cmap='Greys')
    ax[1] = remove_ticks(ax[1])
    plt.show();
    return

def train(proposals):
    """Takes all proposals (df) as input and displays information to "train" user
    in an interactive session.
    """
    sample = proposals.sample(1)
    idx = sample.index
    proposals = proposals.drop(idx, axis=0)
    if sample.iloc[0].crater==1:
        print('This is an example of a positive candidate (true crater).')
    elif sample.iloc[0].crater==0:
        print('This is an example of a negative candidate (false crater).')
    print('press q to cycle through images')
    show_example(sample.iloc[0]['id'])
    response = None
    while response not in ['y', 'n']:
        response = input('Would you like to see another example? (y/n)')
    if response == 'y':
        return train(proposals)
    elif response == 'n':
        return proposals
    
def get_result(id_number):
    """Takes id number, shows user, accepts user
    input, and returns 0, 1 for guess.
    """
    print('showing example... (press q to cycle through images)')
    show_example(id_number)
    result = input('Is it a crater? (y/n) (q to quit, a to see again)')
    if result not in ['y', 'n', 'q', 'a']:
        print('sorry, invalid input. must be: y, n, q, or a.')
        return(get_result(id_number))
    elif result == 'y':
        return 1
    elif result == 'n':
        return 0
    elif result == 'q':
        quit()
    elif result == 'a':
        return(get_result(id_number))
    

if __name__ in '__main__':
    print('Welcome to the crater identification test!')
    print('The purpose of this program is to test the human ability to classify craters from non-crater proposals.')
    print('You will first be shown as many examples as you wish. You will then be prompted to begin classifying crater candidates.')
    input('OK (press enter)')
    proposals = pd.read_csv('proposals.csv')
    results = pd.DataFrame(columns=list(proposals.columns)+['prediction'])
    train(proposals)
    save_path = input('What filename would you like to save the results with? (exclude extension)')
    while True:
        sample = proposals.sample(1)
        idx = sample.index
        proposals = proposals.drop(idx, axis=0)
        next_result = pd.DataFrame(columns=proposals.columns, index=[len(results)], data=sample.values)
        id_ = sample['id'].iloc[0]
        prediction = get_result(id_)
        next_result['prediction'] = prediction
        results = pd.concat([results, next_result], axis=0)
        print('{} results recorded so far.'.format(len(results)))
        results.to_csv('{}.csv'.format(save_path), index=False)
