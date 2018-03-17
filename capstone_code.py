import PIL
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from collections import OrderedDict
from math import sin
from scipy import ndimage as ndi
from scipy.ndimage import find_objects
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.draw import circle
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.metrics import classification_report
import pycda
from pycda.sample_data import get_sample_image, get_sample_csv
from pycda.error_stats import ErrorAnalyzer
from pycda.classifiers import ConvolutionalClassifier
from pycda.extractors import FastCircles, WatershedCircles
from proposal_code import tiles
from generate_masks import pad_mask, crop_square
import test_human
from test_human import remove_ticks


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

tile_path = './tiles/raw/'
tiles = {}
for name in tile_names:
    num = name[4:]
    tiles[num] = PIL.Image.open(tile_path + name + 's.pgm')
    
def plot_tiles(tiles, tile_names, regions):
    """Plots the tiles for demonstration purposes."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9,6))
    axes = axes.reshape(6,)
    for i, ax in enumerate(axes):
        ax = test_human.remove_ticks(ax)
        tile = tile_names[i][4:]
        ax.set_title(tile + ', ' + regions[i])
        img = tiles[tile]
        ax.imshow(np.array(img), cmap='Greys_r')
        
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
        
samp_img = Image.open('./capstone_files/sample_image.png')
samp_mask = Image.open('./capstone_files/sample_mask.png')
filt_img = Image.open('./capstone_files/filtered.png')    
unet_img = Image.open('./capstone_files/unet_vis.png')

def plot_filters():
    fig, ax = plt.subplots(ncols=2, figsize=(9, 5))
    fig.suptitle('Automatically-learned Image Filters');
    ax[0].imshow(samp_img, cmap='Greys_r')
    ax[0] = test_human.remove_ticks(ax[0])
    ax[1].imshow(filt_img, cmap='CMRmap')
    ax[1] = test_human.remove_ticks(ax[1])
    plt.tight_layout();
    plt.show()
    
def plot_mask():
    fig, ax = plt.subplots(ncols=2, figsize=(6, 3))
    fig.suptitle('Sample Image and Target Mask');
    ax[0].imshow(samp_img, cmap='Greys_r')
    ax[0] = test_human.remove_ticks(ax[0])
    ax[1].imshow(samp_mask, cmap='CMRmap')
    ax[1] = test_human.remove_ticks(ax[1])
    plt.show()
    
def inspect_detection(id_no):
    """Displays a previously-saved prediction for inspection
    by loading it from hard drive.
    """
    base_path = './tiles/keras_folders/test'
    prediction = np.load('./tiles/predictions/{}.npy'.format(id_no))
    source_image = crop_square(Image.open(base_path+'/{}.png'.format(id_no)), 172, orgn=(42, 42))
    ground_truth = Image.open(base_path+'_mask/{}_mask.png'.format(id_no))
    fig, ax = plt.subplots(ncols=3, figsize=(8, 4))
    fig.suptitle('Results for Tile {}'.format(id_no))
    ax = [remove_ticks(x) for x in ax]
    ax[0].imshow(np.array(source_image), cmap='Greys_r');
    ax[0].set_title('Input image');
    ax[1].imshow(np.array(ground_truth), cmap='CMRmap');
    ax[1].set_title('"Ground Truth"');
    ax[2].imshow(prediction, cmap='CMRmap');
    ax[2].set_title('Model Prediction');
    plt.tight_layout()
    plt.show()
    return

def inspect_circle_output(id_no):
    """Displays a previously-saved prediction for inspection
    by loading it from hard drive.
    """
    base_path = './tiles/keras_folders/test'
    prediction = np.load('./tiles/predictions/{}.npy'.format(id_no))
    source_image = crop_square(Image.open(base_path+'/{}.png'.format(id_no)), 172, orgn=(42, 42))
    ground_truth = Image.open(base_path+'_mask/{}_mask.png'.format(id_no))
    
    circles = FastCircles()
    craters = circles(prediction)
    extraction = build_target(craters)
    
    fig, ax = plt.subplots(ncols=4, figsize=(12, 3))
    fig.suptitle('Desirable Behavior from Circle Procedure')
    ax = [remove_ticks(x) for x in ax]
    ax[0].imshow(np.array(source_image), cmap='Greys_r');
    ax[0].set_title('Input image');
    ax[1].imshow(np.array(ground_truth), cmap='CMRmap');
    ax[1].set_title('"Ground Truth"');
    ax[2].imshow(prediction, cmap='CMRmap');
    ax[2].set_title('Model Prediction');
    ax[3].imshow(extraction, cmap='CMRmap');
    ax[3].set_title('Extracted Circles');
    plt.show()
    return

def detection_error_example():
    big_crat = np.asarray(tiles['3_25'].crop((800, 725, 1450, 1350)))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Erroneous Detection Criterion')
    ax[0].imshow(big_crat, cmap='Greys_r')
    ax[0].set_title('Diameter Error')
    circle = plt.Circle((322, 329), 144, fill=False, color='green', linewidth=2);
    ax[0].add_artist(circle);
    circle = plt.Circle((322, 329), 201, fill=False, color='red', linewidth=2);
    ax[0].add_artist(circle);
    circle = plt.Circle((322, 329), 87, fill=False, color='red', linewidth=2);
    ax[0].add_artist(circle);
    ax[1].imshow(big_crat, cmap='Greys_r')
    ax[1].set_title('Location Error')
    circle = plt.Circle((322, 329), 144, fill=False, color='green', linewidth=2);
    ax[1].add_artist(circle);
    circle = plt.Circle((322, 444), 144, fill=False, color='red', linewidth=2);
    ax[1].add_artist(circle);
    ax[2].imshow(big_crat, cmap='Greys_r')
    ax[2].set_title('Combination of Error Types')
    circle = plt.Circle((322, 329), 144, fill=False, color='green', linewidth=2);
    ax[2].add_artist(circle);
    circle = plt.Circle((256, 395), 111, fill=False, color='red', linewidth=2);
    ax[2].add_artist(circle);
    circle = plt.Circle((388, 263), 177, fill=False, color='red', linewidth=2);
    ax[2].add_artist(circle);
    ax[0], ax[1], ax[2] = remove_ticks(ax[0]), remove_ticks(ax[1]), remove_ticks(ax[2])
    handles = []
    handles.append(mpatches.Patch(color='green', label='Perfect Detection'))
    handles.append(mpatches.Patch(color='red', label='Rejected by Criteria'))
    plt.legend(handles=handles);

prediction = np.load('./tiles/predictions/tile_2_25.npy')
prediction = prediction[150:300, 10:160]
blob_image = np.where(prediction>.5, 1, 0)
mask = np.array(Image.open('./tiles/mask/2_25_mask.bmp'))
mask = mask[150:300, 10:160]
tile_array = np.array(tiles['2_25'])
blob_input = tile_array[150:300, 10:160]


def build_target(craters):
    """Takes a list of craters and returns a mask image, 150x150"""
    try:
        assert isinstance(craters, pd.DataFrame)
    except:
        craters = pd.DataFrame(columns = ['y', 'x', 'd'], data=craters)
    size = (150, 150)
    image = np.zeros(size, dtype='uint8')
    for i, crater in craters.iterrows():
        x = crater['long']
        y = crater['lat']
        r = crater['diameter']/2
        if r < 80:
            rr, cc = circle(y, x, r)
            try:
                image[rr, cc] = (i+1) * 254/len(craters)
            except:
                pass
    return image


def show_circles():
    circles = FastCircles()
    craters = circles(blob_image)
    result = build_target(craters)

    fig, axes = plt.subplots(ncols=4, figsize=(12, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    fig.suptitle('Problematic Behavior with Circle Procedure')

    ax[0].imshow(blob_input, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Input Image')
    ax[1].imshow(mask, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Target Output')
    ax[2].imshow(prediction, cmap='CMRmap', interpolation='nearest')
    ax[2].set_title('Model Output')
    ax[3].imshow(result, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax[3].set_title('Extracted Circles')

    for a in ax:
        a.set_axis_off()

    plt.show()

def show_watershed():
    distance = ndi.distance_transform_edt(blob_image)
    local_maxi = peak_local_max(distance, indices=False,
                                labels=blob_image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=blob_image)
    watcir = WatershedCircles()
    craters = watcir(blob_image)
    result = build_target(craters)

    fig, axes = plt.subplots(ncols=4, figsize=(12, 3), sharex=True, sharey=True)
    fig.suptitle('Improved Extractions with Watershed Procedure')
    ax = axes.ravel()

    ax[0].imshow(blob_image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Binarized Output')
    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distance Transform')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax[2].set_title('Segmented Basins')
    ax[3].imshow(result, cmap=plt.cm.nipy_spectral, interpolation='nearest')
    ax[3].set_title('Watershed Extractions')

    for a in ax:
        a.set_axis_off()

    plt.show()
    
def show_dbscan():
    pixels = np.argwhere(blob_image)
    #indices = [i for i in range(len(pixels))]
    #idx = np.random.choice(indices, round(len(pixels)/10))
    #pixels = pixels[idx]
    dbscan = DBSCAN(eps=1)
    pred = dbscan.fit_predict(pixels)
    cs = ['r', 'b', 'green', 'yellow', 'cyan', 'violet', 'black', 'orange']
    cs = cs * 100
    print('number of clusters: {}'.format(len(np.unique(pred))))
    fig, ax = plt.subplots()
    for cluster in np.unique(pred):
        indices = np.argwhere(np.where(pred == cluster, 1, 0));
        ax.scatter(x=pixels[indices, 1], y = (-pixels[indices,0]+150), color=cs[cluster]);
    ax = remove_ticks(ax)
    plt.show()
    

def get_classifier_results():
    cda = pycda.CDA(classifier='none')
    prediction = cda.get_prediction(get_sample_image())

    prediction.known_craters = get_sample_csv()
    an = ErrorAnalyzer()
    an.analyze(prediction, verbose=False)
    proposals, craters = an.return_results()

    ground_truth = proposals

    cda.classifier = ConvolutionalClassifier()
    prediction_2 = cda.get_prediction(get_sample_image())
    classification = prediction_2.proposals

    Y_true = ground_truth.positive
    Y_pred = np.where(classification.likelihood > .5, 1, 0)
    return Y_true, Y_pred

