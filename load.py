import json
import numpy as np
import torch
from torch.nn import functional as F

from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor,
import pathlib

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, StanfordCars, ImageFolder
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from datasets import _transform, CUBDataset
from collections import OrderedDict
import clip

from loading_helpers import *

import json, os
import argparse

def parse_training_params(parser):
    parser.add_argument('--model_size', type=str, default='ViT-L/14@336px', help='Model size')
    parser.add_argument('--descriptor_fname', type=str, help='Descriptors to load')
    parser.add_argument('--already_complete_descriptors',type=int,default=0)
    parser.add_argument('--eval_fname', type=str, help='Filename to save evaluation results')
    parser.add_argument('--eval_dir', type=str, help='Dirname to save evaluation results')
    parser.add_argument('--batch_size', type=int , default=64*10)
    parser.add_argument('--xxx', type=int , default=0)
    parser.add_argument('--device', type=str , default='cuda')
    return parser

parser = argparse.ArgumentParser()
parser = parse_training_params(parser)
opt = parser.parse_args()

opt.xxx=bool(opt.xxx)

if os.path.exists(f'/export/home/ru86qer/classify_by_description_release/{opt.eval_dir}') is False:
    os.mkdir(f'/export/home/ru86qer/classify_by_description_release/{opt.eval_dir}')

print(opt.eval_dir)

eval_path = os.path.join(f'/export/home/ru86qer/classify_by_description_release/{opt.eval_dir}',opt.eval_fname+'.json')  # Hallo Test
if os.path.exists(eval_path) is False:
    with open(eval_path, 'w') as fp:
        json.dump({}, fp, indent=4)


print(opt)

hparams = {}
# hyperparameters

hparams['eval_path'] = eval_path

hparams['already_complete_descriptors'] = opt.already_complete_descriptors

hparams['model_size'] = opt.model_size 
# Options:
# ['RN50',
#  'RN101',
#  'RN50x4',
#  'RN50x16',
#  'RN50x64',
#  'ViT-B/32',
#  'ViT-B/16',
#  'ViT-L/14',
#  'ViT-L/14@336px']
hparams['dataset'] = 'cars'

hparams['batch_size'] = opt.batch_size
hparams['device'] = opt.device if torch.cuda.is_available() else "cpu"
hparams['category_name_inclusion'] = 'prepend' #'append' 'prepend'

hparams['apply_descriptor_modification'] = True

hparams['verbose'] = False
hparams['image_size'] = 224
if hparams['model_size'] == 'ViT-L/14@336px' and hparams['image_size'] != 336:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 336.')
    hparams['image_size'] = 336
elif hparams['model_size'] == 'RN50x4' and hparams['image_size'] != 288:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 288
elif hparams['model_size'] == 'RN50x16' and hparams['image_size'] != 384:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 384
elif hparams['model_size'] == 'RN50x64' and hparams['image_size'] != 448:
    print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
    hparams['image_size'] = 448

hparams['before_text'] = ""
hparams['label_before_text'] = ""
hparams['between_text'] = ', '
# hparams['between_text'] = ' '
# hparams['between_text'] = ''
hparams['after_text'] = ''
hparams['unmodify'] = True
# hparams['after_text'] = '.'
# hparams['after_text'] = ' which is a type of bird.'
hparams['label_after_text'] = ''
# hparams['label_after_text'] = ' which is a type of bird.'
hparams['seed'] = 1

# TODO: fix this... defining global variable to be edited in a function, bad practice
# unmodify_dict = {}

# classes_to_load = openai_imagenet_classes
hparams['descriptor_fname'] = None

IMAGENET_DIR = '/proj/vondrick3/datasets/ImageNet/' # REPLACE THIS WITH YOUR OWN PATH
IMAGENETV2_DIR = '/proj/vondrick/datasets/ImageNetV2/' # REPLACE THIS WITH YOUR OWN PATH
CARS_DIR = '/export/scratch/ru86qer/datasets/'
if opt.xxx:
    CUB_DIR = '/export/scratch/ru86qer/datasets/cub_xxx_modified/CUB_200_2011' # REPLACE THIS WITH YOUR OWN PATH
else:
    CUB_DIR = '/export/scratch/ru86qer/datasets/cub_200_complete/CUB_200_2011'

# PyTorch datasets
tfms = _transform(hparams['image_size'])



if hparams['dataset'] == 'imagenet':
    if hparams['dataset'] == 'imagenet':
        dsclass = ImageNet        
        hparams['data_dir'] = pathlib.Path(IMAGENET_DIR)
        # train_ds = ImageNet(hparams['data_dir'], split='val', transform=train_tfms)
        dataset = dsclass(hparams['data_dir'], split='val', transform=tfms)
        classes_to_load = None
    
        if hparams['descriptor_fname'] is None:
            hparams['descriptor_fname'] = 'descriptors_imagenet'
        hparams['after_text'] = hparams['label_after_text'] = '.'
        
    elif hparams['dataset'] == 'imagenetv2':
        hparams['data_dir'] = pathlib.Path(IMAGENETV2_DIR)
        dataset = ImageNetV2(location=hparams['data_dir'], transform=tfms)
        classes_to_load = openai_imagenet_classes
        hparams['descriptor_fname'] = 'descriptors_imagenet'
        dataset.classes = classes_to_load
    
elif hparams['dataset'] == 'cars':
    hparams['data_dir'] = pathlib.Path(CARS_DIR)
    dataset = StanfordCars(root=hparams['data_dir'],download=True,transform=tfms)
    classes_to_load = None
    hparams['descriptor_fname'] = opt.descriptor_fname
    hparams['after_text'] = hparams['label_after_text'] = '.'

elif hparams['dataset'] == 'cub':
    # load CUB dataset
    hparams['data_dir'] = pathlib.Path(CUB_DIR)
    classes_to_load = None#opt.classes_to_load
    dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)
    hparams['descriptor_fname'] = opt.descriptor_fname


hparams['descriptor_fname'] = './cars_descriptors/' + hparams['descriptor_fname']
    

print("Creating descriptors...")

gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load)
label_to_classname = list(gpt_descriptions.keys())


n_classes = len(list(gpt_descriptions.keys()))

def compute_description_encodings(model):
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        tokens = clip.tokenize(v).to(hparams['device'])
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings

def compute_label_encodings(model):
    label_encodings = F.normalize(model.encode_text(clip.tokenize([hparams['label_before_text'] + wordify(l) + hparams['label_after_text'] for l in label_to_classname]).to(hparams['device'])))
    return label_encodings


def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")

def show_from_indices(indices, images, labels=None, predictions=None, predictions2 = None, n=None, image_description_similarity=None, image_labels_similarity=None):
    if indices is None or (len(indices) == 0):
        print("No indices provided")
        return
    
    if n is not None:
        indices = indices[:n]
    
    for index in indices:
        show_single_image(images[index])
        print(f"Index: {index}")
        if labels is not None:
            true_label = labels[index]
            true_label_name = label_to_classname[true_label]
            print(f"True label: {true_label_name}")
        if predictions is not None:
            predicted_label = predictions[index]
            predicted_label_name = label_to_classname[predicted_label]
            print(f"Predicted label (ours): {predicted_label_name}")
        if predictions2 is not None:
            predicted_label2 = predictions2[index]
            predicted_label_name2 = label_to_classname[predicted_label2]
            print(f"Predicted label 2 (CLIP): {predicted_label_name2}")
        
        print("\n")
        
        if image_labels_similarity is not None:
            if labels is not None:
                print(f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name: 
                    print("Predicted label (ours) matches true label")
                else: 
                    print(f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches true label")
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else: 
                    print(f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}")
        
            print("\n")
        
        if image_description_similarity is not None:
            if labels is not None:
                print_descriptor_similarity(image_description_similarity, index, true_label, true_label_name, "true")
                print("\n")
            if predictions is not None:
                if labels is not None and true_label_name == predicted_label_name:
                    print("Predicted label (ours) same as true label")
                    # continue
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label, predicted_label_name, "descriptor")
                print("\n")
            if predictions2 is not None:
                if labels is not None and true_label_name == predicted_label_name2:
                    print("Predicted label 2 (CLIP) same as true label")
                    # continue
                elif predictions is not None and predicted_label_name == predicted_label_name2: 
                    print("Predicted label 2 (CLIP) matches predicted label 1")
                else:
                    print_descriptor_similarity(image_description_similarity, index, predicted_label2, predicted_label_name2, "CLIP")
            print("\n")

def print_descriptor_similarity(image_description_similarity, index, label, label_name, label_type="provided"):
    # print(f"Total similarity to {label_name} ({label_type} label) descriptors: {aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    print(f"Total similarity to {label_name} ({label_type} label) descriptors:")
    print(f"Average:\t\t{100.*aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
    label_descriptors = gpt_descriptions[label_name]
    for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key = lambda x: x[1], reverse=True):
        k = unmodify_dict[label_name][k]
        # print("\t" + f"matched \"{k}\" with score: {v}")
        print(f"{k}\t{100.*v}")
        
def print_max_descriptor_similarity(image_description_similarity, index, label, label_name):
    max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
    label_descriptors = gpt_descriptions[label_name]
    print(f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")
    
def show_misclassified_images(images, labels, predictions, n=None, 
                              image_description_similarity=None, 
                              image_labels_similarity=None,
                              true_label_to_consider: int = None, 
                              predicted_label_to_consider: int = None):
    misclassified_indices = yield_misclassified_indices(images, labels=labels, predictions=predictions, true_label_to_consider=true_label_to_consider, predicted_label_to_consider=predicted_label_to_consider)
    if misclassified_indices is None: return
    show_from_indices(misclassified_indices, images, labels, predictions, 
                      n=n,
                      image_description_similarity=image_description_similarity, 
                      image_labels_similarity=image_labels_similarity)

def yield_misclassified_indices(images, labels, predictions, true_label_to_consider=None, predicted_label_to_consider=None):
    misclassified_indicators = (predictions.cpu() != labels.cpu())
    if true_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (labels.cpu() == true_label_to_consider)
    if predicted_label_to_consider is not None:
        misclassified_indicators = misclassified_indicators & (predictions.cpu() == predicted_label_to_consider)
        
    if misclassified_indicators.sum() == 0:
        output_string = 'No misclassified images found'
        if true_label_to_consider is not None:
            output_string += f' with true label {label_to_classname[true_label_to_consider]}'
        if predicted_label_to_consider is not None:
            output_string += f' with predicted label {label_to_classname[predicted_label_to_consider]}'
        print(output_string + '.')
            
        return
    
    misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
    return misclassified_indices


from PIL import Image
def predict_and_show_explanations(images, model, labels=None, description_encodings=None, label_encodings=None, device=None):
    if type(images) == Image:
        images = tfms(images)
        
    if images.device != device:
        images = images.to(device)
        labels = labels.to(device)

    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    n_classes = len(description_encodings)
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        
        
        dot_product_matrix = image_encodings @ v.T
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    
    show_from_indices(torch.arange(images.shape[0]), images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)


def evaluate_pred_vs_true_acc(pred_vs_ture_acc,label_to_classname,top_k=5):
    predictions = pred_vs_ture_acc[0]
    labels = pred_vs_ture_acc[1]

    eval_dict = {str(label): [] for label in range(0,torch.max(labels).item()+1)}

    for i in range(0, len(predictions)):
        eval_dict[str(labels[i].item())].append(predictions[i].item())
    
    tmp_dict = {k:None for k in eval_dict.keys()}

    for k, v in eval_dict.items():
        eval_dict[k] = np.array(v)
        counts = np.bincount(eval_dict[k],minlength=len(eval_dict.keys()))
        percentages = 100.*counts/eval_dict[k].size
        tmp_dict[k] = percentages.tolist()
    
    for k,v in tmp_dict.items():
        index = int(k)
        tmp_dict[k][index] = -1
    
    flattened = [(k, i, v) for k, values in tmp_dict.items() for i, v in enumerate(values)]

    top_k_misclassifications = sorted(flattened, key=lambda x: x[2], reverse=True)[:top_k]

    for i in range(0,len(top_k_misclassifications)):
        top_k_misclassifications[i] = (label_to_classname[int(top_k_misclassifications[i][0])],"Misclassified as: "+label_to_classname[int(top_k_misclassifications[i][1])],top_k_misclassifications[i][2])
    
    return top_k_misclassifications