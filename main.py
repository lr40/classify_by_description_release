from load import *
import torchmetrics
from tqdm import tqdm


seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)

label_encodings = compute_label_encodings(model)


print("Evaluating...")
lang_accuracy_metric = torchmetrics.Accuracy().to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

clip_accuracy_metric = torchmetrics.Accuracy().to(device)
clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

pred_vs_true_acc = torch.zeros(2,len(dataset)).to(device)

for batch_number, batch in enumerate(tqdm(dataloader)):
    images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)
    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    
    clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
    clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
    
    
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        
        
        dot_product_matrix = image_encodings @ v.T
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])  #Jedes i steht für eine Klasse, innerhalb dieser Klasse wird die Ähnlichkeit über die verschiedenen Deskriptoren aggregiert
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    
    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)

    start_idx = batch_number * dataloader.batch_size
    end_idx = start_idx + batch[1].size(0)  # inputs.size(0) is current batch size
    
    # Populate the pred_vs_true_acc tensor
    pred_vs_true_acc[0, start_idx:end_idx] = descr_predictions
    pred_vs_true_acc[1, start_idx:end_idx] = labels
    
top_k_misclassifications = evaluate_pred_vs_true_acc(pred_vs_true_acc.int(),label_to_classname,top_k=(30 if 30 <= len(label_to_classname) else len(label_to_classname)))   

print("\n")

accuracy_logs = {}
accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

accuracy_logs["info"] = hparams['model_size'],hparams['descriptor_fname']

with open(eval_path, 'r') as fp:
    data = json.load(fp)

data[hparams['model_size']] = accuracy_logs
data[hparams['model_size']+' top_k_misclassifications'] = top_k_misclassifications

# print the accuracy logs into the json haparams['eval_path']
with open(hparams['eval_path'], 'w') as fp:
    json.dump(data, fp, indent=4)

# print the dictionary
print("\n")
for key, value in accuracy_logs.items():
    print(key, value)