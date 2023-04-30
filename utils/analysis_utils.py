from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from captum.attr import Saliency
import librosa

from utils import SAMPLE_RATE, SAMPLE_LENGTH_MS
from utils.audio_utils import AudioHelper

from tqdm.notebook import tqdm

def gen_confusion_matrix(y_true, y_pred, class_dict):
    classes = [label for (_, label) in sorted(list(class_dict.items()))]

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sns.heatmap(df_cm, annot=True)
    # plt.savefig('output.png')
    plt.show()


def gen_tsne(layers, values, class_dict):
    fig, axs = plt.subplots()
    for layer in layers:
        tsne = TSNE(n_components=2)
        z = tsne.fit_transform(values[layer])
        
        df = pd.DataFrame()
        df["y"] = values['label']
        df["label"] = df['y'].map(class_dict)
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]
        ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.label.tolist(),
                        palette=sns.color_palette("husl", 9),
                        data=df,
                        style=df.label.tolist())
        ax.set(title=f"{layer} Layer Activation T-SNE Projection")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.show()

def get_module_by_name(model, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, model)

def eval(model, loader, layers=[]):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if (type(output) == tuple):
                output = output[0]
            activation[name] = output.detach().to('cpu')
        return hook
    
    hooks = {}
    for layer in layers:
        hooks[layer] = get_module_by_name(model, layer).register_forward_hook(get_activation(layer))

    model.eval()

    values = {'label': [], 'pred': []}
    for layer in layers:
        values[layer] = None

    with torch.no_grad():
        progress_bar = tqdm(loader, ascii=True)

        for i, (inputs, labels) in enumerate(progress_bar):
            outputs = model(inputs).cpu()

            _, prediction = torch.max(outputs,1)
            values['pred'].extend(prediction.numpy())
            values['label'].extend(labels.numpy())

            for layer in layers:
                actv = activation[layer]
                if values[layer] == None:
                    values[layer] = actv
                else:
                    values[layer] = torch.cat((values[layer], actv), dim=0)

    for _, hook in hooks.items():
        hook.remove()

    return values

def gen_saliency(model, input_file):
    # Load data for model
    audio, sr = librosa.load(input_file, sr=SAMPLE_RATE)
    audio, sr = AudioHelper.pad_trunc((audio, sr), SAMPLE_LENGTH_MS)
    audio = torch.tensor(audio, requires_grad=True)
    audio = audio.unsqueeze(0)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    saliency = Saliency(model)
    attribution = saliency.attribute(audio, target=3)
    return attribution

def get_input_from_file(file):
    audio, sr = librosa.load(file, sr=SAMPLE_RATE)
    audio = torch.Tensor(audio)
    audio, sr = AudioHelper.pad_trunc((audio, sr), SAMPLE_LENGTH_MS)
    audio = audio.unsqueeze(0)
    return audio

def get_content_loss(model, source, target):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            if (type(input) == tuple):
                input = input[0]
            activation[name] = input.detach().to('cpu')
        return hook
    
    hook = get_module_by_name(model, 'linear1').register_forward_hook(get_activation('linear1'))
    model.eval()

    model(source)
    src_content = activation['linear1']
    model(target)
    target_content = activation['linear1']

    score = torch.nn.functional.cosine_similarity(src_content, target_content)

    hook.remove()
    return score