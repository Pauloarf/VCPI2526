from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix



def show_history_plus(history, fields):

    # summarize history for accuracy
    for hist in fields:

        plt.plot(history[hist])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
    
    plt.legend(list(history.keys()), loc='lower right')
    plt.show()



def show_history(history):

    for key in history.keys():
    
        # summarize history for accuracy
        plt.plot(history[key])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.show()

def show_history_2x2(history):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axs[0, 0].plot(history['accuracy'], color='blue')
    axs[0, 0].set_title('Training Accuracy')
    axs[0, 0].set_ylabel('Accuracy (%)')
    
    axs[0, 1].plot(history['val_acc'], color='orange')
    axs[0, 1].set_title('Validation Accuracy')
    
    # Loss
    axs[1, 0].plot(history['loss'], color='green')
    axs[1, 0].set_title('Training Loss')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_xlabel('Epoch')
    
    axs[1, 1].plot(history['val_loss'], color='red')
    axs[1, 1].set_title('Validation Loss')
    axs[1, 1].set_xlabel('Epoch')
    
    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.show()

def show_histories(histories, model_names, field='accuracy'):

    # summarize history for accuracy
    for hist in histories:
        plt.plot(hist[field])

    plt.ylabel(field)
    plt.xlabel('epoch')
    plt.legend(model_names, loc='lower right')
    plt.show()


def show_accuracies(train_acc, eval_acc, model_names): 
    fig, ax = plt.subplots()
    X = np.arange(len(model_names))

    minT = min(train_acc)
    minE = min(eval_acc)
    min_val = min([minT, minE])
    
    plt.bar(X, eval_acc, width = 0.4, color = 'b', label='eval')
    plt.bar(X + 0.4, train_acc, color = 'r', width = 0.4, label = "train")
    plt.xticks(X + 0.4 / 2, model_names)
    plt.ylim(top = 100, bottom = min_val - 2)
    plt.legend(loc='lower right')
    plt.show()    


def show_histogram(data, classes):

    target_np = [x.numpy().item() for x in data]
    res = Counter(target_np)
    
    print(res)

    values = [res[x] for x in range(len(classes))]
    indexes = np.arange(len(classes))

    plt.figure(figsize=(14, 6))
    plt.bar(indexes, values, width=0.7, edgecolor='black', linewidth=0.5)
    plt.xticks(indexes, classes, rotation=45, ha='right', fontsize=9)
    plt.tight_layout()    
    plt.show() 


def show_confusion_matrix(ground_truth, preds, num_classes):    

    cf_matrix = confusion_matrix(ground_truth, preds)


    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], range(num_classes), range(num_classes))
    plt.figure(figsize=(12,6))

    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10} , fmt='.3f') # font size

    plt.show()

def show_filtered_confusion_matrix(ground_truth, preds, classes):    
    # 1. Calcular a matriz de confusão bruta
    cf_matrix = confusion_matrix(ground_truth, preds)
    
    # 2. Criar uma cópia e zerar a diagonal para isolar apenas os erros
    error_matrix = cf_matrix.copy()
    np.fill_diagonal(error_matrix, 0)
    
    # 3. Encontrar os índices das classes que tiveram erros (na linha ou na coluna)
    # Row sum > 0 significa que a classe foi confundida com outra
    # Col sum > 0 significa que outra classe foi erradamente classificada como esta
    error_indices = np.where((error_matrix.sum(axis=1) > 0) | (error_matrix.sum(axis=0) > 0))[0]
    
    if len(error_indices) == 0:
        print("Perfeito! O modelo não cometeu nenhum erro no conjunto de teste.")
        return

    # 4. Filtrar a matriz original e a lista de nomes das classes
    filtered_matrix = cf_matrix[error_indices][:, error_indices]
    filtered_class_names = [classes[i] for i in error_indices]
    
    # 5. Normalizar a matriz filtrada (proporção por linha - Recall)
    # Adicionamos um pequeno valor (1e-9) para evitar divisões por zero se uma classe tiver 0 amostras
    row_sums = filtered_matrix.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 
    df_cm = pd.DataFrame(filtered_matrix / row_sums, index=filtered_class_names, columns=filtered_class_names)
    
    # 6. Ajustar o tamanho do gráfico dinamicamente com base no número de classes com erro
    num_error_classes = len(error_indices)
    plt.figure(figsize=(max(10, num_error_classes * 0.5), max(8, num_error_classes * 0.4)))
    
    # 7. Plotar o Heatmap apenas com o essencial
    # Usamos o cmap 'Blues' ou 'Oranges' para destacar visualmente onde estão os erros fora da diagonal
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 9}, fmt='.2f', cmap='Blues')
    
    plt.title("Matriz de Confusão Filtrada (Apenas classes com erros)", fontsize=12, pad=15)
    plt.ylabel("Classe Real", fontsize=10)
    plt.xlabel("Classe Prevista", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    

def show_loaded_images(rows, cols, data,classes):

    width= 2 * rows
    height= 2 * cols

    fig, axes= plt.subplots(rows,cols,figsize=(height,width))

    for a in range(rows*cols):
        img, label = data[a]
        subplot_title=(classes[label])
        axes.ravel()[a].set_title(subplot_title)  
        axes.ravel()[a].imshow(np.transpose(img.numpy(), (1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')
    
    fig.tight_layout()    
    plt.show() 


def show_transformed_images(rows, cols, data, classes):

    width= 2 * rows
    height= 2 * cols

    fig, axes= plt.subplots(rows,cols,figsize=(height,width))

    for a in range(rows*cols):
        img, target = data[a]
        subplot_title=(classes[target])
        axes.ravel()[a].set_title(subplot_title)  
        axes.ravel()[a].imshow(np.transpose(img.numpy(),(1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')

    fig.tight_layout()    
    plt.show()         

def show_predicted_images(rows, cols, data, classes):

    width= 2 * rows
    height= 2 * cols

    fig, axes= plt.subplots(rows,cols,figsize=(height,width))

    for a in range(rows*cols):
        img = data[a]
        axes.ravel()[a].imshow(np.transpose(img.numpy(),(1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')

    fig.tight_layout()    
    plt.show()         

def show_images(title, rows, cols, images, targets, classes):

    width= 2 * rows
    height= 2 * cols

    fig, axes= plt.subplots(rows,cols,figsize=(height,width))

    for a in range(rows*cols):
        img, target = images[a], targets[a]
        subplot_title=(classes[target])
        axes.ravel()[a].set_title(subplot_title)  
        axes.ravel()[a].imshow(np.transpose(img.numpy(),(1,2,0)), cmap=plt.cm.gray)
        axes.ravel()[a].axis('off')
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()    
    plt.show()      


def plot_image(i, predictions_array, true_label, img, classes):

    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])  
    img_np = np.transpose(img.numpy(), (1,2,0))
    plt.imshow(img_np, cmap=plt.cm.gray)  
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red' 
    plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                  100*np.max(predictions_array),
                                  classes[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label, num_classes):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(num_classes))
    plt.yticks([])
    thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)  
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

def plot_predictions(images, predictions, ground_truth, classes, num_rows= 5, num_cols=3 ):

    num_images = min(num_rows*num_cols, len(predictions))
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], ground_truth, images, classes)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], ground_truth, len(classes))
    plt.tight_layout()
    plt.show()


