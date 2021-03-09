def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    #print(prediction)
    #print(ground_truth)
    
    tp=0
    fp=0
    fn=0
    tn=0
    
    for i in range(len(prediction)):        
        if prediction[i] == 0 and ground_truth[i] ==0:
            tn+=1
        elif prediction[i] == 1 and ground_truth[i] ==0:
            fp+=1
        elif prediction[i] == 1 and ground_truth[i] ==1:
            tp+=1
        elif prediction[i] == 0 and ground_truth[i] ==1:
            fn+=1
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(fp+fn+tp+tn)
    f1 = 2*precision*recall/(precision+recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    t=0
    f=0
    
    for i in range(len(prediction)):        
        if prediction[i] == ground_truth[i] :
            t+=1
        else:
            f+=1
    
    return t/(t+f)
