# Author: Damandeep Riat
# Utility Functions and Classes for neural network model analysis and plotting routines
# For use in the MNist_Digit_Classification.ipynb project ONLY




class calculate_metrics():
    """ Calculates the accuracy, precision, recall, and F1 score of a model classified on a given dataset
    
    Args:
        model (tensorflow.keras.models.*****) : The model to analyze metrics on
        X_cv  (ndarray (N,F)) : Dataset to run predictions on 
        y_cv  (ndarray (N,))  : Truth values for the above dataset

    Access Parameters:
        accuracy (scalar): The accuracy metric for the model
        F1_score (dict): dict containing class precision, recall, and F1 scores along with macro and micro weighted values
    """
    def __init__(self,model, X_cv, y_cv):
        self.model = model
        self.X_cv = X_cv
        self.y_cv = y_cv
        
        # convert raw logit results from output layer into a probability distribution
        results = tf.nn.softmax(model.predict(X_cv))
        # Find largest probability class and have model classify it as final prediction
        preds = np.argmax(results, axis = 1)
        self.y_pred = preds
        self.compute_metrics()
        
        

    def compute_accuracy(self):
        correct = sum(prediction == target for prediction , target in zip(self.y_pred,self.y_cv))
        accuracy = correct / self.y_cv.shape[0]
        return accuracy

    def compute_F1_score(self):
        num_classes = 10
        # Step 1: Create confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(self.y_cv, self.y_pred):
            confusion_matrix[true][pred] += 1

    # Initialize variables for macro and micro calculations
        precision_per_class = []
        recall_per_class = []
        TP = 0
        FP = 0
        FN = 0

        for k in range(num_classes):
            TP_k = confusion_matrix[k][k]
            FP_k = np.sum(confusion_matrix[:, k]) - TP_k
            FN_k = np.sum(confusion_matrix[k, :]) - TP_k
    
            precision_k = TP_k / (TP_k + FP_k) if (TP_k + FP_k) > 0 else 0
            recall_k = TP_k / (TP_k + FN_k) if (TP_k + FN_k) > 0 else 0
    
            precision_per_class.append(precision_k)
            recall_per_class.append(recall_k)
    
            TP += TP_k
            FP += FP_k
            FN += FN_k

        # Macro-Averaged Precision and Recall
        macro_precision = np.mean(precision_per_class)
        macro_recall = np.mean(recall_per_class)
    
        # Micro-Averaged Precision and Recall
        micro_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        micro_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
        # F1-Score
        F_1 = 1/ (((1/ np.array(precision_per_class)) + (1 / np.array(recall_per_class))) * 0.5)
    
        return {
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "F1_score": F_1,
            "average_F1_score": np.mean(F_1)
        }

    def compute_metrics(self):
        # convert raw logit results from output layer into a probability distribution
        self.accuracy = self.compute_accuracy()
        self.F1_score = self.compute_F1_score()

        return self.accuracy,self.F1_score
    
