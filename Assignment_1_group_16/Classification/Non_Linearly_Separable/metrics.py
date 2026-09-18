import numpy as np

class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = np.unique(self.y_true)
    
    def mean_squared_error(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())
    
    def percentage_root_mean_squared_error(self):
        return 100 * self.root_mean_squared_error() / np.mean(self.y_true)
    
    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)
    
    def confusion_matrix(self):
        classes = len(self.classes)
        matrix = np.zeros((classes, classes), dtype=int)
        for actual, pred in zip(self.y_true, self.y_pred):
            matrix[actual, pred] += 1
        return matrix
    
    def true_positive(self,class_label):
        return np.sum((self.y_true==class_label) & (self.y_pred == class_label))
    
    def false_positive(self,class_label):
        return np.sum((self.y_true!=class_label) & (self.y_pred == class_label))
    
    def false_negative(self,class_label):
        return np.sum((self.y_true==class_label) & (self.y_pred != class_label))      
    
    def true_negative(self,class_label):
        return np.sum((self.y_true!=class_label) & (self.y_pred != class_label))
    

    def precision(self):
        class_precisions = {}
        
        for c in self.classes:
            tp = self.true_positive(c)
            fp = self.false_positive(c)
            if tp + fp == 0:
                class_precisions[c] = 0.0    
            else:
                class_precisions[c] = tp/(tp+fp)
        
        return class_precisions

    def recall(self):
        class_recalls = {}
                
        for c in self.classes:
            tp = self.true_positive(c)
            fn = self.false_negative(c)
            if tp + fn == 0:
                class_recalls[c] = 0.0    
            else:
                class_recalls[c] = tp/(tp+fn)
        
        return class_recalls

    def f1_score(self):
        precisions = self.precision()
        recalls = self.recall()
        f_scores = {}
        for c in self.classes:
            precision = precisions[c]
            recall = recalls[c]
            if precision==0 and recall==0:
                f_scores[c] = 0.0
            else:
                f_scores[c] = 2*precision*recall/(precision+recall)
                
        return f_scores

    def macro_precision(self):
        precisions = self.precision()
        return np.mean(list(precisions.values()))
    
    def macro_recall(self):
        recalls = self.recall()
        return np.mean(list(recalls.values()))
    
    def macro_f_score(self):
        f_scores = self.f1_score()
        return np.mean(list(f_scores.values()))
    
    def __print_confusion_matrix(self, matrix, labels):
        """
        Prints a confusion matrix to the terminal with dynamic alignment padding.
        
        :param matrix: 2D list or 2D numpy array (similar to scikit-learn output)
        :param labels: List of strings representing class names
        """
        # 1. Determine string lengths for dynamic layout padding
        max_label_len = max(len(str(lbl)) for lbl in labels)
        max_val_len = max(len(str(item)) for row in matrix for item in row)
        col_width = max(max_label_len, max_val_len) + 2
        
        # 2. Print header (Predicted Labels)
        header_title = "Predicted"
        # Center 'Predicted' across the space occupied by columns
        total_cols_width = col_width * len(labels)
        print(f"{' ' * (max_label_len + 5)}{header_title.center(total_cols_width)}")
        
        # Print the specific column labels
        header_row = "".join(f"{str(lbl):>{col_width}}" for lbl in labels)
        print(f"{' ' * (max_label_len + 5)}{header_row}")
        print(f"{' ' * (max_label_len + 4)}┌" + "─" * (total_cols_width + 1))
        
        # 3. Print rows (True Labels vs Matrix Rows)
        is_first_row = True
        for label, row in zip(labels, matrix):
            # Y-axis label alignment logic
            y_axis_title = "True " if is_first_row else "     "
            is_first_row = False
            
            # Format columns string
            row_str = "".join(f"{str(item):>{col_width}}" for item in row)
            
            # Combine everything together cleanly
            print(f"{y_axis_title}{str(label):>{max_label_len}} │ {row_str}")       
    
    def classification_report(self):
        confusion_matrix = self.confusion_matrix() 
        accuracy  = self.accuracy()
        precisions = self.precision()
        recalls = self.recall()
        f_scores = self.f1_score()
        
        print("-"*60)
        print("Confusion Matrix")
        print('-'*60)
        print()
        self.__print_confusion_matrix(confusion_matrix, self.classes)
        print("\n",'-'*60)
        print("Accuracy :", accuracy)
        print('-'*60)
        print(f"     Class        Precision        Recall         F-score    ")
        for c in self.classes:
            print(f"      {c+1}            {precisions[c]:.2f}            {recalls[c]:.2f}            {f_scores[c]:.2f}      ")
        print('-'*60)
        print(f"Macro Precision = {self.macro_precision():.4f}")
        print(f"Macro Recall = {self.macro_recall():.4f}")
        print(f"Macro F-Score = {self.macro_f_score():.4f}")
        
        
        
        