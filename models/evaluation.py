#evaluation.py

    def _print_performance_across_folds(self):
        """Print the accuracy, AUROC, and average precision for the cross
        validation folds."""
        # print the individual accuracies and the average accuracy
        print("\n\n The accuracies of the cross validation folds are:\n")
        tot_acc = 0
        for k in range(len(fold_acc)):
            print("Fold ",str(k+1), ": ", str(fold_acc[k]))
            tot_acc += fold_acc[k]
        # print the individual accuracies and the average accuracy
        print("\n\n The auroc of the cross validation folds are:\n")
        tot_auroc = 0
        for k in range(len(fold_auroc)):
            print("Fold ",str(k+1), ": ", str(fold_auroc[k]))
            tot_auroc += fold_auroc[k]
        # print the individual accuracies and the average accuracy
        print("\n\n The average precision of the cross validation folds are:\n")
        tot_avg_prec = 0
        for k in range(len(fold_avg_prec)):
            print("Fold ",str(k+1), ": ", str(fold_avg_prec[k]))
            tot_avg_prec += fold_avg_prec[k]
        
        # get the overall metrics
        print('The overall mean accuracy is', tot_acc/self.cv_fold_mlp)
        print('The overall mean AUROC is',tot_auroc/self.cv_fold_mlp)
        print('The overall mean Average Precision is',tot_avg_prec/self.cv_fold_mlp)
        print('Done')