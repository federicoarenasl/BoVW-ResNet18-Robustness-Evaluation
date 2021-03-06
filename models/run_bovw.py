# Import bovw class
from bovw import *
from tqdm import tqdm

def retreive_histograms():
    # Define number of clusters and split to go through
    clusters = [50, 100, 150, 200, 250, 300]
    splits = [1,2,3]

    for split in tqdm(splits):
        print(f"Retreiving histograms from Split {split}...")
        bovw = BovW(split, "./data")
        print("Getting data...")
        train_dict, val_dict = bovw.get_train_val_dict(split)
        if split == 1:
            print("On left overs from split 1...")
            for c in [250]:
                print("On left overs from Split 1, cluser", c)
                # Get all histograms for all clusters
                bovw.get_all_histograms(c, train_dict, val_dict)
            continue
        for cluster in clusters:
            print(f"On cluster {cluster}...")
            # Initialize BovW model
            # Get all histograms for all clusters
            bovw.get_all_histograms(cluster, train_dict, val_dict)

def retrieve_accuracies():
    # Define hyperparameter's value range
    c_range =np.arange(0.5, 100.5, 2)
    clusters = [50, 100, 150, 200, 250, 300]
    kernels = ['linear', 'poly', 'rbf']
    # Run through splits and clusters
    for split in tqdm(range(1,4)):
        # Loop through kernels
        for k in tqdm(kernels):
            # Define place holder for accuracie values
            df_train_dict = {}
            df_val_dict = {}
            # Loop through clusters
            for cluster in tqdm(clusters):
                # Load data
                train_histograms = np.load("output/bovw/split_"+str(split)+"/histograms/train/train_visual_words_k_"+str(cluster)+".npy")
                train_classes = np.load("output/bovw/split_"+str(split)+"/histograms/train/train_classes_k_"+str(cluster)+".npy")
                val_histograms = np.load("output/bovw/split_"+str(split)+"/histograms/val/val_visual_words_k_"+str(cluster)+".npy")
                val_classes = np.load("output/bovw/split_"+str(split)+"/histograms/val/val_classes_k_"+str(cluster)+".npy")
                # Cluster accuracies
                train_accuracies = []
                val_accuracies = []
                # Loop through c values
                c_vals = []
                for c in tqdm(c_range):
                    c_vals.append(c)
                    # Get SVM classifier
                    bovw = BovW(split, "./data")
                    svm_classifier = bovw.train_svm(train_histograms, train_classes, c, k)
                    # Get train and val accuracies
                    train_acc = svm_classifier.score(train_histograms, train_classes)
                    val_acc = svm_classifier.score(val_histograms, val_classes)
                    # Append accuracy values
                    train_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)

                # Store accuracies in dictionary
                df_train_dict['c'] = c_vals
                df_train_dict[str(cluster)] = train_accuracies
                df_val_dict['c'] = c_vals
                df_val_dict[str(cluster)] = val_accuracies
            
            # Store dataframes
            df_train = pd.DataFrame.from_dict(df_train_dict) 
            df_val = pd.DataFrame.from_dict(df_val_dict)

            # Save dataframes
            df_train.to_csv("./output/bovw/split_"+str(split)+"/svm_results/train_acc_"+str(k)+".csv")
            df_val.to_csv("./output/bovw/split_"+str(split)+"/svm_results/val_acc_"+str(k)+".csv")


if __name__=="__main__":
    retrieve_accuracies()
