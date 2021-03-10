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

def train_full_splits(best_cluster_n):
    # Define number of clusters and split to go through
    splits = [1,2,3]

    for split in tqdm(splits):
        print(f"Retreiving histograms from Split {split}...")
        bovw = BovW(split, "./data", full_split=True)
        print("Getting data...")
        train_dict, val_dict = bovw.get_train_val_dict(split)
        print("Getting histograms...")
        bovw.get_all_histograms(best_cluster_n, train_dict, val_dict)

def evaluate_robustness(best_c, best_kernel):
    # Create list of sub-directory names
    perturb_ids = list(range(1,10))
    # Perturbation levels
    perturb_levels = list(range(1,11))
    # Load pre-trained model
    for split in tqdm(range(3,4)):
        print(f"### On Split {split} ###")
        # Get trained histograms
        train_histograms = np.load("./output/bovw/full_split_"+str(split)+"/histograms/train/train_visual_words_k_250.npy")
        train_classes = np.load("./output/bovw/full_split_"+str(split)+"/histograms/train/train_classes_k_250.npy")
        train_centers = np.load("./output/bovw/full_split_"+str(split)+"/histograms/train/train_centers_k_250.npy")
        
        # Instantiate network
        # Skip level 5_4 since something is happening  with Nones, we can debug it ad the end
        if split == 1:
            perturb_ids = [4] # This one is creating problems for all splits, we must debug it
        if split == 2:
            perturb_ids = [4]
        if split == 3:
            perturb_ids = [4]
        #for perturb_id in perturb_ids:
        for perturb_id in [4]:
            # Initialize list to store accuracies
            best_accs = []
            results_dict = {}
            # Get id folder name
            id_fname = "/5_"+str(perturb_id)
            output_data_dir = "./output/bovw/robustness/"+id_fname+"/"
            # Loop through perturbation levels
            for pertur_level in perturb_levels:
            #for pertur_level in [7]:
                print(f"On perturbation level {pertur_level}, of perturbation id {perturb_id}")
                input_data_dir = "./data/robustness"+id_fname+"/"+str(pertur_level)
                bovw = BovW(split, input_data_dir, full_split=True)
                # Get validation information from perturbed images
                val_df = bovw.get_valid_splits(split)
                print("Getting validation images...")
                val_dict = bovw.image_reader(val_df)
                val_descriptor_list, val_sift_vectors = bovw.sift_features(val_dict)
                val_histograms, val_classes = bovw.get_histogram_arrays(val_sift_vectors, train_centers)
                # Train SVM classifier on perturbed images
                svm_classifier = bovw.train_svm(train_histograms, train_classes, best_c, best_kernel)
                # Get train and val accuracies
                train_acc = svm_classifier.score(train_histograms, train_classes)
                val_acc = svm_classifier.score(val_histograms, val_classes)
                # Append accuracy values
                print(f"Accuracy of {val_acc} on 5_{perturb_id} of level {pertur_level}.")
                best_accs.append(val_acc)
            
            # Store results
            results_dict['perturb_level'] = perturb_levels
            results_dict['best_accs'] = best_accs
            results_df = pd.DataFrame.from_dict(results_dict)
            # Ouput results to .csv file
            csv_fname = output_data_dir+"full_split_"+str(split)+"/robustness_results_5_"+str(perturb_id)+".csv"
            print(f"Outputting appending data to {csv_fname}...")
            results_df.to_csv(csv_fname, index=False)

if __name__=="__main__":
    evaluate_robustness(2.5, 'rbf')
