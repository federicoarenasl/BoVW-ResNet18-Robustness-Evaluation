from resnet18 import *

# Fine-tune models with learning rate
def train_hyperparameters():
    lrs = [0.01, 0.05, 0.1]
    for split in range(1, 4):
        print(f"#####  On split {split} ######")
        for lr in lrs:
            print(f"Learning rate = {lr}")
            # Instantiate network
            resnet = ResNet18(split, "./data", "./output/resnet18/", 
                                batch_size=50, num_epochs=50, 
                                num_classes=2, training=True, pretrained_model=None, feature_extract=True)

            # Train ResNet18
            lr_name = "_lr_"+str(lr).replace('.','_')
            resnet.run(lr=lr, hyp_name=lr_name)
    
def train_full_splits():
    # Training on lr=0.01 for all full splits
    for split in range(1, 4):
        print(f"#####  On split {split} ######")
        # Instantiate network
        resnet = ResNet18(split, "./data", "./output/resnet18", 
                            batch_size=50, num_epochs=50, 
                            num_classes=2, training=True, pretrained_model=None, feature_extract=True, full_split=True)

        # Train ResNet18
        resnet.run(lr=0.01)

def evaluate_test_splits():
    # Load pre-trained model
    for split in range(1,4):
        print(f"### On Split {split} ###")
        pretrained = torch.load("./output/resnet18/full_split_"+str(split)+"/weights/trained_model.pth")
        # Instantiate network
        input_data_dir = "./data/robustness/5_3/3"
        resnet = ResNet18(1, input_data_dir, "./output/resnet18/", 
                        batch_size=50, num_epochs=1, 
                        num_classes=2, training=False, pretrained_model=pretrained,
                        feature_extract=True, full_split=True)

        # Train ResNet18
        best_acc = resnet.run()
        
        print(f"From Split {split}, the best accuracy was {best_acc}")

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_placeholder_dirs():
    # Create root directory
    robustness_output_path = "./output/resnet18/robustness" 
    create_dir(robustness_output_path)
    # Create list of sub-directory names
    perturb_ids = list(range(1,10))
    # Create all directories
    print("Creating directories...")
    for perturb_id in tqdm(perturb_ids):
        for split in [1,2,3]:
            # Create perturbation id folder
            full_path = robustness_output_path+'/5_'+str(perturb_id)+'/full_split_'+str(split)+'/progress'
            create_dir(full_path)

def evaluate_robustness():
    # Create list of sub-directory names
    perturb_ids = list(range(1,10))
    # Perturbation levels
    perturb_levels = list(range(1,11))
    # Load pre-trained model
    for split in tqdm(range(1,4)):
        print(f"### On Split {split} ###")
        pretrained = torch.load("./output/resnet18/full_split_"+str(split)+"/weights/trained_model.pth")
        # Instantiate network
        for perturb_id in [1]:
            # Initialize list to store accuracies
            best_accs = []
            results_dict = {}
            # Get id folder name
            id_fname = "/5_"+str(perturb_id)
            output_data_dir = "./output/resnet18/robustness/"+id_fname
            for pertur_level in perturb_levels:
                print(f"On perturbation level {pertur_level}, of perturbation id {perturb_id}")
                input_data_dir = "./data/robustness"+id_fname+"/"+str(pertur_level)
                resnet = ResNet18(split, input_data_dir, output_data_dir, 
                                batch_size=50, num_epochs=1, 
                                num_classes=2, training=False, pretrained_model=pretrained,
                                feature_extract=True, full_split=True)

                # Train ResNet18
                best_acc = resnet.run()
                best_accs.append(best_acc)

            # Store results
            results_dict['perturb_level'] = perturb_levels
            results_dict['best_accs'] = best_accs
            results_df = pd.DataFrame.from_dict(results_dict)
            # Ouput results to .csv file
            csv_fname = output_data_dir+"/full_split_"+str(split)+"/robustness_results_5_"+str(perturb_id)+".csv"
            results_df.to_csv(csv_fname, index=False)
        

if __name__ == "__main__":
    evaluate_robustness()

