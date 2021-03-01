from resnet18 import *


# Fine tune ResNet18
# Load pre-trained model
#pretrained = torch.load("./output/resnet18/split_1/weights/trained_model.pth")

# Fine-tune models with learning rate
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
    
