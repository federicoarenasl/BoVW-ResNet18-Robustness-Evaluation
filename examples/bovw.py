"""

BOVW and SVM Classification task.

For the time being the main points of investigation will be finding the right number of clusters
and using the appropriate feature extractor.

"""         
# Load the train and validation data into the dictionary
                    
from utils import Utils



train = Utils.load_images_from_folder('fold0')
validate = Utils.load_images_from_folder("fold1")


# Extract features from images using a desired feature extractor
# sift_features returns first a list of all extracted features, whilst the 
# second output is features divided by class
sifts = Utils.dense_sift_features(train) 
descriptor_list = sifts[0] 
all_bovw_feature = sifts[1] 
test_bovw_feature = Utils.dense_sift_features(validate)[1]


print("Extracting visual words with kmeans.")
# Use kmeans clustering to create our visual descriptors
c_values = [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
kernel_values = ['poly', 'rbf']
centers = [150, 175, 200]
for center in centers:
    visual_words = Utils.kmeans(center, descriptor_list)
    
    
    # Creates histograms for train data    
    bovw_train = Utils.image_class(all_bovw_feature, visual_words) 
    # Creates histograms for test data
    bovw_valid = Utils.image_class(test_bovw_feature, visual_words) 
    
    
    # Convert the training data for x,y pairs
    # 0 is cats and 1 is dogs
    X_train, y_train  = Utils.convert(bovw_train)
    X_valid, y_valid  = Utils.convert(bovw_valid)
    
    # Train the SVM
    for c in c_values:
        for k in kernel_values:
            from sklearn.svm import SVC
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            
            
            clf = make_pipeline(StandardScaler(), SVC(C=c, kernel=k, gamma='auto'))
            clf.fit(X_train, y_train)
            if (clf.score(X_valid, y_valid) > 0.86):
                print("Using this C value %f and this kernel %s" % (c, k))
                print(center)
                print("Train Score: ")
                print(clf.score(X_train, y_train))
                print("Validation Score: ")
                print(clf.score(X_valid, y_valid))
                


