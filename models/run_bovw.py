# Import bovw class
from bovw import *
from tqdm import tqdm

def retreive_histograms():
    # Define number of clusters and split to go through
    clusters = [150, 175, 200]
    splits = [1,2,3]

    for split in tqdm(splits):
        print(f"Retreiving histograms from Split {split}...")
        for cluster in clusters:
            print(f"On cluster {cluster}...")
            # Initialize BovW model
            bovw = BovW(split, "./data")

            # Get all histograms for all clusters
            bovw.get_all_histograms(cluster)

if __name__=="__main__":
    retreive_histograms()
