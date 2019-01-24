import pandas as pd
import os.path
import math
from utils import *
import pickle
from sklearn.model_selection import train_test_split

def GenerateUnlabeled(vocabs, batch_size):
    print("Loading the unlabeled dataset")
    patch = pd.read_csv("Data/patch_clean.csv")

    print("Unlabeled set shape:", patch.shape)

    print("Converting unlabeled dataset to batches")

    test = TrainToBags(patch, vocabs, True)
    print("Saving the unlabeled dataset into patch.pkl")

    pickle.dump(test, open("Data/patch.pkl", "wb"))
    return test

def GenerateTrain(batch_size):
    """
    if os.path.isfile("Data/train.csv"):
        df = pd.read_csv("Data/train.csv")
        # should I remove stop words?

    elif os.path.isfile("Data/patch_sample_annotated.csv"):
        propublica = pd.read_csv("Data/propublica.csv")["text"].dropna()

        print("Number of articles in Propublica:", propublica.shape[0])
        print("Getting", count / 2, "random articles from Propublica")
        propublica_sample = propublica.sample(1000).tolist()
        propublica_sample = [re.sub(r"[\s]+", " ", sent) for sent in propublica_sample]
        labels = [1 for i in range(len(propublica_sample))]

        propublica_df = pd.DataFrame.from_dict({"text": propublica_sample, "labels": labels})
        patch_sample = pd.read_csv("Data/patch_sample_annotated.csv")

        df = propublica_df.append(patch_sample).sample(count)
        print("Train set shape:", df.shape)
        df.to_csv("Data/train.csv", index=False)
    else:
        patch = pd.read_csv("Data/patch.csv")["text"].dropna()
        print("Number of articles in Patch:", patch.shape[0])
        print("Getting", count / 2, "random articles from each set")
        patch_sample = patch.sample(math.floor(count / 2)).tolist()

        df = pd.DataFrame.from_dict({"text": patch_sample, "labels": [0 for i in range(len(patch_sample))]})

        print("Output the patch sample to be annotated")
        df.to_csv("Data/patch_sample.csv", index=False)
        print("Please annotate the patch sample first and then save it as 'patch_sample_annotated.csv'")
        exit(1)
    """
    df = pd.read_csv("Data/train_all.csv")
    print("Train set includes", df.shape[0], "annotated data points")

    print("Learning vocabs")
    vocabs = get_vocabs(df["text"].tolist())

    print("Converting articles to bags of sentences")
    bags = TrainToBags(df, vocabs)

    print("Splitting into train and dev set")
    train, dev_test = train_test_split(bags, test_size=0.3, random_state=33)
    test, dev = train_test_split(dev_test, test_size=0.33, random_state=33)

    print("Loading pretrained word embeddings")
    embedding = read_embedding(vocabs)

    print("All datasets are saved in data.pkl")
    pickle.dump((train, dev, test, vocabs, embedding), open("Data/data.pkl", "wb"))
    return (train, dev, test, vocabs, embedding)

