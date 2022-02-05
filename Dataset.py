import pandas as pd
import numpy as np
import scipy.sparse as sp
from pandas.core.frame import DataFrame
import collections
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Dataset:
    def __init__(self):
        self.tw_num = 163684887
        self.user_num = 4907822
        self.topic_num = 20
        self.neg_ratio = 2
        # if (FLAGS.cal_candidate != 0):
        #     self.get_data()
        #     self.get_train_test_data()

    def get_data(self):
        # self.tw2user = np.array(pd.read_csv('data/tw_final/fast/tw2user.csv')['user'], dtype=np.int)
        self.user2tw = list(pd.read_pickle("data/tw_final/fast/user2tw.pkl")['tweets'])
        # self.user2retw = list(pd.read_pickle("data/tw_final/fast/user2retw.pkl")['retweets'])
        self.users_features = np.load("data/tw_final/fast/users_features.npy")
        self.tweets_features = np.load("data/tw_final/fast/tweets_features.npy")
        # self.followers = list(pd.read_pickle("data/tw_final/fast/followers.pkl")['follower'])
        self.followees = list(pd.read_pickle("data/tw_final/fast/followees.pkl")['followee'])

        self.user_feat_dims = self.users_features.shape[1]
        self.tw_feat_dims = self.tweets_features.shape[1]

    def get_train_test_data(self):
        candidate = []

        user_label = np.load("data/tw_final/fast/users_label.npy")
        tw_label = np.load("data/tw_final/fast/tweets_label.npy")
        vectors = np.load("data/tw_final/fast/influence_vector.npy")
        control_user = np.load("data/tw_final/fast/control_users.npy")

        influence = vectors[0]
        origin_vector = vectors[1]
        if (FLAGS.method == 'inf'):
            print("generate influence candidate")
            user1_influences = np.sum(
                (self.users_features - np.expand_dims(origin_vector[:self.user_feat_dims], axis=0)) * np.expand_dims(
                    influence[:self.user_feat_dims], axis=0), axis=1)
            user2_influences = np.sum(
                (self.users_features - np.expand_dims(origin_vector[-self.user_feat_dims:], axis=0)) * np.expand_dims(
                    influence[-self.user_feat_dims:], axis=0), axis=1)
            tw_influences = np.sum(
                (self.tweets_features - np.expand_dims(origin_vector[self.user_feat_dims:-self.user_feat_dims],
                                                       axis=0)) * np.expand_dims(
                    influence[self.user_feat_dims:-self.user_feat_dims], axis=0), axis=1)
        elif (FLAGS.method == 'sim'):
            print("generate similar candidate")
            user_center = np.load("data/tw_final/fast/user_center.npy")
            tw_center = np.load("data/tw_final/fast/tw_center.npy")
            user1_influences = np.sum(self.users_features * user_center[:1], axis=1) / (
                    np.linalg.norm(self.users_features, axis=1) * np.linalg.norm(user_center[:1], axis=1) + 1e-10)
            user2_influences = np.random.rand(len(user1_influences))
            tw_influences = np.sum(self.tweets_features * tw_center[6:7], axis=1) / (
                    np.linalg.norm(self.tweets_features, axis=1) * np.linalg.norm(tw_center[6:7], axis=1) + 1e-10)
        else:
            print("error")
            exit(0)
        select_users1 = np.argsort(-user1_influences[control_user])[:int(len(control_user) // 2.5)]  # 1%
        select_users2 = np.argsort(-user2_influences)[:len(user2_influences) // 50]
        select_tw = np.argsort(-tw_influences)[:len(tw_influences) // 50]

        print(len(select_users1), len(select_users2), len(select_tw))
        user_map = np.zeros(self.user_num)
        tw_map = np.zeros(self.tw_num)
        user_map[select_users1] = 1
        tw_map[select_tw] = 1
        print("begin")
        data_list = [[] for i in range(400)]
        count = 0
        for i, user2 in enumerate(select_users2):
            if (i % 10000 == 0):
                print(i, count)
            followees1 = np.array(self.followees[user2])
            tws1 = np.array(self.user2tw[user2])
            if (len(followees1) > 0 and len(tws1) > 0):
                idx1 = np.where(user_map[followees1] == 1)[0]
                idx2 = np.where(tw_map[tws1] == 1)[0]
                if (len(idx1) > 0 and len(idx2) > 0):
                    followees = followees1[idx1]
                    tws = tws1[idx2]
                    for user in followees:
                        for tw in tws:
                            count += 1
                            data_list[user_label[user] * 20 + tw_label[tw]].append(
                                [user, user2, tw, user1_influences[user] + user2_influences[user2] + tw_influences[tw]])
        train_data = np.load("data/tw_final/fast/train_x.npy")
        train_label = np.load("data/tw_final/fast/train_y.npy")
        idx = np.where(train_label == 1)[0]
        train_data = train_data[idx]
        distribution = np.zeros(400)
        for i in range(len(train_data)):
            user = np.argmax(train_data[i, :20])
            tw = np.argmax(train_data[i, self.user_feat_dims:self.user_feat_dims + 20])
            distribution[user * 20 + tw] += 1
        distribution /= np.sum(distribution)
        idxs = np.random.choice(np.arange(400), 100000, p=distribution)
        for i in range(400):
            if (len(data_list[i]) > 0):
                need = len(np.where(idxs == i)[0])
                if (need <= len(data_list[i])):
                    data_temp = np.array(data_list[i])
                    idx = np.argsort(-data_temp[:, 3])[:need]
                else:
                    idx = list(np.arange(len(data_list[i])))
                    idx1 = list(np.random.choice(np.arange(len(data_list[i])), need - len(data_list[i])))
                    idx += idx1
                for ii in idx:
                    candidate.append(data_list[i][ii])
        print(len(candidate))
        candidate = np.array(candidate)
        np.save("data/tw_final/fast/candidate_%s.npy" % FLAGS.method, candidate)


def process(data, tw2user):
    data1 = np.zeros((len(data), 4), np.int)
    data1[:, 0] = data[:, 0]
    data1[:, 2] = data[:, 1]
    data1[:, 3] = data[:, 2]
    data1[:, 1] = tw2user[data[:, 1]]
    print(len(data1), np.max(data1[:, 0]), np.max(data1[:, 1]), np.max(data1[:, 2]), np.max(data1[:, 3]))
    return data1


# dataset = Dataset()
# tw2user = np.array(pd.read_csv('data/tw_final/fast/tw2user.csv')['user'], dtype=np.int)
# train_data = np.load("data/tw_final/fast/train_data.npy")
# test_data = np.load("data/tw_final/fast/test_data.npy")
# val_data = np.load("data/tw_final/fast/val_data.npy")
# train_data = process(train_data, tw2user)
# test_data = process(test_data, tw2user)
# val_data = process(val_data, tw2user)
# np.save("data/tw_final/fast/train_data.npy", train_data)
# np.save("data/tw_final/fast/test_data.npy", test_data)
# np.save("data/tw_final/fast/val_data.npy", val_data)
