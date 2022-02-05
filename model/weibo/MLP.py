import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import sklearn

FLAGS = tf.flags.FLAGS


class MLP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.load_data()
        self.create_init()

    def create_placeholder(self):
        self.input1 = tf.placeholder(dtype=tf.int32)
        self.input2 = tf.placeholder(dtype=tf.int32)
        self.label = tf.placeholder(dtype=tf.float32)
        self.feature_tensor = tf.placeholder(dtype=tf.float32)
        self.w = (tf.get_variable('w', [self.users_features.shape[0], self.users_features.shape[1]]))
        f1 = tf.gather(self.w, self.input1)
        f2 = tf.gather(self.w, self.input2)
        self.input = tf.stop_gradient(tf.concat([f1, f2], axis=1))

    def create_weights(self, layers=[128, 16, 1]):
        self.weights = []
        self.biases = []
        w_init = tf.glorot_normal_initializer()
        b_init = tf.zeros_initializer()
        pre_layer = self.feature_num
        for i, layer in enumerate(layers):
            weight = tf.get_variable('w%d' % i, [pre_layer, layer], initializer=w_init)
            bias = tf.get_variable('b%d' % i, [layer], initializer=b_init)
            self.weights.append(weight)
            self.biases.append(bias)
            pre_layer = layer

    def create_model(self, input):
        output = input
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            output = tf.matmul(output, weight) + bias
            output = tf.nn.relu(output)
        output = tf.sigmoid(tf.matmul(output, self.weights[-1]) + self.biases[-1])
        return output

    def create_optimizer(self):
        self.score = self.create_model(self.input)
        self.loss = tf.reduce_mean(
            -self.label * tf.log(self.score + 1e-10) - (1. - self.label) * tf.log(1. - self.score + 1e-10))
        self.reg_loss = tf.add_n([tf.nn.l2_loss(weight) for weight in self.weights + self.biases])
        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_op = self.optimizer.minimize(self.loss + self.reg_loss * 0.00001, name='optimizer')

    def create_init(self):
        self.create_placeholder()
        self.create_weights()
        self.create_optimizer()
        self.build_influence()

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, epochs, batch_size, poisoning):
        if (FLAGS.train == 1 or poisoning == 1):
            print("begin train")
            self.sess.run(self.w.assign(self.feature_tensor), feed_dict={self.feature_tensor: self.users_features})
            for i in range(epochs):
                idx_batches = self.get_batches(self.train_data, batch_size)
                for j, idx in enumerate(idx_batches):
                    label = self.train_label[idx]
                    self.sess.run(self.train_op, feed_dict={self.input1: self.train_data[idx, 0],
                                                            self.input2: self.train_data[idx, 1], self.label: label})
                train_acc, train_f1, _ = self.eval_batch(self.train_data, self.train_label, batch_size)
                test_acc, test_f1, _ = self.eval_batch(self.test_data, self.test_label, batch_size)
                target_acc, target_f1, target_pred = self.eval_batch(self.target_data, self.target_label, batch_size)
                print("epoch: ", i, train_acc, train_f1, test_acc, test_f1, target_acc, target_f1)
            if (poisoning != 1):
                self.saver.save(self.sess, 'save_model/weibomlp.ckpt')
            np.save("result/%s_%s_%.2f_pred.npy" % (FLAGS.model, FLAGS.method, FLAGS.data_size), np.array(target_pred))
        else:
            # self.saver.restore(self.sess, 'save_model/weibomlp.ckpt')
            print("load model")
        if (FLAGS.save):
            with open("result/%s_%s_%.2f.txt" % (FLAGS.model, FLAGS.method, FLAGS.data_size), 'a+') as f:
                f.write("%f %f\n" % (test_acc, target_acc))
        if (FLAGS.cal_inf):
            self.sess.run(self.w.assign(self.feature_tensor), feed_dict={self.feature_tensor: self.users_features})
            inf, origin_vector = self.influence_vector()
            np.save("data/weibo/fast/influence_vector.npy", np.array([inf, origin_vector]))
        if (FLAGS.cal_cand):
            self.cal_candidate()

    def get_features(self, data):
        feature1 = self.users_features[data[:, 0]]
        feature2 = self.users_features[data[:, 1]]
        return np.concatenate([feature1, feature2], axis=1)

    def get_actions(self, data):
        data_list = []
        for i in range(len(data)):
            data_list.append(self.users_actions['%d-%d' % (data[i, 0], data[i, 1])])
        return np.array(data_list)

    def get_batches(self, train_data, batch_size):
        idx = np.arange(len(train_data))
        np.random.shuffle(idx)
        data_list = []
        for i in range(len(idx[:-1]) // batch_size + 1):
            cur_idx = idx[i * batch_size:min((i + 1) * batch_size, len(idx))]
            data_list.append(cur_idx)
        return data_list

    def eval_data(self, data, label):
        prediction = self.sess.run(self.score,
                                   feed_dict={self.input1: data[:, 0], self.input2: data[:, 1]})
        acc = np.mean((prediction > 0.5) == label)
        if (np.mean(label) == 0 or np.mean(label) == 1):
            label[0] = 1 - label[0]
        f1 = sklearn.metrics.average_precision_score(label[:], prediction[:])
        return acc, f1, prediction

    def eval_batch(self, data, label, batch_size=10000000):
        idx = np.arange(len(data))
        data1 = data[idx]
        label1 = label[idx]
        acc_list = []
        f1_list = []
        pred_list = []
        for i in range((len(data) - 1) // batch_size + 1):
            cur_data = data1[i * batch_size:(i + 1) * batch_size]
            cur_label = label1[i * batch_size:(i + 1) * batch_size]
            acc, f1, pred = self.eval_data(cur_data, cur_label)
            acc_list.append(acc)
            f1_list.append(f1)
            pred_list += list(pred)
        return np.mean(acc_list), np.mean(f1_list), pred_list

    def build_influence(self):
        with tf.variable_scope('influence'):
            self.params = self.weights + self.biases
            self.attack_loss = tf.reduce_sum(self.score)

            self.scale = 100.

            dty = tf.float32
            self.v_cur_est = [tf.placeholder(dty, shape=a.get_shape(), name="v_cur_est" + str(i)) for i, a in
                              enumerate(self.params)]
            self.Test = [tf.placeholder(dty, shape=a.get_shape(), name="test" + str(i)) for i, a in
                         enumerate(self.params)]

            hessian_vector_val = utils.hessian_vector_product(self.loss, self.params, self.v_cur_est, True, 10000000.)
            self.estimation_IHVP = [g + cur_e - HV
                                    for g, HV, cur_e in zip(self.Test, hessian_vector_val, self.v_cur_est)]

            self.attack_grad = tf.gradients(self.attack_loss, self.params)
            self.per_loss = self.loss

    def influence_vector(self):
        import time
        self.build_influence()
        i_epochs = 20000
        batch_size = 8192

        # IHVP
        start_time = time.time()
        idx = np.random.choice(np.arange(len(self.target_data)), 300000, False)
        optimized_data = self.target_data[idx]

        print(len(optimized_data))
        feed_dict = {self.input1: optimized_data[:, 0], self.input2: optimized_data[:, 1]}
        test_val = self.sess.run(self.attack_grad, feed_dict)

        cur_estimate = test_val.copy()
        feed1 = {place: cur for place, cur in zip(self.Test, test_val)}
        for j in range(i_epochs):
            feed2 = {place: cur for place, cur in zip(self.v_cur_est, cur_estimate)}
            idx = np.random.choice(np.arange(len(self.train_data)), batch_size)
            feed_dict = {self.input1: self.train_data[idx, 0], self.input2: self.train_data[idx, 1],
                         self.label: self.train_label[idx]}
            cur_estimate = self.sess.run(self.estimation_IHVP, feed_dict={**feed_dict, **feed1, **feed2})
            if (j % 1000 == 0):
                print(np.max(cur_estimate[0][0]))
        inverse_hvp1 = [b / self.scale for b in cur_estimate]
        print(np.max(inverse_hvp1[0]))
        duration = time.time() - start_time
        print('Inverse HVP by HVPs+Lissa: took %s minute %s sec' % (duration // 60, duration % 60))

        val_lissa = []
        mean_data = self.train_data[-1:]
        mean_label = np.mean(self.train_label, axis=0, keepdims=True)
        mean_label = np.ones_like(mean_label)
        feed_dict = {self.input1: mean_data[:, 0], self.input2: mean_data[:, 1], self.label: mean_label}
        feed2 = {place: cur for place, cur in zip(self.v_cur_est, inverse_hvp1)}
        pert_vector_val = utils.pert_vector_product(self.per_loss, self.params, self.input,
                                                    self.v_cur_est, True)

        lissa_list = self.sess.run(pert_vector_val, feed_dict={**feed_dict, **feed2})
        temp = -np.sum(np.concatenate(lissa_list, axis=0), axis=0)
        val_lissa.append(temp)
        return temp, mean_data[0]

    def load_data(self):
        all_data = pd.read_csv("data/weibo/fast/rec_log_train.csv")
        idx = np.load("data/weibo/fast/observed/data_1.00.npy")
        data = np.array(all_data[['0', '1']])
        labels = np.array(all_data['2'])[:, None]
        labels = (labels + 1) / 2
        data = data[idx]
        labels = labels[idx]
        idx1 = np.where(labels[:, 0] == 1)[0]
        idx0 = np.where(labels[:, 0] == 0)[0][:len(idx1) * 2]
        idx = np.concatenate([idx1, idx0])
        np.random.seed(100)
        np.random.shuffle(idx)

        self.users_features = np.load("data/weibo/fast/users_features.npy")
        users_words = np.load("data/weibo/fast/users_words.npy")
        users_tags = np.load("data/weibo/fast/users_tags.npy")
        self.users_features = np.concatenate([self.users_features, users_tags, users_words], axis=1)
        if ((FLAGS.poison == 1 and FLAGS.method == 'inf' and FLAGS.modify_user) or FLAGS.method == 'mod'):
            print("load influence feature")
            self.users_features = np.load("data/weibo/fast/partial/users_features_inf_%.2f.npy" % FLAGS.data_size)[:-2]

        import collections
        self.users_actions = collections.defaultdict(lambda: np.zeros(3))
        actions = pd.read_csv("data/weibo/fast/user_action.csv")
        max_action = np.array(actions[['2', '3', '4']])
        max_action = np.max(max_action, axis=0)
        for action in actions.itertuples():
            self.users_actions["%d-%d" % (action[1], action[2])] = np.array(
                [action[3] / max_action[0], action[4] / max_action[1], action[5] / max_action[2]])
        self.feature_num = self.users_features.shape[1] * 2
        self.user_feature_num = self.users_features.shape[1]
        val_idx = idx[int(len(idx) * 0.7):int(len(idx) * 0.8)]
        test_idx = idx[int(len(idx) * 0.8):]
        train_idx = idx[:int(len(idx) * 0.7)]

        self.train_data = data[train_idx]
        self.train_label = labels[train_idx]
        self.val_data = data[val_idx]
        self.val_label = labels[val_idx]
        self.test_data = data[test_idx]
        self.test_label = labels[test_idx]
        self.target_data = np.load("data/weibo/fast/target_test.npy")
        self.target_label = np.zeros((len(self.target_data), 1))
        print("observed data: ", FLAGS.data_size)
        print("train data", len(self.train_data))
        self.poison(FLAGS.poison)

    def poison(self, poisoning):
        num = 100000
        if (poisoning == 1 and FLAGS.method != 'mod'):
            print("poisoning")
            print(FLAGS.method)
            poison = np.load("data/weibo/fast/partial/candidate_%s_%d_%.2f.npy" % (
                FLAGS.method, FLAGS.modify_user, FLAGS.data_size)).astype(np.int)
            idx = np.random.permutation(len(poison))
            poison = poison[idx][:num]
            print(len(poison))
            self.train_data = np.concatenate([self.train_data, poison], axis=0)
            self.train_label = np.concatenate([self.train_label, np.ones_like(self.train_label[:len(poison)])],
                                              axis=0)
        else:
            print(self.users_features.shape, self.train_data.shape)
            data = np.load("data/weibo/fast/mean_data.npy")
            self.users_features = np.concatenate([self.users_features, data], axis=0)
            add_data = np.array([self.users_features.shape[0] - 2, self.users_features.shape[0] - 1]).reshape((1, 2))
            self.users_actions[
                "%d-%d" % (self.users_features.shape[0] - 2, self.users_features.shape[0] - 1)] = np.zeros(3)
            self.train_data = np.concatenate([self.train_data, add_data], axis=0)
            self.train_label = np.concatenate([self.train_label, np.ones((1, 1))], axis=0)
            print(self.users_features.shape, self.train_data.shape)

            print("no poisoning")

    def cal_candidate(self):
        items = np.array(pd.read_csv("data/weibo/fast/item.csv")['0'])
        vectors = np.load("data/weibo/fast/influence_vector.npy")
        control_user = np.load("data/weibo/fast/control_users.npy")

        influence = vectors[0]

        if (FLAGS.method == 'inf'):
            if (FLAGS.modify_user):
                if (influence[0] < 0):
                    self.users_features[control_user, 0] = 0
                else:
                    self.users_features[control_user, 0] = 1
                idx = np.argmax(influence[1:4])
                sex = np.zeros(3)
                sex[idx] = 1
                self.users_features[control_user, 1:4] = sex
                self.users_features[control_user, 7:107] = self.cutmix(influence[7:107],
                                                                       self.users_features[:, 7:107].copy(),
                                                                       len(control_user), 1)
                np.save("data/weibo/fast/users_features_inf.npy", self.users_features[:, :408])

            print("generate influence candidate")
            user1_influences = np.sum(self.users_features * influence[:self.user_feature_num], axis=1)
            user2_influences = np.sum(self.users_features * influence[self.user_feature_num:], axis=1)
            user1_influences = (user1_influences - np.min(user1_influences)) / (
                    np.max(user1_influences) - np.min(user1_influences))
            user2_influences = (user2_influences - np.min(user2_influences)) / (
                    np.max(user2_influences) - np.min(user2_influences))
        elif (FLAGS.method == 'sim'):
            print("generate similar candidate")
            user_center = np.load("data/weibo/fast/user_center.npy")
            user1_influences = np.sum(self.users_features * user_center[1:2], axis=1) / (
                    np.linalg.norm(self.users_features, axis=1) * np.linalg.norm(user_center[1:2], axis=1) + 1e-10)
            user2_influences = np.sum(self.users_features * user_center[61:62], axis=1) / (
                    np.linalg.norm(self.users_features, axis=1) * np.linalg.norm(user_center[61:62], axis=1) + 1e-10)
        elif (FLAGS.method == 'grad'):
            print("generate gradient candidate")
            gradient = np.load("data/weibo/fast/gradient.npy")[None, :]
            user1_influences = np.sum(self.users_features * gradient[:, :self.user_feature_num], axis=1) / (
                    np.linalg.norm(self.users_features, axis=1) * np.linalg.norm(gradient[:, :self.user_feature_num],
                                                                                 axis=1) + 1e-10)
            user2_influences = np.sum(self.users_features * gradient[:, self.user_feature_num:], axis=1) / (
                    np.linalg.norm(self.users_features, axis=1) * np.linalg.norm(gradient[:, self.user_feature_num:],
                                                                                 axis=1) + 1e-10)
        else:
            print("generate random candidate")
            user1 = np.random.choice(control_user, FLAGS.number)[:, None]
            user2 = np.random.choice(items, FLAGS.number)[:, None]
            candidate = np.concatenate([user1, user2], axis=1)
            np.save("data/weibo/fast/candidate_%s_%d.npy" % (FLAGS.method, FLAGS.modify_user), candidate)
            return
        distribution = np.zeros(400)
        user_label = np.load("data/weibo/fast/cluster_labels.npy")
        for i in range(len(self.train_data[:1000000])):
            if (self.train_label[i, 0] == 1):
                user1 = user_label[self.train_data[i, 0]]
                user2 = user_label[self.train_data[i, 1]]
                distribution[user1 * 20 + user2] += 1
        distribution /= np.sum(distribution)
        np.save("data/weibo/fast/distributions.npy", distribution)

        idxs = np.random.choice(np.arange(400), FLAGS.number, p=distribution)
        control_user_label = user_label[control_user]
        item_label = user_label[items]

        candidates = []
        for i in range(400):
            need = len(np.where(idxs == i)[0])
            if (need > 0):
                user1 = control_user[np.where(control_user_label == i // 20)[0]]
                user2 = items[np.where(item_label == i % 20)[0]]
                user2_num = min(len(user2), 5)
                user2 = user2[np.argsort(-user2_influences[user2])][:user2_num]
                user1 = user1[np.argsort(-user1_influences[user1])][:need]
                if (len(user1) > 0 and len(user2) > 0):
                    candidate_list = []
                    score_list = []
                    for u1 in user1:
                        # if (len(candidate_list) >= need):
                        #     break
                        for u2 in user2:
                            candidate_list.append([u1, u2])
                            score_list.append(user1_influences[u1] + user2_influences[u2])
                            # if (len(candidate_list) >= need):
                            #     break
                    # idx = np.argsort(-np.array(score_list))[:need]
                    idx = np.random.permutation(len(candidate_list))[:need]
                    candidates.append(np.array(candidate_list)[idx])
                    # candidates.append(np.array(candidate_list))
        candidate = np.concatenate(candidates, axis=0)
        print(len(candidate))
        np.save("data/weibo/fast/candidate_%s_%d.npy" % (FLAGS.method, FLAGS.modify_user), candidate)

    def cutmix(self, influence, feature, num, ratio=1):
        feature_list = [feature]
        for i in range(ratio - 1):
            idx = np.random.permutation(len(feature))
            mask = np.random.beta(0.5, 0.5, (feature.shape[0], feature.shape[1])) < 0.5
            feature_list.append(feature * mask + feature[idx] * (1 - mask))
        feature_all = np.concatenate(feature_list, axis=0)
        inf = np.sum(feature_all * influence, axis=1)
        idx = np.argsort(-inf)[:num]
        return feature_all[idx]

    def gradient_attack(self):
        attack_grad = tf.gradients(tf.reduce_sum(self.score), self.input)
        idx = np.random.choice(np.arange(len(self.target_data)), 300000, False)
        optimized_data = self.target_data[idx]
        optimized_data = np.concatenate([optimized_data, self.train_data[-1:]], axis=0)
        feed_dict = {self.input1: optimized_data[:, 0], self.input2: optimized_data[:, 1]}
        for i in range(10):
            gradient = self.sess.run(attack_grad, feed_dict)[0][-1]
            self.users_features[-2] = np.clip(self.users_features[-2] + np.sign(gradient[:self.user_feature_num]), 0, 1)
            self.users_features[-1] = np.clip(self.users_features[-1] + np.sign(gradient[self.user_feature_num:]), 0, 1)
        np.save("data/weibo/fast/gradient.npy", np.concatenate([self.users_features[-2], self.users_features[-1]]))
