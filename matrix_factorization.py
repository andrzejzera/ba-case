import os
import random
import sys
import pickle
import numpy as np
import pandas as pd
#from decimal import Decimal
from collections import defaultdict
import math
from datetime import datetime

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class MatrixFactorization(object):

    Regularization = 0.002
    BiasLearnRate = 0.005
    BiasReg = 0.002

    LearnRate = 0.005
    all_beers_mean = 0
    number_of_ratings = 0

    item_bias = None
    user_bias = None
    beta = 0.02

    iterations = 0

    def __init__(self, save_path, max_iterations=30):
        self.save_path = save_path
        self.user_factors = None
        self.item_factors = None
        self.item_counts = None
        self.item_sum = None
        self.u_inx = None
        self.i_inx = None
        self.user_ids = None
        self.beer_ids = None

        self.all_beers_mean = 0.0
        self.number_of_ratings = 0
        self.MAX_ITERATIONS = max_iterations
        random.seed(42)

        ensure_dir(save_path)

    def initialize_factors(self, ratings, k=25):
        self.user_ids = set(ratings['review_profilename'].values)
        self.beer_ids = set(ratings['beer_beerid'].values)
        self.item_counts = ratings[['beer_beerid', 'review_overall']].groupby('beer_beerid').count()
        self.item_counts = self.item_counts.reset_index()

        self.item_sum = ratings[['beer_beerid', 'review_overall']].groupby('beer_beerid').sum()
        self.item_sum = self.item_sum.reset_index()

        self.u_inx = {r: i for i, r in enumerate(self.user_ids)}
        self.i_inx = {r: i for i, r in enumerate(self.beer_ids)}

        self.item_factors = np.full((len(self.i_inx), k), 0.1)
        self.user_factors = np.full((len(self.u_inx), k), 0.1)

        self.all_beers_mean = calculate_all_beers_mean(ratings)
        print("user_factors are {}".format(self.user_factors.shape))
        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)

    def predict(self, user, item):

        pq = np.dot(self.item_factors[item], self.user_factors[user].T)
        b_ui = self.all_beers_mean + self.user_bias[user] + self.item_bias[item]
        prediction = b_ui + pq

        if prediction > 5:
            prediction = 5
        elif prediction < 1:
            prediction = 1
        return prediction

    def build(self, ratings, params):

        if params:
            k = params['k']
            self.save_path = params['save_path']

        self.train(ratings, k)

    def split_data(self, min_rank, ratings):

        users = self.user_ids

        train_data_len = int((len(users) * 50 / 100))
        test_users = set(random.sample(users, (len(users) - train_data_len)))
        train_users = users - test_users

        train = ratings[ratings['review_profilename'].isin(train_users)]
        test_temp = ratings[ratings['review_profilename'].isin(test_users)].sort_values('review_time', ascending=False)
        test = test_temp.groupby('review_profilename').head(min_rank)
        additional_training_data = test_temp[~test_temp.index.isin(test.index)]

        train = train.append(additional_training_data)

        return test, train

    def calculate_rmse(self, ratings, factor):

        def difference(row):
            user = self.u_inx[row[0]]
            item = self.i_inx[row[1]]

            pq = np.dot(self.item_factors[item][:factor + 1], self.user_factors[user][:factor + 1].T)
            b_ui = self.all_beers_mean + self.user_bias[user] + self.item_bias[item]
            prediction = b_ui + pq
            MSE = (prediction - row[2]) ** 2
            return MSE

        squared = np.apply_along_axis(difference, 1, ratings).sum()
        return math.sqrt(squared / ratings.shape[0])

    def train(self, ratings_df, k=40):

        self.initialize_factors(ratings_df, k)
        print("training matrix factorization at {}".format(datetime.now()))
        
        valid_data, train_data = self.split_data(30, ratings_df) # new
        print(len(valid_data))
        print(len(train_data))
        
        columns = ['review_profilename', 'beer_beerid', 'review_overall']
        ratings = train_data[columns].to_numpy()
        valid = valid_data[columns].to_numpy()

        index_randomized = random.sample(range(0, len(ratings)), (len(ratings) - 1))

        for factor in range(k):
            factor_time = datetime.now()
            iterations = 0
            last_err = sys.maxsize
            last_valid_mse = sys.maxsize
            
            iteration_err = sys.maxsize
            finished = False

            while not finished:
                start_time = datetime.now()
                iteration_err = self.stocastic_gradient_descent(factor,
                                                              index_randomized,
                                                              ratings)

                valid_mse = self.calculate_rmse(valid, factor)  # new
                
                iterations += 1
                print("epoch in {}, f={}, i={} err={} valid_err={}".format(datetime.now() - start_time,
                                                                       factor,
                                                                       iterations,
                                                                       iteration_err,
                                                                       valid_mse))  # new
                finished = self.finished(iterations,
                                         last_err,
                                         iteration_err,
                                         last_valid_mse,  # new
                                         valid_mse)  # new
                last_err = iteration_err
                last_valid_mse = valid_mse  # new
            self.save(factor, finished)
            print("finished factor {} on f={} i={} err={} valid_err={}".format(factor,
                                                                  datetime.now() - factor_time,
                                                                  iterations,
                                                                  iteration_err,
                                                                  valid_mse))  # new

    def stocastic_gradient_descent(self, factor, index_randomized, ratings):

        lr = self.LearnRate
        b_lr = self.BiasLearnRate
        r = self.Regularization
        bias_r = self.BiasReg

        for inx in index_randomized:
            rating_row = ratings[inx]

            u = self.u_inx[rating_row[0]]
            i = self.i_inx[rating_row[1]]
            rating = rating_row[2]

            err = (rating - self.predict(u, i))

            self.user_bias[u] += b_lr * (err - bias_r * self.user_bias[u])
            self.item_bias[i] += b_lr * (err - bias_r * self.item_bias[i])

            user_fac = self.user_factors[u][factor]
            item_fac = self.item_factors[i][factor]

            self.user_factors[u][factor] += lr * (err * item_fac
                                                  - r * user_fac)
            self.item_factors[i][factor] += lr * (err * user_fac
                                                  - r * item_fac)
        return self.calculate_rmse(ratings, factor)

    def finished(self, iterations, last_err, current_err,
                 last_valid_mse=0.0, valid_mse=0.0):

        if current_err > last_err or iterations >= self.MAX_ITERATIONS or last_valid_mse - valid_mse < 0.0001:
            print('Finish w iterations: {}, last_err: {}, current_err {}, lst_valid_mse {}, valid_mse {}'
                             .format(iterations, last_err, current_err, last_valid_mse, valid_mse))
            return True
        else:
            self.iterations += 1
            return False

    def save(self, factor, finished):

        save_path = self.save_path + '/model/'
        if not finished:
            save_path += str(factor) + '/'

        ensure_dir(save_path)

        print("saving factors in {}".format(save_path))
        user_bias = {uid: self.user_bias[self.u_inx[uid]] for uid in self.u_inx.keys()}
        item_bias = {iid: self.item_bias[self.i_inx[iid]] for iid in self.i_inx.keys()}

        uf = pd.DataFrame(self.user_factors,
                          index=self.user_ids)
        it_f = pd.DataFrame(self.item_factors,
                            index=self.beer_ids)

        with open(save_path + 'user_factors.json', 'w') as outfile:
            outfile.write(uf.to_json())
        with open(save_path + 'item_factors.json', 'w') as outfile:
            outfile.write(it_f.to_json())
        with open(save_path + 'user_bias.data', 'wb') as ub_file:
            pickle.dump(user_bias, ub_file)
        with open(save_path + 'item_bias.data', 'wb') as ub_file:
            pickle.dump(item_bias, ub_file)

    def recommend_items(self, user_id, ratings_df, num=3):

        active_user_items = ratings_df[ratings_df['review_profilename'] == user_id]  # new
        active_user_items = active_user_items[['beer_beerid', 'review_overall']]  # new

        return self.recommend_items_by_ratings(user_id, ratings_df, active_user_items)

    def recommend_items_by_ratings(self, user_id, ratings_df, active_user_items, num=3):
        
        avg = calculate_all_beers_mean(ratings_df)

        rated_beers = {beer['beer_beerid']: beer['review_overall'] \
                       for _, beer in active_user_items.iterrows()}  # new
        recs = {}
        
        # new
        uf = pd.DataFrame(self.user_factors,
                          index=self.user_ids).T
        
        # new
        it_f = pd.DataFrame(self.item_factors,
                            index=self.beer_ids).T
        
        # new
        user_bias_all = {uid: self.user_bias[self.u_inx[uid]] for uid in self.u_inx.keys()}
        item_bias_all = {iid: self.item_bias[self.i_inx[iid]] for iid in self.i_inx.keys()}
        
        if str(user_id) in uf.columns:

            user = uf[str(user_id)]

            scores = it_f.T.dot(user)

            sorted_scores = scores.sort_values(ascending=False)
            result = sorted_scores[:num + len(rated_beers)].astype(float)
            user_bias = 0

            if user_id in user_bias_all.keys():
                user_bias = user_bias_all[user_id]
            elif int(user_id) in user_bias_all.keys():
                user_bias = user_bias_all[int(user_id)]
                print(f'it was an int {user_bias}')

            rating = float(user_bias + avg)
            result += rating

            recs = {r[0]: {'prediction': r[1] + float(item_bias_all[r[0]])}
                    for r in zip(result.index, result) if r[0] not in rated_beers}

        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]

        return sorted_items


def load_all_ratings(ratings, min_ratings=1):
    columns = ['review_profilename', 'beer_beerid', 'review_overall', 'beer_style', 'review_time']
    ratings = ratings[columns]
    
    #print(len(ratings))

    user_count = ratings[['review_profilename', 'beer_beerid']].groupby('review_profilename').count()
    user_count = user_count.reset_index()
    user_ids = user_count[user_count['beer_beerid'] >= min_ratings]['review_profilename']
    ratings = ratings[ratings['review_profilename'].isin(user_ids)]

    ratings['review_overall'] = ratings['review_overall'].astype(float)
    return ratings


def calculate_all_beers_mean(ratings):
    avg = ratings['review_overall'].sum() / ratings.shape[0]
    return avg

