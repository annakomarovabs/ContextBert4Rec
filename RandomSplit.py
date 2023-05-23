class RandomSplit():
    def __init__(self, test_fraction = 0.3):
        self.test_fraction = test_fraction

    def __call__(self, data):
        data = data.sort_values(['user_id','timestamp'])
        num_per_user = data.groupby('user_id').count().to_dict()['item_id']
        train_per_user = {key: int(value * (1-self.test_fraction)) for key, value in num_per_user.items()}
        valid_per_user = {key: int(value * (1-self.test_fraction)*self.test_fraction) for key, value in num_per_user.items()}
        test_per_user = {key: int(value * self.test_fraction) for key, value in num_per_user.items()}
        
        test = data.groupby('user_id', group_keys=False).apply(lambda g: g.tail(test_per_user[g.name]))
        train = data.groupby('user_id', group_keys=False).apply(lambda g: g.head(train_per_user[g.name]))
        valid = train.groupby('user_id', group_keys=False).apply(lambda g: g.tail(valid_per_user[g.name]))  
        train = train.loc[train.index.difference(valid.index)]                                                         
        
        return train, valid, test