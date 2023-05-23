from typing import List
import pandas as pd
import numpy as np

class TopPopular:

    def __init__(self):

        self.trained = False
    
    def fit(self, df, col='train_interactions'):
        
        counts = {}
        for _, row in df.iterrows():
            for item, _ in row[col]:
                if item in counts:
                    counts[item] += 1
                else:
                    counts[item] = 1
                    
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        self.recommenations = [x[0] for x in counts]
        self.trained = True
        
    def predict(self, df, topn=10)  -> List[np.ndarray]:
        
        assert self.trained
        return [self.recommenations[:topn]]*len(df)
    

    
class ModifiedTopPopular(TopPopular):

    def __init__(self):

        self.trained = False

        
    def predict(self, df, topn=10)  -> List[np.ndarray]:
        
        assert self.trained
        
        all_recs = []
        
        
        for idx, row in df.iterrows():
            
            user_recs = []
            
            user_interactions = [x[0] for x in row['train_interactions']]
            user_interactions += [x[0] for x in row['valid_interactions']]
            
            user_interactions = set(user_interactions)
            
            for elem in self.recommenations:
                if elem not in user_interactions:
                    user_recs.append(elem)
                    
                if len(user_recs) == topn:
                    break
                    
            all_recs.append(user_recs)
                
        return all_recs
     