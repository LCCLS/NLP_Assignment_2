import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support


class ClassDistributions:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, header=0)
        self.ex1_table = pd.DataFrame(columns=["Class label",
                                               "Number of instances",
                                               "Relative label frequency (%)",
                                               "Example tweet with this label"])
        self.update_table()
        self.print_table()

    def update_table(self):
        for label in range(2):
            random.seed(0)

            occurr = self.df['labels'].value_counts()[label]
            freq = self.df['labels'].value_counts(normalize=True)
            label_df = self.df[self.df['labels'].values == label]
            example = label_df.at[random.randint(0, len(label_df)), 'text']

            self.ex1_table.loc[len(self.ex1_table.index)] = [label, occurr, freq[1], example]

    def print_table(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("""
        1.  Class distributions (1 point)
        """)
        print(self.ex1_table.to_string())
        print("\n")

class Baselines:

    def __init__(self, train_path=None, test_path=None):

        self.test_set = pd.read_csv(test_path, header=0)
        self.train_set = pd.read_csv(train_path, header=0)
        self.baselines = {'gold': self.test_set["labels"].to_list()}

        self.majority_baseline()
        self.random_baseline()
        self.ex2_table = self.baseline_table(["random", "majority"])
        self.print_table(table=self.ex2_table)

    def get_majority(self):
        major = (0, 0)

        for label in self.train_set['labels'].value_counts().index.to_list():
            if self.train_set['labels'].value_counts()[label] > major[1]:
                major = label, self.train_set['labels'].value_counts()[label]

        return major[0]

    def majority_baseline(self):

        majority_label = self.get_majority()
        majority_prediction = [majority_label for i in range(len(self.test_set['labels']))]

        self.baselines['majority'] = majority_prediction

    def random_baseline(self):
        random.seed(0)

        random_prediction = [random.choice(self.test_set['labels'].value_counts().index.to_list()) for i in
                             range(len(self.test_set['labels']))]
        self.baselines['random'] = random_prediction

    def baseline_table(self, baselines):

        multi_index = ["Macro_avg", "Weighted_avg"]
        cols = pd.MultiIndex.from_tuples([("Random", "Precision"),
                                          ("Random", "Recall"),
                                          ("Random", "F1"),
                                          ("Majority", "Precision"),
                                          ("Majority", "Recall"),
                                          ("Majority", "F1")])

        data = []
        for baseline in baselines:
            macro_avg = precision_recall_fscore_support(self.baselines['gold'], self.baselines[baseline],
                                                        average='macro', zero_division=0)
            weighted_avg = precision_recall_fscore_support(self.baselines['gold'], self.baselines[baseline],
                                                           average='weighted', zero_division=0)

            macro_list = [macro_avg[0], macro_avg[1], macro_avg[2]]
            weighted_list = [weighted_avg[0], weighted_avg[1], weighted_avg[2]]
            data.append(macro_list)
            data.append(weighted_list)

        baseline_data = [data[0] + data[2], data[1] + data[3]]
        df_table = pd.DataFrame(data=baseline_data, columns=cols, index=multi_index)
        return df_table

    def print_table(self, table):
        print("-------------------------------------------------------------------------------------------------------")
        print("""
        2.  Baselines (1 point)
               """)
        print(table.to_string())
        print("\n")
        print("-------------------------------------------------------------------------------------------------------")
        print("NOT SURE ABOUT THE VALUES FOR NOW. DOUBLE CHECK AGAIN")
        print("-------------------------------------------------------------------------------------------------------")



exercise1 = ClassDistributions('data/olid-train.csv')
exercise2 = Baselines(train_path='data/olid-train.csv', test_path='data/olid-test.csv')
