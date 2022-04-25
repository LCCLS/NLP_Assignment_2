import pandas as pd
import random


class ClassDistributions:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, header=0)
        self.table = pd.DataFrame(columns=["Class label",
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

            self.table.loc[len(self.table.index)] = [label, occurr, freq[1], example]

    def print_table(self):
        print("-------------------------------------------------------------------------------------------------------")
        print("""
        1.  Class distributions (1 point)
        """)
        print(self.table.to_string())
        print("\n")
        print("-------------------------------------------------------------------------------------------------------")


class Baselines:

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, header=0)


exercise1 = ClassDistributions('data/olid-train.csv')
exercise2 = Baselines('data/olid-train.csv')
