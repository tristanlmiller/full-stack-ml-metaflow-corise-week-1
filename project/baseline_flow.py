from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

labeling_function = lambda row: (row['rating']>4) + 0


class BaselineNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        _has_review_df['label'] = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        df = _has_review_df
        self.df = _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        #add a random_state here for reproducibility
        self.traindf, self.valdf = train_test_split(df, test_size=self.split_size,random_state=2023)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import accuracy_score, roc_auc_score

        self.traindf['baseline'] = 1
        self.valdf['baseline'] = 0
        self.base_acc = accuracy_score(self.valdf["label"], self.valdf["baseline"])
        self.base_rocauc = roc_auc_score(self.valdf["label"], self.valdf["baseline"])

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        import numpy as np
        msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        false_pos = (self.valdf['label']==0) & (self.valdf['baseline']>0.5)
        if np.sum(false_pos) > 0:
            fp_reviews = self.valdf.loc[false_pos,'review_text'].sample(2)
            current.card.append(Markdown('\n\n'.join(fp_reviews)))
        else:
            current.card.append(Markdown('none'))

        # Documentation: https://docs.metaflow.org/api/cards#table

        current.card.append(Markdown("## Examples of False Negatives"))
        false_neg = (self.valdf['label']==1) & (self.valdf['baseline']<=0.5)
        if np.sum(false_neg) > 0:
            fn_reviews = self.valdf.loc[false_neg,'review_text'].sample(2)
            current.card.append(Markdown('\n\n'.join(fn_reviews)))
        else:
            current.card.append(Markdown('none'))


if __name__ == "__main__":
    BaselineNLPFlow()
