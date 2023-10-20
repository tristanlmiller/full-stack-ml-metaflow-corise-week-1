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


class ImprovedNLPFlow(FlowSpec):
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

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        _has_review_df['label'] = _has_review_df.apply(labeling_function, axis=1)
        df = _has_review_df
        self.df = _has_review_df

        self.next(self.preprocess)

    @step
    def preprocess(self):
        from sklearn.feature_extraction.text import CountVectorizer
        import nltk
        import numpy as np
        from sklearn.model_selection import train_test_split

        # drop stop words and punctuation
        nltk.download("stopwords")
        stopwords = list(nltk.corpus.stopwords.words("english"))
        processed_reviews = ['']*self.df.shape[0]
        for i, review in enumerate(self.df['review_text']):
            non_stopwords = []
            for word in review.split():
                word = word.translate(str.maketrans("", "", string.punctuation))
                if word == "":
                    continue
                if not word.lower() in stopwords:
                    non_stopwords.append(word)
            processed_reviews[i] = ' '.join(non_stopwords)

        #vectorize
        vectorizer = CountVectorizer(analyzer = "word",min_df=0.05,max_features=25)
        bag_of_words = vectorizer.fit_transform(df['review_text'].head()).toarray()
        
        #drop ignore list
        ignore_list = ['dress','love','like','wear','great','im','would','really','ordered','perfect','one','flattering','well','nice']
        keep_col = []
        vocab = []
        for word,idx in vectorizer.vocabulary_.items():
            if word not in ignore_list:
                keep_col.append(idx)
                vocab.append(word)
        bag_of_words = bag_of_words[:,keep_col]
        bag_of_words = bag_of_words/np.sum(bag_of_words,axis=0)
        self.vocab = vocab

        # split the data 80/20, or by using the flow's split-sz CLI argument
        #add a random_state here for reproducibility
        self.x_train, self.y_train, self.x_val, self.y_val, self.val = \
            train_test_split(bag_of_words, self.df['label'], test_size=self.split_size,random_state=2023)
        print(f"num of rows in train set: {self.x_train.shape[0]}")
        print(f"num of rows in validation set: {self.x_val.shape[0]}")

        self.next(self.modeling)

    @step
    def modeling(self):
        "Compute the baseline"
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score

        model = LogisticRegression()
        model = fit(self.x_train, self.y_train)
        
        self.pred_val = model.predict(x_val)
        self.prob_val = model.predict_proba(x_val)[:,0]
        self.acc = accuracy_score(self.valdf["label"], self.pred_val)
        self.rocauc = roc_auc_score(self.valdf["label"], self.prob_val)

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        import numpy as np
        import pandas as pd
        msg = "Model Accuracy: {}\Model AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Thematic words and their coefficients"))
        coefs = pd.DataFrame({'Word':self.vocab,'Coefficient':self.model.coef_[0]})
        coefs = coef.sort_values('Coefficient',key=abs,ascending=False)
        current.card.append(Table.from_dataframe(coefs))

        current.card.append(Markdown("## Examples of False Positives"))
        false_pos = (self.y_val==0) & (self.pred_val==1)
        if np.sum(false_pos) > 0:
            fp_reviews = self.valdf.loc[false_pos,'review_text'].sample(2)
            current.card.append(Markdown('\n\n'.join(fp_reviews)))
        else:
            current.card.append(Markdown('none'))

        # Documentation: https://docs.metaflow.org/api/cards#table

        current.card.append(Markdown("## Examples of False Negatives"))
        false_neg = (self.y_val==1) & (self.pred_val==0)
        if np.sum(false_neg) > 0:
            fn_reviews = self.valdf.loc[false_neg,'review_text'].sample(2)
            current.card.append(Markdown('\n\n'.join(fn_reviews)))
        else:
            current.card.append(Markdown('none'))
        


if __name__ == "__main__":
    ImprovedNLPFlow()
