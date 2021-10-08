#Import libraries
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class Diamond:
    """Diamon price prediction class"""
    def __init__(self, path):
        """Load DataFrame"""
        self.df = pd.read_csv(path)
        print(self.df.describe())
        
    def column_names(self):
        """the function returns the name of all columns"""
        columns = list(self.df.columns.drop("price"))
        return columns

    def vsuialize_1(self):
        """Visualizing scatterplot on important features related to price"""
        sns.scatterplot(x=self.df["carat"], y = self.df["price"])
        sns.scatterplot(x= self.df["depth"] , y=self.df["price"])
        sns.scatterplot(x= self.df["table"] , y=self.df["price"])
        sns.scatterplot(x=self.df["x"], y = self.df["price"])
        sns.scatterplot(x=self.df["y"], y = self.df["price"])
        sns.scatterplot(x=self.df["z"], y = self.df["price"])
        plt.show()

    def detect_outliers(self):
        """Detect outliers, returns outliers indexes"""
        carat_index = list(self.df[self.df["carat"] >= 3].index)
        table_index = list(self.df[self.df["table"] >90].index)
        x_index = list(self.df[self.df["x"] < 2.5].index)
        y_index = list(self.df[self.df["y"] >30].index)
        z_index = list(self.df[self.df["z"] >30 ].index)
        indexes = carat_index + table_index + x_index + y_index + z_index
        return indexes

    def delete_outliers(self,indexes):
        """Delete outliers by knowing their indexes"""
        for outlier in indexes:
            self.df.drop(outlier, inplace=True)

    def correlations(self):
        """Dsiplay correlation using heatmap or pairplot with all features in the DataFrame"""
        type = input("Choose a type of visualization 'corr', or 'pair': ")
        if type == "corr":
            sns.heatmap(self.df.corr(), annot=True)
        elif type == "pair":
            sns.pairplot(self.df)
        plt.show()

    def grade(self,x):
        """convert cut grade string feature into numerical value"""
        x = str(x)
        if x == "Fair":
            return 1
        elif x == "Good":
            return 2
        elif x == "Very Good":
            return 3
        elif x == "Premium":
            return 4
        elif x == "Ideal":
            return 5

    def color_grade(self,x):
        """convert color grade feature into numerical value"""
        x = str(x)
        if x == "J":
            return 1
        elif x == "I":
            return 1
        elif x == "H":
            return 2
        elif x == "G":
            return 3
        elif x == "F":
            return 4
        elif x == "E":
            return 5
        elif x == "D":
            return 6

    def clarity_quality(self,x):
        """convert clarity quality feature into numerical value"""
        x = str(x)
        if x == "I1":
            return 1
        elif x == "SI2":
            return 2
        elif x == "SI1":
            return 3
        elif x == "VS2":
            return 4
        elif x == "VS1":
            return 5
        elif x == "VVS2":
            return 6
        elif x == "VVS1":
            return 7
        elif x == "IF":
            return 8

    def convert_columns(self):
        """Recall previous functions to convert the three features 1-cut_grade 2-color_grade and 3-clar_quality"""
        self.df["cut_grade"] = self.df["cut"].apply(self.grade)
        self.df["color_grade"] = self.df["color"].apply(self.color_grade)
        self.df["clar_quality"] = self.df["clarity"].apply(self.clarity_quality)

    def delete_columns(self):
        """Delete unncessary columns"""
        self.df.drop(["cut","color","clarity","depth","table"], inplace=True, axis=1)

    def split_data(self):
        """Split DataFrame into X -> features and y -> the target
        the function returns X_train, X_test, y_train, y_test"""
        X = self.df.drop("price", axis = 1)
        y = self.df["price"]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=30)
        return X_train, X_test, y_train, y_test

    def split_display_shapes(self, X_train, X_test, y_train, y_test):
        """Display the shapes of training and testing"""
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    def regression_model(self,X_train, y_train):
        """Evaluate and fit the DecisionTreeRegressor model by x_train, y_train
        the function returns the fitted model variable that can be used for predictions later"""
        model = DecisionTreeRegressor()
        fitted_model = model.fit(X_train, y_train)
        return fitted_model

    def model_accuracy_score(self,model, X_train, X_test, y_train, y_test):
        """How accurate is DecisionTreeRegressor model on training and testing"""
        print("Model train accuracy score :",model.score(X_train, y_train))
        print("Model test accuracy score :",model.score(X_test, y_test))

    def save_model(self, model, name):
        """Save the fitted model to deploy it and use it with it`s weight later"""
        joblib.dump(model,name + ".h5")
        print("Model saved sucessfuully")

    def load_model(self, name):
        """Load the model that has been saved alrady"""
        file = joblib.load(name + ".h5")
        print(f"The model {file} has been loaded sucessfully")
        return file

#if this file is the main file excute the codes
if __name__ == "__main__":
    #1- load DataFrame
    diamo= Diamond("diamonds.csv")
    #Correlation of the Ddataset before modified
    diamo.correlations()
    #2- convert object data (i.e categorical data) into numerical data.
    diamo.convert_columns()
    #3- function detect outliers
    indexes = diamo.detect_outliers()
    #4- delete the outliers
    diamo.delete_outliers(indexes)
    #5- delete columns
    diamo.delete_columns()
    #6- split data into training and testing
    X_train, X_test, y_train, y_test = diamo.split_data()
    #display the shape of training and testing
    diamo.split_display_shapes(X_train, X_test, y_train, y_test)
    print("--"*18)
    #7- excute and train the decision tree model
    decision_tree = diamo.regression_model(X_train, y_train)
    #8- predict the y_testing using x_testing...
    y_prediction = decision_tree.predict(X_test)
    #9- display: How accuate is the model on training and testing
    diamo.model_accuracy_score(decision_tree, X_train, X_test, y_train, y_test)
    print("--"*18)
    #10- model metrics r2_score used for decision tree
    print("R2_score :", r2_score(y_test, y_prediction).round(3))

    #11- Finally: save and load model
    # diamo.save_model(decision_tree, input("Type in the name of the file: "))
    # diamo.load_model(input("Type in the name of the file: "))

    #Try to predict your diamond price according to it`s charactristic and the order of features...
    columns = diamo.column_names()
    predictions = decision_tree.predict([[ float(input(f"Enter a charctirstic of your diamond to predict it`s price that is {_}: ")) for _ in columns ]])
    print(f"The price of your diamon is approximatly {predictions[0]}$")
    #End project