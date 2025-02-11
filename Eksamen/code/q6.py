import numpy as np
from sklearn.ensemble import RandomForestClassifier

def loaddata(filename):
    """Load the balloon data set from filename and return t, X
        t - N-dim. vector of target (temperature) values
        X - N-dim. vector containing the inputs (lift) x for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=',')
    # Split into data matrix and target vector
    X = Xt[:,1:]
    t = Xt[:,0]
    
    return t, X

# Load data
t, X = loaddata('../data/accent-mfcc-data_shuffled_train.txt')
tVal, XVal = loaddata('../data/accent-mfcc-data_shuffled_validation.txt')

def __randomForests(X, t):
    predictor = RandomForestClassifier()
    predictor.fit(X, t)
    return predictor

predictor = __randomForests(X, t)
accuracy = predictor.score(X, t)
print("accuracy :", accuracy * 100, "%")

def randomForests(X, t, XVal, tVal, criterion, depth, samples):
    predictor = RandomForestClassifier(criterion = criterion, max_depth = depth, max_features = samples)
    predictor.fit(X, t)
    accuracy = predictor.score(XVal, tVal)
    pred_proba = predictor.predict_proba(XVal)
    return accuracy, pred_proba

def optimalValues():
    optimalParameters = []
    optimalGuess = [0, 0]
    for criterion in ["entropy", "gini"]:
        for depth in [2, 5, 7, 10, 15]:
            for feature in ["sqrt", "log2"]:
                accuracy, pred_proba_init = randomForests(X, t, XVal, tVal, criterion, depth, feature)
                accuracy = accuracy * XVal.shape[0]
                pred_proba = []
                for i in range(pred_proba_init.shape[0]):
                    temp = pred_proba_init[i]
                    pred_proba.append(temp[int(tVal[i] + 0.0005)])
                pred_proba = sum(pred_proba) / len(pred_proba)
                if (accuracy > optimalGuess[0] or (accuracy == optimalGuess[0] and pred_proba > optimalGuess[1])):
                    optimalParameters = [criterion, depth, feature]
                    optimalGuess = [accuracy, pred_proba]
                    print("Improvement. New values:\n")
                    print("Criterion = ", criterion)
                    print("max_depth = ", depth)
                    print("feature = ", feature)
                    print("average probability = ", pred_proba * 100, "%")
                    print("num of correct guesses = ", int(accuracy + 0.0005))                        
                else:
                    print("De nye værdier er ikke bedre, beholder de gamle\n")
    return optimalParameters, optimalGuess

optimalParameters, optimalGuess = optimalValues()
print("De optimale parametre er: ", optimalParameters)
print("Korrekt gættet: ", int(optimalGuess[0] + 0.0005))
print("Gennemsnitlig sandsynlighed: ", optimalGuess[1] * 100, "%")
