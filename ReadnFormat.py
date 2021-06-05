import pandas as pd

def ReadnFormat(TestingDataPercent, SamplingPercentageOfBancrupt, ConvertToNp = 0):
    df = pd.read_csv('data.csv')

    df.pop('id')
    df = df.dropna()

    testing = df[0:int(len(df)*TestingDataPercent)]
    training = df[int(len(df)*TestingDataPercent):]

    df1 = training.loc[training['class'] == 1]
    df2 = training.loc[training['class'] == 0][:int(len(df1)*SamplingPercentageOfBancrupt)] 

    training = pd.concat([df1,df2])

    Testtarget = testing.pop('class')
    TrainTarget = training.pop('class')
    if ConvertToNp == 1:
        testing = testing.to_numpy()
        training = training.to_numpy()

        Testtarget = Testtarget.to_numpy()
        TrainTarget = TrainTarget.to_numpy()
    return training, testing, TrainTarget, Testtarget