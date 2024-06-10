import os


def filter(ensemble, origin='./metadata.csv', train='./metadata_train.csv', test='./metadata_test.csv'):
    """Split a CSV into training and testing CSVs by ensemble

    Args:
        ensemble (_type_): Ensemble to separate by. 
        origin (str, optional): Original CSV file. Defaults to './metadata.csv'.
        train (str, optional): CSV for training data. Defaults to './metadata_train.csv'.
        test (str, optional): CSV for testing data. Defaults to './metadata_test.csv'.
    """
    with open('./metadata.csv', 'r') as metadata, open('./metadata_train.csv', 'w', newline='') as train, open('./metadata_test.csv', 'w', newline='') as test:
        for line in metadata:
            if (split := line.split(',"'))[4] == f'{ensemble}\"': # Why does this dataset use commas IN their entries and as splits?
                test.write(line) if f'{split[0]}.wav' in os.listdir('./test_data') else train.write(line)
    metadata.close()
    train.close()
    test.close()

if __name__ == '__main__':
    filter('Solo Piano')