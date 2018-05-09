'''
    Module to import multilabels. It assumes that you have run the corresponding Notebook once.
    Notebook: ./MultiLabelProcessor.ipynb
'''

def get_multilabels_train():
    return np.load(train_multilabel_filename)

def get_multilabels_validation():
    return np.load(validation_multilabel_filename)
