'''
    Module to import multilabels. It assumes that you have run the corresponding Notebook once.
    Notebook: ./MultiLabelProcessor.ipynb
'''

def get_multilabels_train(filename):
    return np.load(filename)

def get_multilabels_validation(filename):
    return np.load(filename)
