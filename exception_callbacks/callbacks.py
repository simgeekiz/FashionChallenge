# these have to be defined in a notebook

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.mean_f1s = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        y_true = self.validation_data[1]

        mean_f1 = f1_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        self.mean_f1s.append(mean_f1)
        self.recalls.append(recall)
        self.precisions.append(precision)

        print('mean_F1: {} — precision: {} — recall: {}'.format(mean_f1, precision, recall))

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.grid()
        plt.legend()
        plt.show()

def all_call_backs():
    callbacks_list = []

    a = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.15,
        patience=3,
        min_lr=0.0001
    )

    b = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=8,
        verbose=0,
        mode='auto'
    )

    c = keras.callbacks.ModelCheckpoint(
        filepath=network_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1
    )

    callbacks_list = [a, b, c]
    callbacks_list = callbacks_list + [PlotLosses()]
    callbacks_list = callbacks_list + [Metrics()]
