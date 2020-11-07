import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from pandas import DataFrame

from src.models.loss.loss import bce_dice_loss, dice_coefficient, dice_metric
from src.utils.utility_functions import *
from src.models.network.Unet2D import unet2d_8_128
from src.models.network.Unet2DResidual import unet2d_res_32_512
from src.models.network.Unet2DResidualRnn import unet2D_res_srnn_32_512
from src.model_dev.Process_Raw_Input import *
from src.model_dev.Callback import PlotLearning

tf.compat.v1.disable_eager_execution()
np.random.seed(144)
tf.random.set_seed(144)
K.set_image_data_format('channels_last')


class TPEFingernail:
    """
    Class for Training, Predicting and Evaluating and hence TPE class.
    """
    train_hist_df: DataFrame
    RUN_ID: str

    def __init__(self, model_class, x_train, y_train, x_test, y_test, **kwargs):
        """
        Constants for each training class object.
        :param model_class: the model class
        :param x_train: train image dataset
        :param y_train: train mask dataset
        :param x_test: test image dataset
        :param y_test: test mask dataset
        :param kwargs: etc arguments
        """
        self.STUDY_NAME = kwargs.get('run_name', 'Fingernail_dataset1')
        self.TO_SAVE = True
        self.VERBOSE = 0

        self.IMAGE_DATA_TYPE = np.int16
        self.IMG_ROWS = 160
        self.IMG_COLS = 192
        self.IMG_CHNS = 3

        self.TEST_RATIO = 0.08
        self.BATCH_SIZE = 2
        self.EPOCH = 220

        self.IMAGE_RAW_DATA_FOLDER = 'data/label/image/'
        self.LABEL_RAW_DATA_FOLDER = 'data/label/label/'
        self.SAVED_MODEL_FOLDER = 'model/weight/'
        self.SAVED_LOGS_FOLDER = 'model/log/'
        self.SAVED_RESULTS_FOLDER = 'model/prediction/'
        check_exist_folder(self.SAVED_MODEL_FOLDER, create_if_not_exist=True)
        check_exist_folder(self.SAVED_LOGS_FOLDER, create_if_not_exist=True)
        check_exist_folder(self.SAVED_RESULTS_FOLDER, create_if_not_exist=True)

        self.x_train = x_train[0:10]
        self.y_train = y_train[0:10]
        self.x_test = x_test[0:10]
        self.y_test = y_test[0:10]
        self.y_pred = None
        self.model_class = model_class
        self.model = model_class(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS)

        self.optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199, name='AdamLR-5')
        self.loss = bce_dice_loss
        self.loss_name = 'DICE'
        self.metrics = [dice_metric]
        self.make_run_id()

    def make_run_id(self):
        self.RUN_ID = '%s_%s_%s_EP%d_BS%d_%dx%d' % (self.STUDY_NAME, self.model.name, self.loss_name, self.EPOCH,
                                                    self.BATCH_SIZE, self.IMG_ROWS, self.IMG_COLS)

    def set_img_size(self, img_rows, img_cols):
        self.IMG_ROWS = img_rows
        self.IMG_COLS = img_cols
        self.model = self.model_class(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS)
        self.make_run_id()
        return self

    def change_model(self, model_class):
        self.model = model_class(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS)
        self.make_run_id()
        return self

    def set_run_name(self, run_name):
        self.STUDY_NAME = run_name
        self.make_run_id()
        return self

    def set_num_epochs(self, new_epochs):
        self.EPOCH = new_epochs
        self.make_run_id()
        return self

    def set_batch_size(self, new_batch_size):
        self.BATCH_SIZE = new_batch_size
        self.make_run_id()
        return self

    def set_optimizer(self, optimizer_inp):
        self.optimizer = optimizer_inp
        return self

    def set_loss(self, loss, loss_name):
        self.loss = loss
        self.loss_name = loss_name
        self.make_run_id()
        return self

    def set_metrics(self, metrics_inp):
        self.metrics = metrics_inp
        return self

    def train(self, **kwargs):
        """
        Train the model and save if to_save is True.
        :return: train results dictionary
        """
        run_id = kwargs.get('run_id', self.RUN_ID)
        to_save = kwargs.get('to_save', self.TO_SAVE)
        x_test = kwargs.get('x_test', self.x_test)
        y_test = kwargs.get('y_test', self.y_test)
        x_train = kwargs.get('x_train', self.x_train)
        y_train = kwargs.get('y_train', self.y_train)

        print('_' * 70 + '\nTrain process started on train/test %s/%s.' % (str(x_train.shape), str(y_train.shape)))

        print('_' * 70 + '\nCreating and compiling model...')
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        callbacks = self.get_callbacks()

        print('_' * 70 + '\nFitting model...')
        if x_test is not None:
            train_history = self.model.fit(self.x_train, self.y_train, batch_size=self.BATCH_SIZE,
                                           epochs=self.EPOCH, verbose=self.VERBOSE, shuffle=False,
                                           validation_data=(x_test, y_test), callbacks=callbacks)
        else:
            train_history = self.model.fit(self.x_train, self.y_train, batch_size=self.BATCH_SIZE,
                                           epochs=self.EPOCH, verbose=self.VERBOSE, shuffle=False,
                                           validation_split=self.TEST_RATIO, callbacks=callbacks)
        self.train_hist_df = pd.DataFrame(train_history.history, index=train_history.epoch)
        print('_' * 70 + '\nTraining finished')

        if to_save:
            if not os.path.exists(self.SAVED_MODEL_FOLDER):
                os.mkdir(self.SAVED_MODEL_FOLDER)
            model_weights_addr = os.path.join(self.SAVED_MODEL_FOLDER, run_id + '_best_weights.h5')
            # self.model.save_weights(model_weights_addr)
            print('Best weights saved at %s.' % model_weights_addr)

            train_history_addr = os.path.join(self.SAVED_LOGS_FOLDER, run_id + 'train_history.csv')
            self.train_hist_df.to_csv(path_or_buf=train_history_addr)
            print('Train history saved at %s.' % train_history_addr)
        self.train_res_dic = {'model': self.model, 'train_history': train_history}
        return self.train_res_dic

    def predict(self, x_test=None, y_test=None, **kwargs):
        """
        Predict the model and save predication results if to_save is True.
        :param x_test: test image dataset
        :param y_test: test mask dataset
        :param kwargs: etc arguments
        :return: prediction results dictionary
        """
        to_save = kwargs.get('to_save', self.TO_SAVE)
        run_id = kwargs.get('run_id', self.RUN_ID)
        if x_test is None or y_test is None:
            x_test = self.x_test
            y_test = self.y_test
        self.y_pred = self.model.predict(self.x_test, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        print('_' * 70 + '\nPrediction on test data shape %s output shape %s.' % (self.x_test.shape, self.y_pred.shape))

        if to_save:
            if not os.path.exists(self.SAVED_RESULTS_FOLDER):
                os.mkdir(self.SAVED_RESULTS_FOLDER)
            np.save(os.path.join(self.SAVED_RESULTS_FOLDER, run_id + '_test_feature.npy'), self.x_test)
            np.save(os.path.join(self.SAVED_RESULTS_FOLDER, run_id + '_test_label.npy'), self.y_test)
            np.save(os.path.join(self.SAVED_RESULTS_FOLDER, run_id + '_test_predict.npy'), self.y_pred)
            print('Predicted label saved to ' + os.path.join(self.SAVED_RESULTS_FOLDER,
                                                             run_id + '_test_predict.npy'))
        self.pred_res_dic = {'x_test': x_test, 'y_test': y_test, 'y_pred': self.y_pred}
        return self.pred_res_dic

    def evaluate(self, y_test=None, y_pred=None, is_plot=False):
        """
        Evaluate prediction results.
        :param y_test: test mask dataset
        :param y_pred: prediction mask data predicted on test data
        :param is_plot: to plot the results
        :return: evaluation results
        """
        if y_test is None or y_pred is None:
            y_test = self.y_test
            y_pred = self.y_pred
        evaluation_res = dice_coefficient(y_test, y_pred)

        if is_plot:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(y_test[0, :, :, 0])
            axs[0, 0].set_title('Test Label')
            axs[0, 0].get_xaxis().set_visible(False)

            axs[0, 1].imshow(y_pred[0, :, :, 0])
            axs[0, 1].set_title('Test Pred')
            axs[0, 1].get_xaxis().set_visible(False)

            axs[1, 0].imshow(y_test[-1, :, :, 0])
            axs[1, 0].set_title('Test Label')
            axs[1, 0].get_xaxis().set_visible(False)

            axs[1, 1].imshow(y_pred[-1, :, :, 0])
            axs[1, 1].set_title('Test Pred')
            axs[1, 1].get_xaxis().set_visible(False)
            plt.show()
        print('Dice score accuracy on test dataset is %f02' % evaluation_res)
        return evaluation_res

    def get_callbacks(self):
        """
        Generates callbacks for training.
        :return: array of callbacks
        """
        plot_learning = PlotLearning(train_object=self)
        model_checkpoint = ModelCheckpoint(os.path.join(self.SAVED_MODEL_FOLDER, self.RUN_ID + '_best_weights.h5'),
                                           monitor='val_dice_metric', mode='max', save_best_only=True,
                                           save_weights_only=True)
        # stop_train = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.00001)
        if not os.path.exists(self.SAVED_LOGS_FOLDER):
            os.mkdir(self.SAVED_LOGS_FOLDER)
        csv_logger = CSVLogger(os.path.join(self.SAVED_LOGS_FOLDER, self.RUN_ID + 'train_logs.txt'), separator=',',
                               append=False)
        return [model_checkpoint, reduce_lr, csv_logger, plot_learning]

    def tpe_main(self, **kwargs):
        self.train(**kwargs)
        self.predict(**kwargs)
        self.evaluate(is_plot=True)
        # self.plot_train_hist()
        print(self.RUN_ID + '**Completed**')

    def plot_train_hist(self):
        """
        Plot train history at the end of training.
        :return: None
        """
        ax = self.train_hist_df.plot(y=['loss', 'val_loss', 'dice_metric', 'val_dice_metric'], grid=True,
                                     secondary_y=['dice_metric', 'val_dice_metric'], title=self.RUN_ID)
        ax.set_xlabel('Epoch #')
        ax.set_ylabel('Loss Value')
        ax.right_ax.set_ylabel('Dice Score')
        plt.show()


if __name__ == '__main__':
    print('Loading and pre_processing train data...')
    TARGET_IMG_ROWS = 160
    TARGET_IMG_COLS = 192
    data_dic_1 = load_image_data_1('data/raw/nails/', resize_row=TARGET_IMG_ROWS, resize_col=TARGET_IMG_COLS)
    # data_dic_2 = load_image_data_2('data/raw/dataset2/', resize_row=TARGET_IMG_ROWS, resize_col=TARGET_IMG_COLS)
    # for key, val in data_dic_2.items():
    #     data_dic_2[key] = np.concatenate((data_dic_1[key], data_dic_2[key]), axis=0)

    train_test_eval_unet2d_dice = TPEFingernail(**data_dic_1, model_class=unet2d_8_128)
    train_test_eval_unet2d_dice.tpe_main()
    train_test_eval_unet2d_res = TPEFingernail(**data_dic_1, model_class=unet2d_res_32_512)
    train_test_eval_unet2d_res.tpe_main()
    train_test_eval_unet2d_res_srnn = TPEFingernail(**data_dic_1, model_class=unet2D_res_srnn_32_512)
    train_test_eval_unet2d_res_srnn.tpe_main()
