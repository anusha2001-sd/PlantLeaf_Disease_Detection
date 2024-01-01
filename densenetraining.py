from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
def process(path):
        imagePaths = list(paths.list_images(path))
        #print("image path=",imagePaths)
        data = []
        labels = []
        # loop over the image paths
        for imagePath in imagePaths:
                # extract the class label from the filename
                label = imagePath.split(os.path.sep)[-2]
                #print("Label for images",label)
                # load the input image (224x224) and preprocess it
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                # update the data and labels lists, respectively
                data.append(image)
                labels.append(label)
                #print("Labels:",labels)
        # convert the data and labels to NumPy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels)
        #print("Data===",data)
        #print("Labels==",labels)
        #The next step is to load the pre-trained model and customize it according to our problem. So we just remove the top layers of this pre-trained model and add few layers of our own. As you can see the last layer has two nodes as we have only two outputs. This is called transfer learning.

        baseModel = DenseNet121(weights="imagenet", include_top=False,
                input_shape=(224, 224, 3))
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(3, activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)
        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
                layer.trainable = False
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        #labels = to_categorical(labels)
        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        #nsamples, nx, ny = data.shape
        #d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))
        (trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.33, stratify=labels, random_state=42)
        # construct the training image generator for data augmentation
        aug = ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest")
        INIT_LR = 1e-4
        EPOCHS = 20
        BS = 32
        print("[INFO] compiling model...")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
        opt =  tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                metrics=["accuracy"])
        # train the head of the network
        print("[INFO] training head...")
        H = model.fit(
                aug.flow(trainX, trainY, batch_size=BS),
                steps_per_epoch=len(trainX) // BS,
                validation_data=(testX, testY),
                validation_steps=len(testX) // BS,
                epochs=EPOCHS)
        N = EPOCHS
        print(H.history.keys())
        plt.plot(H.history['accuracy'])
        plt.plot(H.history['val_accuracy'])
        #plt.title('model accuracy')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig('results/Densenet_Training and validation accuracy.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        # summarize history for loss
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        #plt.title('model loss')
        plt.title('Training and validation Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig('results/Densenet_Training and validation Loss.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        model.save('Leafdisease_densenet.h5')
        predIdxs = model.predict(testX, batch_size=BS)
        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)
        # show a nicely formatted classification report
        print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))
        # compute the confusion matrix and and use it to derive the raw
        # accuracy, sensitivity, and specificity
        cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
        total = sum(sum(cm))
        #acc = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        # show the confusion matrix, accuracy, sensitivity, and specificity
        print(cm)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["DOWNY_MILDEW","HEALTHY", "LEAFY_MINOR"])
        cm_display.plot()
        plt.savefig("results/Densenet_CM.png")
        plt.show()
        fpr = {}
        tpr = {}
        thresh ={}
        n_class = 3
        for i in range(n_class):    
                fpr[i], tpr[i], thresh[i] = roc_curve(testY.argmax(axis=1), predIdxs, pos_label=i)
        # plotting    
        plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='DOWNY_MILDEW vs Rest')
        plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='HEALTHY vs Rest')
        plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='LEAFY_MINOR vs Rest')
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('results/Densenet-Multiclass ROC',dpi=300);
        plt.show() 
        #print("acc: {:.4f}".format(acc))
        print("sensitivity: {:.4f}".format(sensitivity))
        print("specificity: {:.4f}".format(specificity))
        #return (acc*100)
#process("./Data")

