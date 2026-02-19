from keras.src.legacy.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten #build the CNN
from keras.layers import Conv2D,MaxPooling2D  #build the CNN
import os

train_data_dir='Facial Emotion/data/train'
validation_data_dir='Facial Emotion/data/test/'

#improve generalization and avoid overfitting
train_datagen = ImageDataGenerator(
					rescale=1./255, 
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(48, 48),
					batch_size=32,
					class_mode='categorical', #labels
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(48, 48),
							batch_size=32,
							class_mode='categorical',
							shuffle=True)


class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__()


model = Sequential() #create a neural network layer by layer

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) #extracting spatial features
model.add(MaxPooling2D(pool_size=(2, 2))) #reduces spatial dimensions
model.add(Dropout(0.1)) #Dropout layers to prevent overfitting

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu')) #allows model to learn complex patterns
model.add(Dropout(0.2))
#output layer
model.add(Dense(7, activation='softmax')) #multi-class classification

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy']) #Adam optimizer for adaptive learning
print(model.summary())


train_path = "Facial Emotion/data/train"
test_path = "Facial Emotion/data/test/"
#Counts the total number of images
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(num_train_imgs)
print(num_test_imgs)
epochs=100 #The model will train for 100 epochs 

#stored in history for later analysis
history=model.fit(train_generator,
                steps_per_epoch=num_train_imgs//32,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_test_imgs//32)

model.save('model_file.h5') #trained model is saved in HDF5 format so it can be loaded later for predictions
