import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# ✅ Load your dataset
columns = [f'x{i}' if i % 2 == 0 else f'y{i//2}' for i in range(42)] + ['label']
df = pd.read_csv('full_landmark_dataset.csv', header=None, names=columns)

# ✅ Features and Labels
X = df.drop('label', axis=1).values
y = df['label'].values

# ✅ Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ✅ Save the label encoder for prediction
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# ✅ Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ✅ Build the MLP model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train on GPU with good batch size
model.fit(X_train, y_train,
          epochs=35,
          batch_size=32,  # 🚀 Good for GPU
          validation_data=(X_val, y_val))

# ✅ Save the trained model
model.save('hand_gesture_model.h5')
print("✅ Model trained and saved as hand_gesture_model.h5")
