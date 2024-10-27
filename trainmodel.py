from function import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            
            # Check if the frame has the expected shape (63,)
            if res.shape != (63,):
                print(f"Inconsistent shape detected at {action}, sequence {sequence}, frame {frame_num}: {res.shape}")
                continue  # Skip or handle this frame (you can pad or truncate as needed)
            
            window.append(res)  # Append only valid frames

        if len(window) == sequence_length:  # Only append if the sequence is complete
            sequences.append(window)  # Append the sequence (shape: (30, 63))
            labels.append(label_map[action])
        else:
            print(f"Incomplete sequence at {action}, sequence {sequence}, only {len(window)} frames")

# Convert sequences to a NumPy array after ensuring all sequences have the same shape
X = np.array(sequences)  # Shape should be (num_sequences, 30, 63)
y = to_categorical(labels).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
summary = model.summary()

print('--------------------------Model Summary--------------------------\n')
print(summary)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')