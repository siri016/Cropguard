from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("🔍 Starting Model Evaluation...\n")

# ✅ Load trained model
model = load_model("model/crop_model.h5")

# ✅ Load validation data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ✅ Predictions
pred = model.predict(val_data)
y_pred = np.argmax(pred, axis=1)
y_true = val_data.classes

# ✅ Overall Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Overall Validation Accuracy: {accuracy:.4f}\n")

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("📊 Confusion Matrix:\n", cm, "\n")

# ✅ Classification Report
print("📄 Classification Report:\n")
print(classification_report(y_true, y_pred))

# ✅ Class-wise Accuracy (IMPORTANT 🔥)
class_names = list(val_data.class_indices.keys())

print("\n📌 Class-wise Accuracy:")
for i, class_name in enumerate(class_names):
    correct = cm[i][i]
    total = np.sum(cm[i])
    acc = correct / total
    print(f"{class_name}: {acc:.2f}")

# ✅ Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ✅ Confidence Analysis
confidence = np.max(pred, axis=1)

plt.figure(figsize=(6,4))
plt.hist(confidence, bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Number of Samples")
plt.show()

# ✅ Misclassification Analysis (VERY IMPORTANT 🔥)
print("\n❗ Misclassification Summary:")

for i, class_name in enumerate(class_names):
    total = np.sum(cm[i])
    correct = cm[i][i]
    wrong = total - correct
    print(f"{class_name}: {wrong} misclassified out of {total}")

print("\n🔍 Evaluation Completed Successfully!")