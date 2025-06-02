import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# بارگذاری داده‌ها (مثلاً از فایل CSV)
#df = pd.read_csv('cicids_small.csv')  # جایگزین با مسیر فایل واقعی
df = pd.read_csv("cicids_simplified.csv", encoding='latin1')
# حذف ویژگی‌های نامربوط
df.drop(columns=['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], errors='ignore', inplace=True)

# حذف مقادیر گمشده
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# برچسب‌گذاری کلاس‌ها
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# جدا کردن ویژگی‌ها و هدف
X = df.drop('Label', axis=1)
y = df['Label']

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم‌بندی آموزش/آزمون
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# آموزش مدل Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# پیش‌بینی
y_pred = nb_model.predict(X_test)

# ارزیابی مدل
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
