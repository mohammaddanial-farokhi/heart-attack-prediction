# ❤️ Heart Attack Prediction with CatBoost & Tkinter GUI  
### A Machine Learning Model + Interactive Desktop Application

این پروژه یک سیستم پیش‌بینی احتمال حمله قلبی است که با استفاده از **یادگیری ماشین (CatBoostClassifier)** و یک رابط کاربری گرافیکی ساده در **Tkinter** ساخته شده است.  
کاربر می‌تواند اطلاعات مربوط به بیمار (سن، جنسیت، فشار خون، کلسترول، نوار قلب، آنژین ناشی از ورزش، وضعیت تالاسمی و …) را وارد کند و مدل احتمال حمله قلبی را پیش‌بینی می‌کند.

---

## 🏷️ Badges
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13.5-blue?logo=python" />
  <img src="https://img.shields.io/badge/Tkinter-GUI-orange?logo=tkinter" />
  <img src="https://img.shields.io/badge/ML-CatBoost-green?logo=catboost" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

---

## 🧩 Project Sections

### **🔹 Data Preprocessing**  
- بررسی داده‌های گمشده و مقادیر غیرمعتبر  
- حذف رکوردهای دارای مقدار صفر در ستون `thal`  
- آماده‌سازی ویژگی‌های دسته‌ای (Categorical Features) برای مدل  

### **🔹 Modeling**  
- استفاده از الگوریتم **CatBoostClassifier**  
- تقسیم داده‌ها به آموزش و تست (70/30)  
- ذخیره مدل آموزش‌دیده با `joblib`  
- معیارهای ارزیابی: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- امکان استفاده از **GridSearchCV** برای تنظیم هایپرپارامترها  

### **🔹 GUI (Tkinter Application)**  
- فرم ورود اطلاعات بیمار (سن، جنسیت، فشار خون، کلسترول، نوار قلب و …)  
- منوهای کشویی برای انتخاب نوع درد قفسه سینه، وضعیت تالاسمی، شیب ST و …  
- دکمه محاسبه احتمال حمله قلبی و نمایش نتیجه به صورت درصد  
- دکمه پاک کردن فرم برای ورود داده‌های جدید  

---

## 📁 Dataset
لینک دیتاست : https://www.kaggle.com/datasets/arezaei81/heartcsv 

فایل دیتاست باید با نام `heart.csv` در پوشه اصلی پروژه قرار گیرد.  
مدل آموزش‌دیده در فایل `heart_model.pkl` ذخیره می‌شود.

---

## 🧩 Python Version

این پروژه با نسخه Python 3.13.5 توسعه و تست شده است، پیشنهاد می‌شود دقیقاً از همین نسخه استفاده کنید.

---


## 🛠️ Installation


1. Clone project
```bash
git clone https://github.com/your-username/heart-attack-prediction.git
cd heart-attack-prediction
```

2. Install requirements
```bash
pip install -r requirements.txt
```

3. Run application
```bash
python src/main.py
```

