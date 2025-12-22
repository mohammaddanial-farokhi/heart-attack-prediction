import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import KernelPCA
import joblib
from tkinter import *
import tkinter as tk
from tkinter import ttk


data = pd.read_csv("heart.csv")
# print(data.head())

### --- EDA ---###

# data.hist(figsize=(15, 12), bins=20, color="skyblue", edgecolor="black")
# plt.tight_layout()
# plt.show()

# print(data.isnull().sum())
# print((data == 0).sum())

# print(data["cp"].value_counts()) #OK
# print(data["restecg"].value_counts()) #OK
# print(data["exang"].value_counts()) #OK
# print(data["oldpeak"].value_counts()) #OK

# print(data["thal"].value_counts()) # NOT OK


### --- Preprocess ---###
data = data[data["thal"] != 0]

### --- mnodeling ---###

categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


X = data.drop(columns=["target"])
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# pca = KernelPCA(n_components=10, random_state=123)
# X = pca.fit_transform(X)

# model = CatBoostClassifier(
#     verbose=200,
#     random_seed=123,
#     iterations=5000,
#     depth=6,
#     learning_rate=0.01,
#     l2_leaf_reg=3,
#     bootstrap_type="MVS",
#     grow_policy="Depthwise",
# )
# model.fit(X_train, y_train, cat_features=categorical_features)
# joblib.dump(model, "heart_model.pkl")

model = joblib.load("heart_model.pkl")

# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)[:, 1]

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1-score:", f1_score(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# parameters = {
#     "iterations": [2000],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "depth": [6, 8, 10],
#     "l2_leaf_reg": [1, 3, 5, 10],
#     "loss_function": ["Logloss"],
#     "eval_metric": ["F1"],
# }

# grid = GridSearchCV(
#     estimator=model,
#     param_grid=parameters,
#     cv=3,
#     scoring="f1",
#     n_jobs=-1,
#     verbose=2,
# )

# grid.fit(X_train, y_train, cat_features=categorical_features)

# print("Best Parameters:", grid.best_params_)
# print("Best F1 Score:", grid.best_score_)


### --- GUI ---###
def GUI():
    window = tk.Tk()
    window.title("احتمال حمله قلبی")
    window.geometry("800x400")

    window.grid_columnconfigure(0, minsize=400)
    window.grid_columnconfigure(1, minsize=400)

    # row 0
    l0_1 = tk.Label(window, text="سن", font=("Arial", 14))
    l0_1.grid(row=0, column=1, sticky="e", padx=(0, 10), pady=(10, 0))
    e0_1 = tk.Entry(window)
    e0_1.grid(row=0, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    l0_2 = tk.Label(window, text="جنسیت", font=("Arial", 14))
    l0_2.grid(row=0, column=0, sticky="e", padx=(0, 10), pady=(10, 0))
    c0_1 = ttk.Combobox(window, values=["زن", "مرد"], state="readonly", width=17)
    c0_1.current(1)
    c0_1.grid(row=0, column=0, sticky="w", padx=(30, 0), pady=(10, 0))

    # row 1
    l1_1 = tk.Label(window, text="نوع درد قفسه سینه", font=("Arial", 14))
    l1_1.grid(row=1, column=1, sticky="e", padx=(0, 10), pady=(10, 0))
    cp_types = ["1-Typical angina", "2-Atypical angina", "3-Non-anginal pain", "4-Asymptomatic"]
    c1_1 = ttk.Combobox(window, values=cp_types, state="readonly", width=25)
    c1_1.current(0)
    c1_1.grid(row=1, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    l1_2 = tk.Label(window, text="فشار خون در حالت استراحت", font=("Arial", 14))
    l1_2.grid(row=1, column=0, sticky="e", padx=(0, 10), pady=(10, 0))
    e1_1 = tk.Entry(window)
    e1_1.grid(row=1, column=0, sticky="w", padx=(30, 0), pady=(10, 0))

    # row2
    l2_1 = tk.Label(window, text="سطح کلسترول", font=("Arial", 14))
    l2_1.grid(row=2, column=1, sticky="e", padx=(0, 10), pady=(10, 0))
    e2_1 = tk.Entry(window)
    e2_1.grid(row=2, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    l2_2 = tk.Label(window, text="قند خون ناشتا", font=("Arial", 14))
    l2_2.grid(row=2, column=0, sticky="e", padx=(0, 10), pady=(10, 0))
    c2_1 = ttk.Combobox(window, values=["کمتر یا مساوی 120", "بالای 120"], state="readonly", width=17)
    c2_1.current(0)
    c2_1.grid(row=2, column=0, sticky="w", padx=(30, 0), pady=(10, 0))

    # row 3
    l3_1 = tk.Label(window, text="نوار قلب در حالت استراحت", font=("Arial", 14))
    l3_1.grid(row=3, column=1, sticky="e", padx=(0, 10), pady=(10, 0))

    el_types = ["1-Normal", "2-ST-T wave abnormality", "3-Left ventricular hypertrophy"]
    c3_1 = ttk.Combobox(window, values=el_types, state="readonly", width=25)
    c3_1.current(0)
    c3_1.grid(row=3, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    l3_2 = tk.Label(window, text="بیشترین ضربان قلب ثبت شده در ورزش", font=("Arial", 14))
    l3_2.grid(row=3, column=0, sticky="e", padx=(0, 10), pady=(10, 0))
    e3_1 = tk.Entry(window, width=15)
    e3_1.grid(row=3, column=0, sticky="w", padx=(30, 0), pady=(10, 0))

    # row4
    l4_1 = tk.Label(window, text="آنژین ناشی از ورزش", font=("Arial", 14))
    l4_1.grid(row=4, column=1, sticky="e", padx=(0, 10), pady=(10, 0))
    v4 = IntVar()
    r4_1 = tk.Radiobutton(text="بله", variable=v4, value=1)
    r4_1.grid(row=4, column=1, sticky="e", padx=(0, 200), pady=(10, 0))
    r4_2 = tk.Radiobutton(text="خیر", variable=v4, value=2)
    r4_2.grid(row=4, column=1, sticky="e", padx=(0, 250), pady=(10, 0))

    l4_2 = tk.Label(window, text="نسبت به حالت استراحت ST افت", font=("Arial", 14))
    l4_2.grid(row=4, column=0, sticky="e", padx=(0, 10), pady=(10, 0))
    e4_1 = tk.Entry(window, width=15)
    e4_1.grid(row=4, column=0, sticky="w", padx=(30, 0), pady=(10, 0))

    # row 5
    l5_1 = tk.Label(window, text="در تست ورزش ST شیب", font=("Arial", 14))
    l5_1.grid(row=5, column=1, sticky="e", padx=(0, 10), pady=(10, 0))

    st_types = ["1-Upsloping", "2-Flat", "3-Downsloping"]
    c5_1 = ttk.Combobox(window, values=st_types, state="readonly", width=25)
    c5_1.current(0)
    c5_1.grid(row=5, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    l5_2 = tk.Label(window, text="تعداد رگ های رنگ شده توسط فلوروسکپی", font=("Arial", 14))
    l5_2.grid(row=5, column=0, sticky="e", padx=(0, 10), pady=(10, 0))

    color_types = [str(i) for i in range(0, 4)]
    c5_2 = ttk.Combobox(window, values=color_types, state="readonly", width=10)
    c5_2.current(0)
    c5_2.grid(row=5, column=0, sticky="w", padx=(30, 0), pady=(10, 0))

    # row 6
    l6_1 = tk.Label(window, text="وضعیت تالاسمی", font=("Arial", 14))
    l6_1.grid(row=6, column=1, sticky="e", padx=(0, 10), pady=(10, 0))

    thal_types = ["1- Normal", "2- Fixed defect", "3- Reversible defect"]
    c6_1 = ttk.Combobox(window, values=thal_types, state="readonly", width=25)
    c6_1.current(0)
    c6_1.grid(row=6, column=1, sticky="w", padx=(30, 0), pady=(10, 0))

    def Risk_of_attack():
        userAge = int(e0_1.get())

        sex_map = {
            "زن": 0,
            "مرد": 1,
        }
        sex_selected = c0_1.get()
        userSex = sex_map.get(sex_selected)

        userCp = int(c1_1.get().split("-")[0])
        userTrestbps = int(e1_1.get())
        userChol = int(e2_1.get())

        fbs_map = {"بالای 120": 1, "کمتر یا مساوی 120": 0}
        fbs_selcted = c2_1.get()
        userFbs = fbs_map.get(fbs_selcted)

        userRestecg = int(c3_1.get().split("-")[0])
        userThalach = int(e3_1.get())
        userExang = int(v4.get())
        userOldpeak = float(e4_1.get())
        userSlope = (int(c5_1.get().split("-")[0])) - 1
        userCa = int(c5_2.get())
        userThal = int(c3_1.get().split("-")[0])

        user_result = [
            [
                userAge,
                userSex,
                userCp,
                userTrestbps,
                userChol,
                userFbs,
                userRestecg,
                userThalach,
                userExang,
                userOldpeak,
                userSlope,
                userCa,
                userThal,
            ]
        ]

        prob = model.predict_proba(user_result)[0][1]
        result_label.config(text=f"{prob*100:.0f} % : احتمال حمله قلبی")

    def clear_form():
        e0_1.delete(0, tk.END)
        e1_1.delete(0, tk.END)
        e2_1.delete(0, tk.END)
        e3_1.delete(0, tk.END)
        e4_1.delete(0, tk.END)
        c0_1.current(1)
        c1_1.current(0)
        c2_1.current(0)
        c3_1.current(0)
        c5_1.current(0)
        c5_2.current(0)
        c6_1.current(0)
        v4.set(0)

        result_label.config(text="")

    # row 7
    button_frame = tk.Frame(window)
    button_frame.grid(row=7, column=0, columnspan=2, pady=(30, 0))

    but7_1 = tk.Button(button_frame, text="محاسبه", command=Risk_of_attack, width=15)
    but7_1.pack(side="right", padx=10)

    but_clear = tk.Button(button_frame, text="پاک کردن", command=clear_form, width=15)
    but_clear.pack(side="left", padx=10)


    # row 8
    result_label = tk.Label(window, text="", font=("Arial", 14))
    result_label.grid(row=8,columnspan=2, pady=(25, 0))

    window.mainloop()


GUI()
