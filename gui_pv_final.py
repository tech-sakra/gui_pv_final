import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURARE
# ============================================================

TARGET = "ACPWRT"

data = None
model = None

features = []

X_test = None
y_test = None
pred_test = None

mae = None
rmse = None
r2 = None

feature_importance = None

csv_path = ""


# ============================================================
# FEREASTRA PRINCIPALA
# ============================================================

root = tk.Tk()

root.title(
    "Sistem Fotovoltaic Inteligent"
)

root.geometry(
    "1350x850"
)

root.minsize(
    1100,
    700
)

root.configure(
    bg="#ECEFF1"
)


# ============================================================
# CULORI
# ============================================================

BLUE = "#1565C0"
DARK_BLUE = "#0D47A1"

GREEN = "#2E7D32"
LIGHT_GREEN = "#E8F5E9"

ORANGE = "#EF6C00"

RED = "#C62828"

PURPLE = "#6A1B9A"

WHITE = "#FFFFFF"

GRAY = "#455A64"


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=DARK_BLUE
)

header.pack(
    fill="x"
)


tk.Label(
    header,
    text="☀ SISTEM FOTOVOLTAIC INTELIGENT",
    font=("Arial", 24, "bold"),
    fg=WHITE,
    bg=DARK_BLUE,
    pady=10
).pack()


tk.Label(
    header,
    text="Predicția producției fotovoltaice cu Machine Learning",
    font=("Arial", 12),
    fg=WHITE,
    bg=DARK_BLUE,
    pady=5
).pack()


# ============================================================
# STATUS
# ============================================================

status_var = tk.StringVar()

status_var.set(
    "Selectează un fișier CSV."
)


status = tk.Label(
    root,
    textvariable=status_var,
    bg="#CFD8DC",
    fg="#263238",
    anchor="w",
    padx=15,
    pady=7
)

status.pack(
    fill="x",
    side="bottom"
)


def set_status(text):

    status_var.set(text)

    root.update_idletasks()


# ============================================================
# BUTON
# ============================================================

def button(
    parent,
    text,
    command,
    color
):

    return tk.Button(

        parent,

        text=text,

        command=command,

        bg=color,

        fg=WHITE,

        activebackground=color,

        activeforeground=WHITE,

        font=("Arial", 11, "bold"),

        relief="flat",

        padx=15,

        pady=9,

        cursor="hand2"
    )


# ============================================================
# NOTEBOOK
# ============================================================

notebook = ttk.Notebook(
    root
)

notebook.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ============================================================
# TAB PRINCIPAL
# ============================================================

tab_main = tk.Frame(
    notebook,
    bg=WHITE
)

notebook.add(
    tab_main,
    text=" 🏠 Principal "
)


# ============================================================
# CSV
# ============================================================

top = tk.Frame(
    tab_main,
    bg=WHITE
)

top.pack(
    fill="x",
    padx=15,
    pady=15
)


file_label = tk.Label(
    top,
    text="Niciun CSV selectat",
    bg=WHITE,
    fg=GRAY,
    font=("Arial", 10)
)

file_label.pack(
    side="left",
    padx=15
)


def select_csv():

    global csv_path

    path = filedialog.askopenfilename(

        title="Selectează CSV",

        initialdir=os.path.dirname(
            os.path.abspath(__file__)
        ),

        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    if path:

        csv_path = path

        file_label.config(
            text=os.path.basename(path)
        )

        set_status(
            "CSV selectat."
        )


button(
    top,
    "📁 Selectează CSV",
    select_csv,
    GRAY
).pack(
    side="left"
)


# ============================================================
# ANTRENARE
# ============================================================

button(
    top,
    "🤖 Antrenează modelul",
    lambda: threading.Thread(
        target=train_model,
        daemon=True
    ).start(),
    GREEN
).pack(
    side="left",
    padx=5
)


# ============================================================
# METRICI
# ============================================================

metrics = tk.Frame(
    tab_main,
    bg=WHITE
)

metrics.pack(
    fill="x",
    padx=15
)


def create_metric(
    parent,
    title,
    color
):

    frame = tk.Frame(
        parent,
        bg="#FAFAFA",
        bd=1,
        relief="solid"
    )

    frame.pack(
        side="left",
        fill="both",
        expand=True,
        padx=5
    )

    tk.Label(
        frame,
        text=title,
        bg="#FAFAFA",
        fg=GRAY,
        font=("Arial", 11, "bold")
    ).pack(
        pady=(10, 2)
    )

    label = tk.Label(
        frame,
        text="-",
        bg="#FAFAFA",
        fg=color,
        font=("Arial", 20, "bold")
    )

    label.pack(
        pady=(0, 10)
    )

    return label


mae_label = create_metric(
    metrics,
    "MAE",
    BLUE
)

rmse_label = create_metric(
    metrics,
    "RMSE",
    ORANGE
)

r2_label = create_metric(
    metrics,
    "R²",
    GREEN
)

rows_label = create_metric(
    metrics,
    "Date",
    PURPLE
)


# ============================================================
# INFO
# ============================================================

info_frame = tk.LabelFrame(
    tab_main,
    text="Informații",
    bg=WHITE,
    font=("Arial", 12, "bold")
)

info_frame.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=15
)


info_text = tk.Text(
    info_frame,
    bg="#FAFAFA",
    font=("Consolas", 10)
)

info_text.pack(
    fill="both",
    expand=True,
    padx=5,
    pady=5
)


def info(text):

    info_text.insert(
        tk.END,
        text + "\n"
    )

    info_text.see(
        tk.END
    )


info(
    "SISTEM FOTOVOLTAIC INTELIGENT"
)

info(
    "=" * 70
)

info(
    "Selectează CSV-ul și apasă «Antrenează modelul»."
)


# ============================================================
# TAB PREDICTIE
# ============================================================

tab_prediction = tk.Frame(
    notebook,
    bg=WHITE
)

notebook.add(
    tab_prediction,
    text=" ⚡ Predicție "
)


tk.Label(
    tab_prediction,
    text="⚡ PREDICȚIE PRODUCȚIE",
    font=("Arial", 21, "bold"),
    bg=WHITE,
    fg=DARK_BLUE
).pack(
    pady=15
)


tk.Label(
    tab_prediction,
    text=(
        "Modifică valorile direct în GUI. "
        "CSV-ul nu va fi modificat."
    ),
    font=("Arial", 11),
    bg=WHITE,
    fg=GRAY
).pack(
    pady=(0, 10)
)


# ============================================================
# FRAME PENTRU CAMPURI
# ============================================================

input_container = tk.Frame(
    tab_prediction,
    bg=WHITE
)

input_container.pack(
    pady=10
)


# Acestea sunt valorile pe care utilizatorul le poate modifica.
#
# ACPWRT NU apare aici deoarece acesta este rezultatul pe care
# modelul trebuie să îl prezică.

input_columns = [

    "DAY",
    "HOUR",

    "DCVOLT",
    "RESIST",
    "TEMPER",

    "VL1TO2",
    "VL2TO3",
    "VL3TO1",

    "ACCUR1",
    "ACCUR2",
    "ACCUR3",

    "ACVLT1",
    "ACVLT2",
    "ACVLT3",

    "ACFRQ1",
    "ACFRQ2",
    "ACFRQ3",

    "APPWR1",
    "APPWR2",
    "APPWR3",

    "ACPWR1",
    "ACPWR2",
    "ACPWR3",

    "REPWR1",
    "REPWR2",
    "REPWR3"
]


input_entries = {}


# ============================================================
# CREARE CAMPI GUI
# ============================================================

for index, column in enumerate(input_columns):

    row = index % 7

    col = (index // 7) * 2

    tk.Label(
        input_container,
        text=column + ":",
        bg=WHITE,
        fg=GRAY,
        font=("Arial", 9, "bold"),
        width=12,
        anchor="e"
    ).grid(
        row=row,
        column=col,
        padx=5,
        pady=4
    )

    entry = tk.Entry(
        input_container,
        width=14,
        font=("Arial", 10)
    )

    entry.grid(
        row=row,
        column=col + 1,
        padx=5,
        pady=4
    )

    input_entries[column] = entry


# ============================================================
# BUTOANE PENTRU DATE
# ============================================================

button_frame = tk.Frame(
    tab_prediction,
    bg=WHITE
)

button_frame.pack(
    pady=10
)


def fill_from_csv():

    if data is None:

        messagebox.showwarning(
            "Date",
            "Mai întâi antrenează modelul."
        )

        return

    dialog = tk.Toplevel(
        root
    )

    dialog.title(
        "Alege un rând din CSV"
    )

    dialog.geometry(
        "350x180"
    )

    dialog.configure(
        bg=WHITE
    )

    tk.Label(
        dialog,
        text=f"Rând între 0 și {len(data)-1}:",
        bg=WHITE,
        font=("Arial", 11, "bold")
    ).pack(
        pady=20
    )

    entry = tk.Entry(
        dialog,
        font=("Arial", 12)
    )

    entry.pack()


    def load():

        try:

            index = int(
                entry.get()
            )

        except:

            messagebox.showerror(
                "Eroare",
                "Introdu un număr valid."
            )

            return


        if index < 0 or index >= len(data):

            messagebox.showerror(
                "Eroare",
                "Rând invalid."
            )

            return


        row = data.iloc[index]


        for column in input_columns:

            if column in row.index:

                value = row[column]

                if pd.notna(value):

                    input_entries[
                        column
                    ].delete(
                        0,
                        tk.END
                    )

                    input_entries[
                        column
                    ].insert(
                        0,
                        str(value)
                    )


        dialog.destroy()

        set_status(
            f"Rândul {index} a fost încărcat. "
            "Acum poți modifica orice valoare."
        )


    button(
        dialog,
        "📋 Încarcă",
        load,
        BLUE
    ).pack(
        pady=15
    )


button(
    button_frame,
    "📋 Încarcă rând din CSV",
    fill_from_csv,
    BLUE
).pack(
    side="left",
    padx=5
)


# ============================================================
# BUTON RESET
# ============================================================

def clear_fields():

    for entry in input_entries.values():

        entry.delete(
            0,
            tk.END
        )


    prediction_label.config(
        text="ACPWRT estimat: -"
    )

    state_label.config(
        text="Stare: -",
        fg=GRAY
    )

    real_label.config(
        text="ACPWRT real: -"
    )

    difference_label.config(
        text="Diferență: -"
    )


button(
    button_frame,
    "🗑 Șterge valorile",
    clear_fields,
    RED
).pack(
    side="left",
    padx=5
)


# ============================================================
# REZULTAT
# ============================================================

prediction_label = tk.Label(
    tab_prediction,
    text="ACPWRT estimat: -",
    bg=LIGHT_GREEN,
    fg=GREEN,
    font=("Arial", 20, "bold"),
    padx=30,
    pady=15
)

prediction_label.pack(
    pady=10
)


state_label = tk.Label(
    tab_prediction,
    text="Stare: -",
    bg=WHITE,
    fg=GRAY,
    font=("Arial", 14, "bold")
)

state_label.pack(
    pady=5
)


real_label = tk.Label(
    tab_prediction,
    text="ACPWRT real: -",
    bg=WHITE,
    fg=GRAY,
    font=("Arial", 11, "bold")
)

real_label.pack()


difference_label = tk.Label(
    tab_prediction,
    text="Diferență: -",
    bg=WHITE,
    fg=GRAY,
    font=("Arial", 11)
)

difference_label.pack(
    pady=5
)


# ============================================================
# PREDICTIE
# ============================================================

def predict():

    if model is None:

        messagebox.showwarning(
            "Model",
            "Mai întâi antrenează modelul."
        )

        return


    try:

        values = {}


        # --------------------------------------------------------
        # CITIRE VALORI GUI
        # --------------------------------------------------------

        for column, entry in input_entries.items():

            text = entry.get().strip()


            if text == "":

                raise ValueError(
                    f"Completează câmpul {column}."
                )


            values[column] = float(
                text.replace(
                    ",",
                    "."
                )
            )


        # --------------------------------------------------------
        # CREARE RAND NOU
        # --------------------------------------------------------

        new_row = {}


        for feature in features:

            if feature in values:

                new_row[feature] = values[
                    feature
                ]

            else:

                # Pentru caracteristici care nu sunt introduse
                # manual, folosim media din datele de antrenare.

                if feature in data.columns:

                    value = pd.to_numeric(
                        data[feature],
                        errors="coerce"
                    ).median()

                    if pd.isna(value):

                        value = 0

                    new_row[feature] = value

                else:

                    new_row[feature] = 0


        X_new = pd.DataFrame(
            [new_row]
        )


        X_new = X_new[
            features
        ]


        # --------------------------------------------------------
        # PREDICTIE
        # --------------------------------------------------------

        prediction = float(
            model.predict(
                X_new
            )[0]
        )


        # --------------------------------------------------------
        # AFISARE
        # --------------------------------------------------------

        prediction_label.config(

            text=(
                "ACPWRT estimat:\n"
                f"{prediction:.4f}"
            )
        )


        # --------------------------------------------------------
        # STARE
        # --------------------------------------------------------

        target = pd.to_numeric(
            data[TARGET],
            errors="coerce"
        ).dropna()


        q25 = target.quantile(
            0.25
        )

        q75 = target.quantile(
            0.75
        )


        if prediction <= q25:

            state_label.config(
                text="🔴 Stare: Producție scăzută",
                fg=RED
            )

        elif prediction <= q75:

            state_label.config(
                text="🟢 Stare: Producție normală",
                fg=GREEN
            )

        else:

            state_label.config(
                text="🔵 Stare: Producție ridicată",
                fg=BLUE
            )


        difference_label.config(
            text=(
                "Diferență: disponibilă "
                "doar când este încărcat un rând real."
            )
        )


        set_status(
            "Predicția a fost calculată."
        )


    except Exception as error:

        messagebox.showerror(
            "Eroare",
            str(error)
        )


button(
    tab_prediction,
    "⚡ CALCULEAZĂ PREDICȚIA",
    predict,
    GREEN
).pack(
    pady=10
)


# ============================================================
# TAB GRAFICE
# ============================================================

tab_graph = tk.Frame(
    notebook,
    bg=WHITE
)

notebook.add(
    tab_graph,
    text=" 📊 Grafice "
)


tk.Label(
    tab_graph,
    text="📊 Grafice și evaluare",
    font=("Arial", 20, "bold"),
    bg=WHITE,
    fg=DARK_BLUE
).pack(
    pady=20
)


def check_model():

    if model is None:

        messagebox.showwarning(
            "Model",
            "Mai întâi antrenează modelul."
        )

        return False

    return True


# ============================================================
# REAL VS PREDICTIE
# ============================================================

def plot_real_prediction():

    if not check_model():
        return


    count = min(
        300,
        len(y_test)
    )


    plt.figure(
        figsize=(11, 6)
    )


    plt.plot(
        y_test.iloc[:count].values,
        label="Real",
        color="blue"
    )


    plt.plot(
        pred_test[:count],
        label="Predicție",
        color="red"
    )


    plt.title(
        "ACPWRT - Real vs Predicție"
    )

    plt.xlabel(
        "Observație"
    )

    plt.ylabel(
        "ACPWRT"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


button(
    tab_graph,
    "📈 Real vs Predicție",
    plot_real_prediction,
    BLUE
).pack(
    pady=10
)


# ============================================================
# IMPORTANTA
# ============================================================

def plot_importance():

    if not check_model():
        return


    values = feature_importance.sort_values(
        "Importanta"
    )


    plt.figure(
        figsize=(10, 8)
    )


    plt.barh(
        values["Caracteristica"],
        values["Importanta"],
        color=GREEN
    )


    plt.title(
        "Importanța caracteristicilor"
    )

    plt.xlabel(
        "Importanță"
    )

    plt.tight_layout()

    plt.show()


button(
    tab_graph,
    "🔍 Importanța caracteristicilor",
    plot_importance,
    ORANGE
).pack(
    pady=10
)


# ============================================================
# SCATTER
# ============================================================

def plot_scatter():

    if not check_model():
        return


    plt.figure(
        figsize=(8, 8)
    )


    plt.scatter(
        y_test,
        pred_test,
        alpha=0.5,
        color=BLUE
    )


    minimum = min(
        y_test.min(),
        pred_test.min()
    )


    maximum = max(
        y_test.max(),
        pred_test.max()
    )


    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--"
    )


    plt.xlabel(
        "ACPWRT real"
    )

    plt.ylabel(
        "ACPWRT estimat"
    )

    plt.title(
        "Real vs Estimat"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


button(
    tab_graph,
    "🔵 Scatter plot",
    plot_scatter,
    PURPLE
).pack(
    pady=10
)


# ============================================================
# TAB CARACTERISTICI
# ============================================================

tab_features = tk.Frame(
    notebook,
    bg=WHITE
)

notebook.add(
    tab_features,
    text=" 🔍 Caracteristici "
)


tk.Label(
    tab_features,
    text="Importanța caracteristicilor",
    font=("Arial", 20, "bold"),
    bg=WHITE,
    fg=DARK_BLUE
).pack(
    pady=20
)


feature_list = tk.Listbox(
    tab_features,
    width=70,
    height=25,
    font=("Consolas", 11)
)

feature_list.pack(
    padx=20,
    pady=10
)


# ============================================================
# ANTRENARE MODEL
# ============================================================

def train_model():

    global data
    global model
    global features
    global X_test
    global y_test
    global pred_test
    global mae
    global rmse
    global r2
    global feature_importance


    try:

        if csv_path == "":

            messagebox.showwarning(
                "CSV",
                "Selectează mai întâi fișierul CSV."
            )

            return


        set_status(
            "Se citește CSV-ul..."
        )


        info_text.delete(
            "1.0",
            tk.END
        )


        # --------------------------------------------------------
        # CITIRE
        # --------------------------------------------------------

        data = pd.read_csv(
            csv_path
        )


        data.columns = (
            data.columns
            .astype(str)
            .str.strip()
        )


        if TARGET not in data.columns:

            raise ValueError(
                "Coloana ACPWRT nu există în CSV."
            )


        info(
            "CSV încărcat cu succes."
        )

        info(
            f"Rânduri inițiale: {len(data)}"
        )


        # --------------------------------------------------------
        # CONVERSIE
        # --------------------------------------------------------

        for column in data.columns:

            if column != "DAY":

                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )


        # --------------------------------------------------------
        # CARACTERISTICI
        #
        # Nu folosim ACPWRT ca input.
        # ACPWRT este ținta.
        # --------------------------------------------------------

        possible_features = [

            "HOUR",

            "DCVOLT",

            "RESIST",

            "TEMPER",

            "VL1TO2",

            "VL2TO3",

            "VL3TO1",

            "ACCUR1",

            "ACCUR2",

            "ACCUR3",

            "ACVLT1",

            "ACVLT2",

            "ACVLT3",

            "ACFRQ1",

            "ACFRQ2",

            "ACFRQ3",

            "APPWR1",

            "APPWR2",

            "APPWR3",

            "ACPWR1",

            "ACPWR2",

            "ACPWR3",

            "REPWR1",

            "REPWR2",

            "REPWR3"
        ]


        features = [

            column

            for column in possible_features

            if column in data.columns
        ]


        if len(features) < 3:

            raise ValueError(
                "Nu există suficiente coloane pentru model."
            )


        # --------------------------------------------------------
        # DATASET
        # --------------------------------------------------------

        dataset = data[
            features + [TARGET]
        ].copy()


        dataset = dataset.replace(
            [np.inf, -np.inf],
            np.nan
        )


        # --------------------------------------------------------
        # UMPLERE VALORI LIPSĂ
        # --------------------------------------------------------

        for column in features:

            median = dataset[
                column
            ].median()


            if pd.isna(median):

                median = 0


            dataset[column] = (
                dataset[column]
                .fillna(median)
            )


        # Target-ul lipsă nu poate fi folosit.

        dataset = dataset.dropna(
            subset=[TARGET]
        )


        if len(dataset) < 30:

            raise ValueError(
                "Sunt prea puține date pentru antrenare."
            )


        # --------------------------------------------------------
        # X / Y
        # --------------------------------------------------------

        X = dataset[
            features
        ]


        y = dataset[
            TARGET
        ]


        # --------------------------------------------------------
        # SPLIT TEMPORAL
        # --------------------------------------------------------

        split = int(
            len(X) * 0.80
        )


        X_train = X.iloc[
            :split
        ]


        y_train = y.iloc[
            :split
        ]


        X_test = X.iloc[
            split:
        ]


        y_test = y.iloc[
            split:
        ]


        if len(X_test) == 0:

            raise ValueError(
                "Setul de test este gol."
            )


        # --------------------------------------------------------
        # MODEL
        # --------------------------------------------------------

        set_status(
            "Se creează modelul Random Forest..."
        )


        model = RandomForestRegressor(

            n_estimators=300,

            max_depth=None,

            min_samples_split=2,

            min_samples_leaf=1,

            max_features=1.0,

            random_state=42,

            n_jobs=-1
        )


        set_status(
            "Se începe antrenarea modelului..."
        )


        model.fit(
            X_train,
            y_train
        )


        # --------------------------------------------------------
        # TEST
        # --------------------------------------------------------

        pred_test = model.predict(
            X_test
        )


        # --------------------------------------------------------
        # METRICI
        # --------------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            pred_test
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred_test
            )
        )


        r2 = r2_score(
            y_test,
            pred_test
        )


        # --------------------------------------------------------
        # IMPORTANTA
        # --------------------------------------------------------

        feature_importance = pd.DataFrame({

            "Caracteristica": features,

            "Importanta":
                model.feature_importances_

        })


        feature_importance = (
            feature_importance
            .sort_values(
                "Importanta",
                ascending=False
            )
        )


        # --------------------------------------------------------
        # GUI
        # --------------------------------------------------------

        mae_label.config(
            text=f"{mae:.4f}"
        )


        rmse_label.config(
            text=f"{rmse:.4f}"
        )


        r2_label.config(
            text=f"{r2:.4f}"
        )


        rows_label.config(
            text=str(len(dataset))
        )


        info(
            ""
        )

        info(
            "=" * 70
        )

        info(
            "REZULTATELE MODELULUI"
        )

        info(
            "=" * 70
        )

        info(
            f"MAE  = {mae:.4f}"
        )

        info(
            f"RMSE = {rmse:.4f}"
        )

        info(
            f"R²   = {r2:.4f}"
        )

        info(
            ""
        )

        info(
            "CARACTERISTICI UTILIZATE:"
        )


        for feature in features:

            info(
                " - " + feature
            )


        info(
            ""
        )

        info(
            "TOP CARACTERISTICI:"
        )


        for _, row in feature_importance.head(10).iterrows():

            info(
                f"{row['Caracteristica']:<20}"
                f"{row['Importanta']:.5f}"
            )


        feature_list.delete(
            0,
            tk.END
        )


        for _, row in feature_importance.iterrows():

            feature_list.insert(

                tk.END,

                f"{row['Caracteristica']:<25}"
                f"{row['Importanta']:.6f}"
            )


        set_status(
            "Modelul a fost antrenat cu succes."
        )


        messagebox.showinfo(

            "Succes",

            (
                "Modelul a fost antrenat!\n\n"

                f"MAE = {mae:.4f}\n"

                f"RMSE = {rmse:.4f}\n"

                f"R² = {r2:.4f}\n\n"

                "Poți merge acum la "
                "⚡ Predicție."
            )
        )


    except Exception as error:

        set_status(
            "Eroare la antrenare."
        )


        messagebox.showerror(
            "Eroare",
            str(error)
        )


# ============================================================
# START
# ============================================================

root.mainloop()
