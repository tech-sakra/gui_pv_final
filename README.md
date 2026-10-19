# gui_pv_final
Predicția producției fotovoltaice
# Predicția Producției Fotovoltaice - Documentație Proiect

Această documentație prezintă structura, logica și funcționalitatea aplicației Python dezvoltate pentru estimarea puterii active totale a unui sistem fotovoltaic, utilizând tehnici de Machine Learning și o interfață grafică interactivă.

## 👤 Date Identificare (Opțional)
* **Student:** [Numele Tău]
* **Grupa:** [Grupa Ta]

---

## 📁 1. Datele de Intrare (Dataset)
Programul utilizează un fișier în format **CSV** care conține date măsurate direct de la un sistem fotovoltaic. Parametrii incluși în setul de date sunt:

* ⚡ **DCVOLT** – Tensiunea pe partea de curent continuu (DC).
* ⚙️ **RESIST** – Rezistența măsurată în sistem.
* 🌡️ **TEMPER** – Temperatura ambientală / a panourilor.
* 📊 **ACVLT1, ACVLT2, ACVLT3** – Tensiunile pe cele trei faze de curent alternativ (AC).
* 📈 **ACCUR1, ACCUR2, ACCUR3** – Curenții pe cele trei faze AC.
* 🔄 **ACFRQ1, ACFRQ2, ACFRQ3** – Frecvențele măsurate pe rețeaua AC.
* 🔋 **APPWR1, APPWR2, APPWR3** – Puterile aparente pe fiecare fază.
* 💡 **ACPWR1, ACPWR2, ACPWR3** – Puterile active pe fiecare fază.
* 📉 **REPWR1, REPWR2, REPWR3** – Puterile reactive pe fiecare fază.

🎯 **Obiectivul modelului:** Estimarea valorii **ACPWRT**, care reprezintă **puterea activă totală** generată de sistem.

---

## ⚙️ 2. Pregătirea Datelor (Data Preprocessing)
Procesul de prelucrare a datelor înainte de antrenare urmează următorii pași:
1. **Citirea fișierului:** Datele sunt încărcate din fișierul CSV folosind biblioteca `pandas`.
2. **Validare și conversie:** Toate datele sunt verificate și transformate în valori numerice.
3. **Tratarea valorilor lipsă (Imputation):** Dacă există date lipsă în setul de măsurători, programul le completează automat folosind **valoarea mediană** a coloanei respective.
4. **Împărțirea setului de date (Split):** Datele sunt divizate în:
   * **80%** pentru antrenarea modelului (Train set).
   * **20%** pentru testarea și validarea performanței (Test set).

---

## 🤖 3. Modelul de Machine Learning
Pentru realizarea predicției a fost ales algoritmul **Random Forest Regressor**.

* **Principiu de funcționare:** Random Forest este un algoritm de tip ansamblu, format din mai mulți arbori de decizie independenți. Fiecare arbore realizează o estimare proprie, iar rezultatele lor sunt combinate (prin mediere) pentru a genera predicția finală.
* **Motivația alegerii:** Acest algoritm este ideal pentru sistemele fotovoltaice deoarece funcționează excelent cu un număr mare de variabile de intrare și are capacitatea de a identifica relații non-liniare și complexe între parametrii sistemului hardware.

---

## 📉 4. Evaluarea Modelului (Metrics)
După faza de antrenare, modelul este evaluat strict pe baza datelor de test (pe care nu le-a văzut în timpul antrenării), calculându-se trei indicatori statistici principali:
* **MAE (Mean Absolute Error):** Arată eroarea medie absolută dintre valorile reale și cele estimate.
* **RMSE (Root Mean Squared Error):** Penalizează mai sever erorile mari de predicție.
* **R² (R-squared / Coeficientul de determinare):** Indică cât de bine explică modelul variația valorilor reale. Cu cât \(R^2\) este mai aproape de 1, cu atât potrivirea modelului cu datele reale este mai precisă.

---

## 🖥️ 5. Interfața Grafică (GUI)
Aplicația dispune de o interfață grafică intuitivă care permite utilizatorului:
1. **Încărcarea datelor:** Selectarea fișierului CSV direct din calculator.
2. **Antrenare:** Pornirea procesului de antrenare a algoritmului printr-un simplu buton.
3. **Vizualizare rezultate:** Afișarea metricilor de evaluare și a graficelor de performanță.
4. **Modul de Simulare / Predicție manuală:** O secțiune specială unde utilizatorul poate modifica manual parametrii de intrare pentru a vedea impactul direct asupra producției.
   * *Exemplu:* Modificarea temperaturii (`TEMPER` de la 25 → 35) sau a tensiunii (`DCVOLT` de la 500 → 550).
   * La apăsarea butonului **`CALCULEAZĂ PREDICȚIA`**, modelul generează instant noua valoare estimată pentru **ACPWRT**.

---

## 📊 6. Grafice și Vizualizări
Programul generează automat două tipuri de reprezentări grafice:
1. **Comparație Valori Reale vs. Estimate:** Un grafic liniar/punctual care arată cât de aproape sunt predicțiile modelului față de măsurătorile reale din teren.
2. **Importanța Caracteristicilor (Feature Importance):** Un grafic cu bare care evidențiază care dintre parametrii sistemului (ex: tensiunea DC, temperatura) au avut cea mai mare greutate și influență în deciziile luate de model.
