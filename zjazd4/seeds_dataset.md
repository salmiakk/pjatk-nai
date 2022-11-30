# Klasyfikacja nasion pszenicy

## Autorzy
- Mateusz Pioch
- Stanisław Dominiak

## Cel
Za pomocą SVM oraz drzew decyzyjnych, przeprowadzić klasyfikację danych dotyczących nasion pszenicy oraz przewidzieć jej podgatunek (Kama, Rosa lub Canadian).

## Potrzebne narzędzia
Do uruchomienia zadania został użyty Python w wersji 3.8.5 oraz nastepujące biblioteki:
- numpy
- pandas
- sklearn
- matplotlib

## Ładowanie danych

```python
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("dane1.csv", delimiter=";")
```
![Alt text](pszenica_zaladowana.png?raw=true "załadowane dane")
