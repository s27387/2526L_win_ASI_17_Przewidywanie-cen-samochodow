# RevRate - przewidywanie cen samochodow
Celem projektu jest predykcja ceny samochodu na podstawie danych z ogloszen sprzedazy samochodow w Polsce.

Projekt zawiera baseline w notebooku, refaktoryzacje do pipeline'ow Kedro, eksperymenty MLflow, porownanie modeli, pipeline AutoGluon oraz lokalne API predykcyjne w FastAPI.

## Dane

Zrodlem danych jest dataset Kaggle:

```text
bartoszpieniak/poland-cars-for-sale-dataset
```

Pipeline pobiera plik `Car_sale_ads.csv` automatycznie, jezeli nie ma go lokalnie w katalogu `data/`. Dane lokalne nie sa wersjonowane w repozytorium.

Do pobierania danych potrzebny jest token Kaggle. Przykladowy szkielet znajduje sie w pliku `kaggle.example.json`:

```json
{
  "username": "twoj_login_kaggle",
  "key": "twoj_klucz_api"
}
```

Najprostsza opcja to umiescic prawdziwy plik jako:

```powershell
%USERPROFILE%\.kaggle\kaggle.json
```

Alternatywnie mozna trzymac go lokalnie w katalogu projektu jako `kaggle.json` i ustawic przed uruchomieniem:

```powershell
$env:KAGGLE_CONFIG_DIR = (Get-Location).Path
```

Prawdziwy `kaggle.json` jest ignorowany przez Git i nie powinien trafic do repozytorium.

## Struktura projektu

```text
conf/base/parameters.yml                      konfiguracja pipeline'ow
conf/base/catalog.yml                         definicje artefaktow Kedro
notebooks/pipeline.ipynb                      notebook baseline
src/revrate/pipelines/custom_pipeline/        reczny pipeline ML
src/revrate/pipelines/autogluon_pipeline/     pipeline AutoML
src/revrate/api/app.py                        lokalne API FastAPI
models/                                       lokalne modele, ignorowane przez Git
data/                                         lokalne dane, ignorowane przez Git
mlflow.db                                     lokalna baza MLflow, ignorowana przez Git
```

## Uruchomienie

Instalacja srodowiska:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Uruchomienie recznego pipeline'u:

```powershell
python -B -m kedro run --pipelines custom_pipeline
```

Uruchomienie pipeline'u AutoGluon:

```powershell
python -B -m kedro run --pipelines autogluon_pipeline
```

Domyslny pipeline Kedro to `custom_pipeline`, wiec ponizsza komenda uruchamia reczny wariant:

```powershell
python -B -m kedro run
```

## Pipeline custom

Reczny pipeline obejmuje:

- pobieranie danych z Kaggle,
- czyszczenie danych,
- uzupelnianie brakow,
- inzynierie cech,
- selekcje cech,
- preprocessing,
- trening modelu `RandomForestRegressor`,
- strojenie hiperparametrow przez `GridSearchCV`,
- ewaluacje na zbiorze treningowym i testowym,
- zapis artefaktow modelu do katalogu `models/`,
- logowanie eksperymentow w MLflow.

Zapisywane artefakty:

```text
models/custom_model.pkl
models/custom_preprocessor.pkl
models/custom_top_features.pkl
models/reference_stats.json
```

## Pipeline AutoGluon

Pipeline AutoGluon obejmuje:

- pobieranie tych samych danych,
- preprocessing i inzynierie cech,
- podzial train/test,
- trening modeli AutoGluon,
- ewaluacje najlepszego modelu,
- logowanie wynikow i artefaktow w MLflow.

W obecnej konfiguracji AutoGluon porownuje:

- `KNeighbors`,
- `RandomForest`,
- `ExtraTrees`,
- `WeightedEnsemble`.

Modele AutoGluon sa zapisywane w katalogach:

```text
models/autogluon_models/run_YYYYMMDD_HHMMSS/
```

## Wyniki modeli

Metryki dla zbioru testowego:

| Model | Val RMSE | Test RMSE | Test MAE | Test R2 |
|---|---:|---:|---:|---:|
| Custom RandomForest + GridSearchCV | - | **10110.71** | **5780.98** | **0.9199** |
| AutoGluon KNeighbors | 21782.49 | 22351.45 | 14837.13 | 0.6138 |
| AutoGluon RandomForest | 11299.58 | 11400.61 | 6652.61 | 0.8995 |
| AutoGluon ExtraTrees | 11447.63 | 11337.18 | 6754.93 | 0.9006 |
| AutoGluon WeightedEnsemble | 11209.69 | 11229.80 | 6595.02 | 0.9025 |

Wniosek: najlepszy wynik na zbiorze testowym uzyskal recznie przygotowany `RandomForestRegressor` po selekcji cech i strojeniu hiperparametrow przez `GridSearchCV`.

## MLflow

Eksperymenty sa logowane lokalnie przez MLflow i zapisywane w pliku:

```text
mlflow.db
```

W custom pipeline logowane sa m.in. parametry modelu, metryki, wykresy ewaluacyjne, waznosci cech oraz artefakt modelu. W AutoGluon logowany jest leaderboard, metryki i katalog modelu.

## API

Po uruchomieniu `custom_pipeline` mozna wystartowac lokalne API predykcyjne:

```powershell
python -B -m uvicorn src.revrate.api.app:app --host 127.0.0.1 --port 8000
```

Dostepne endpointy:

```text
GET  /health
POST /predict
GET  /monitoring/summary
```

Dokumentacja Swagger UI:

```text
http://127.0.0.1:8000/docs
```

API korzysta z artefaktow zapisanych przez `custom_pipeline`, dlatego przed wykonaniem predykcji trzeba miec wygenerowane pliki w katalogu `models/`. Dokumentacja `/docs` i endpoint `/health` uruchamiaja sie bez ladowania modelu; model jest ladowany dopiero przy pierwszym wywolaniu `/predict`.

Odpowiedz `/predict` zawiera:

```text
predicted_price
currency
drift_detected
drift_warnings
```

## Monitoring API i drift danych

API zapisuje kazda predykcje do lokalnego pliku:

```text
logs/predictions.csv
```

W logu zapisywane sa:

- czas predykcji,
- przewidziana cena,
- czas obslugi requestu,
- informacja czy wykryto drift,
- lista ostrzezen driftu,
- dane wejsciowe requestu.

Prosty drift detection opiera sie na pliku:

```text
models/reference_stats.json
```

Plik jest generowany przez `custom_pipeline` i zawiera statystyki referencyjne danych treningowych. API porownuje nowe requesty z tym punktem odniesienia:

- dla cech numerycznych sprawdza, czy wartosc wychodzi poza zakres `p01-p99`,
- dla cech kategorycznych sprawdza, czy wartosc wystepowala w danych treningowych.

Podsumowanie monitoringu mozna sprawdzic przez:

```text
GET /monitoring/summary
```

Endpoint zwraca m.in. liczbe predykcji, liczbe predykcji z wykrytym driftem, wspolczynnik driftu i czas ostatniej predykcji.

## Status wymagan

| Punkt wymagan | Status | Co jest w projekcie |
|---|---|---|
| 1. Organizacja projektu i zespolu | Zrobione czesciowo | Repozytorium, struktura projektu, srodowisko Python i plik `requirements.txt`. Historia zmian zalezy od repozytorium Git. |
| 2. Wersja podstawowa modelu | Zrobione | Notebook `notebooks/pipeline.ipynb` zawiera EDA, preprocessing, trening i ewaluacje modelu. |
| 3. Struktura projektu i pipeline | Zrobione | Kod podzielony na moduly, pipeline Kedro zawiera pobieranie danych, preprocessing, trening i ewaluacje. |
| 4. Udoskonalanie modelu | Zrobione | MLflow, AutoGluon, inzynieria cech, selekcja cech, GridSearchCV i porownanie kilku modeli. |
| 5. Pipeline produkcyjny | Zrobione czesciowo | Lokalne API FastAPI, artefakty modelu, logowanie predykcji i prosty drift detection. Do dopracowania zostaje ewentualny Docker/chmura i bardziej rozbudowana optymalizacja. |
| 6. MLOps / repozytoria | Zrobione czesciowo | MLflow i lokalne artefakty modelu. Brak DVC/Feast albo pelnego CI/CD/Continuous Training. |
| 7. Dokumentacja | Zrobione czesciowo | README opisuje projekt, dane, pipeline, uruchomienie, wyniki, API i monitoring. Brakuje jeszcze diagramu architektury. |
| 8. Podsumowanie i prezentacja | Do przygotowania | Wyniki i demo API sa gotowe jako material do prezentacji. |

## Co jest gotowe do pokazania

- uruchomienie `custom_pipeline`,
- uruchomienie `autogluon_pipeline`,
- porownanie wynikow kilku modeli,
- lokalne API FastAPI po wytrenowaniu custom modelu,
- monitoring predykcji i prosty drift detection,
- zapis modeli i eksperymentow.

## Najwazniejsze pliki

```text
Wymagania.txt
README.md
requirements.txt
kaggle.example.json
conf/base/parameters.yml
src/revrate/pipelines/custom_pipeline/nodes.py
src/revrate/pipelines/autogluon_pipeline/nodes.py
src/revrate/api/app.py
notebooks/pipeline.ipynb
```
