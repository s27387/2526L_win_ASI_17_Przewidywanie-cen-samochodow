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

## Automatyzacja projektu

Repozytorium zawiera workflow GitHub Actions:

```text
.github/workflows/ci.yml
```

Workflow uruchamia sie po `push`, `pull_request` albo recznie z zakladki Actions w GitHubie. Jego zadaniem jest szybkie sprawdzenie, czy projekt nadal sklada sie technicznie po zmianach w kodzie.

CI wykonuje:

- instalacje zaleznosci z `requirements.txt`,
- sprawdzenie skladni plikow Python w `src/`,
- sprawdzenie, czy Kedro rejestruje pipeline'y `custom_pipeline` i `autogluon_pipeline`,
- sprawdzenie, czy FastAPI buduje schemat OpenAPI i widzi endpointy `/health`, `/predict`, `/monitoring/summary`.

CI nie trenuje modelu i nie pobiera danych z Kaggle. Te kroki sa ciezsze, wymagaja lokalnych danych lub tokenu Kaggle i sa uruchamiane recznie.

Ponowne trenowanie modelu wykonuje sie komenda:

```powershell
python -B -m kedro run --pipelines custom_pipeline
```

Po takim uruchomieniu odswiezane sa artefakty w katalogu `models/`, m.in. model, preprocessor, lista cech oraz statystyki referencyjne do drift detection. AutoGluon mozna uruchomic osobno:

```powershell
python -B -m kedro run --pipelines autogluon_pipeline
```

Wdrozenie modelu w projekcie jest lokalne i odbywa sie przez FastAPI/Uvicorn:

```powershell
python -B -m uvicorn src.revrate.api.app:app --host 127.0.0.1 --port 8000
```

Po starcie API model jest dostepny przez endpoint `/predict`, a monitoring przez `/monitoring/summary`.

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
