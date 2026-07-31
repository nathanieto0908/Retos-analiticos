# Proyecto de Forecast de Recaudo

## Estructura

    src/
    ├── data/
    │   └── raw/
    │   └── results/
    ├── models/
    ├── notebook/
    └── requirements.txt

## Descripción

Este proyecto contiene el análisis exploratorio, modelado y artefactos
de un modelo de pronóstico de recaudo.

### Carpetas

-   **data/raw/**: datos de entrada.
-   **data/results/**: Predicciones.
-   **models/**: modelos entrenados (.pkl).
-   **notebook/**: notebooks de EDA, modelado y funciones auxiliares.
-   **requirements.txt**: dependencias del proyecto.

## Instalación

``` bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

## Flujo de trabajo

1.  Colocar los datos en `data/raw/`.
2.  Ejecutar `notebook/eda.ipynb`.
3.  Ejecutar `notebook/modelado.ipynb`.
4.  El modelo entrenado se almacena en `models/`.

## Requisitos

-   Python 3.11.9
-   Dependencias definidas en `requirements.txt`

## Autor

Proyecto de pronóstico de recaudo.
