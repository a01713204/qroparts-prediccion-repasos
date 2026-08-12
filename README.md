# QroParts - Modelos ML y Dashboard Shiny

## Archivos principales

- `01_entrenar_modelos_qroparts.R`  
  Entrena los modelos nuevos:
  - Tiempo de entrega: Random Forest y XGBoost clasificador.
  - Costos de transporte: Random Forest y XGBoost clasificador.
  - Inventario: arbol de decision.
  - Rutas: Random Forest.

- `app.R`  
  Dashboard Shiny con:
  - Dashboard ejecutivo.
  - Pestaña de tiempo de entrega.
  - Pestaña de costos de transporte.
  - Pestaña de inventario.
  - Pestaña de rutas.
  - Simulador operativo.

## Como correrlo

1. Coloca en esta misma carpeta tu archivo:

   `QroParts.sqlite`

   o, si ya lo tienes exportado:

   `dataset_logistico_qroparts.csv`

2. Abre RStudio en esta carpeta.

3. Corre primero:

```r
source("01_entrenar_modelos_qroparts.R")
```

Esto genera:

- `qroparts_model_bundle.rds`
- `metricas_modelos_qroparts.csv`
- `ranking_rutas_qroparts.csv`
- `ranking_inventario_qroparts.csv`
- `dataset_dashboard_qroparts.csv`
- graficas PNG de apoyo

4. Despues corre la app:

```r
shiny::runApp()
```

## Nota importante

El dashboard no entrena modelos cada vez que abre. Primero se ejecuta el script de entrenamiento, se guarda `qroparts_model_bundle.rds`, y despues Shiny solo lee ese archivo para mostrar resultados y hacer simulaciones.
