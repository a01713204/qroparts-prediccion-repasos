# QroParts - Entrenamiento de modelos ML logisticos
# Modelos solicitados:
# 1) Tiempo de entrega: Random Forest / XGBoost clasificador
# 2) Costos de transporte: Random Forest / XGBoost clasificador
# 3) Inventario: Arbol de decision
# 4) Rutas: Random Forest



# -----------------------------
# 0. Paquetes necesarios
# ------------------------
paquetes <- c(
  "DBI", "RSQLite", "dplyr", "readr", "ggplot2", "tidyr", "forcats",
  "rpart", "rpart.plot", "randomForest", "xgboost", "caret", "scales"
)

instalar <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if(length(instalar) > 0){
  install.packages(instalar)
}

library(DBI)
library(RSQLite)
library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(forcats)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(caret)
library(scales)

set.seed(123)

# 1. Cargar datos
# Coloca en la misma carpeta uno de estos archivos:
# - QroParts.sqlite
# - dataset_logistico_qroparts.csv
#
# Si existe SQLite se usa primero y tambien se exporta el CSV.
# El CSV queda como respaldo para volver a correr el proyecto sin SQLite.

sqlite_path <- "QroParts.sqlite"
csv_path    <- "dataset_logistico_qroparts.csv"

if(file.exists(sqlite_path)){
  cat("Cargando datos desde SQLite...\n")
  con <- dbConnect(SQLite(), sqlite_path)

  # Dataset integrado para ML.
  # Evitamos usar como predictores variables que se conocen despues del pedido
  # cuando esas variables son la respuesta del modelo.
  sql_dataset <- "
  SELECT
      p.pedido_id,
      p.fecha_pedido,
      p.fecha_entrega,

      CAST(strftime('%Y', p.fecha_pedido) AS INTEGER) AS anio_pedido,
      CAST(strftime('%m', p.fecha_pedido) AS INTEGER) AS mes_pedido,
      CAST(strftime('%w', p.fecha_pedido) AS INTEGER) AS dia_semana_pedido,

      ROUND(julianday(p.fecha_entrega) - julianday(p.fecha_pedido), 2) AS dias_entrega,
      CASE WHEN p.retraso_dias > 0 THEN 1 ELSE 0 END AS pedido_retrasado,
      p.retraso_dias,

      p.km_recorridos,
      p.costo_transporte,
      ROUND(p.costo_transporte / NULLIF(p.km_recorridos, 0), 2) AS costo_por_km,

      c.municipio,
      c.tipo_cliente,

      a.id_autoparte,
      a.categoria,
      a.costo_unitario,
      a.stock_actual,
      a.stock_minimo,
      a.stock_actual - a.stock_minimo AS margen_stock,
      ROUND(CAST(a.stock_actual AS REAL) / NULLIF(a.stock_minimo, 0), 2) AS ratio_stock,

      pr.pais_proveedor,
      r.ruta

  FROM pedido p
  JOIN cliente c ON p.cliente_id = c.cliente_id
  JOIN autoparte a ON p.id_autoparte = a.id_autoparte
  JOIN proveedor pr ON p.proveedor_id = pr.proveedor_id
  JOIN ruta r ON p.ruta = r.ruta;
  "

  df <- dbGetQuery(con, sql_dataset)
  dbDisconnect(con)

  write_csv(df, csv_path)
  cat("CSV exportado como: ", csv_path, "\n", sep = "")

} else if(file.exists(csv_path)){
  cat("Cargando datos desde CSV...\n")
  df <- read_csv(csv_path, show_col_types = FALSE)

} else {
  stop("No se encontro QroParts.sqlite ni dataset_logistico_qroparts.csv en la carpeta actual.")
}

cat("Filas y columnas cargadas:\n")
print(dim(df))

# 2. Preparacion de variables objetivo

crear_clase_terciles <- function(x, etiquetas){
  # Convierte una variable numerica en tres clases balanceadas.
  # Para tiempo: Rapido / Medio / Lento
  # Para costo: Bajo / Medio / Alto
  x_num <- as.numeric(x)
  nt <- dplyr::ntile(x_num, 3)

  clase <- dplyr::case_when(
    nt == 1 ~ etiquetas[1],
    nt == 2 ~ etiquetas[2],
    TRUE    ~ etiquetas[3]
  )

  factor(clase, levels = etiquetas)
}

# Si no existe dias_entrega, se intenta aproximar con retraso_dias.
if(!("dias_entrega" %in% names(df)) || all(is.na(df$dias_entrega))){
  if("retraso_dias" %in% names(df)){
    df$dias_entrega <- pmax(1, 3 + as.numeric(df$retraso_dias))
    cat("Aviso: dias_entrega no existia. Se aproximo como 3 + retraso_dias.\n")
  } else {
    stop("Para el modelo de tiempo se necesita dias_entrega, fecha_entrega o retraso_dias.")
  }
}

# Variable binaria para rutas si no venia lista.
if(!("pedido_retrasado" %in% names(df))){
  if("retraso_dias" %in% names(df)){
    df$pedido_retrasado <- ifelse(as.numeric(df$retraso_dias) > 0, 1, 0)
  } else {
    # Ultimo recurso: considerar retraso si el tiempo esta en el tercio mas lento.
    q_lento <- quantile(df$dias_entrega, probs = 2/3, na.rm = TRUE)
    df$pedido_retrasado <- ifelse(df$dias_entrega >= q_lento, 1, 0)
  }
}

# Variables derivadas de stock.
if(!("stock_actual" %in% names(df)) || !("stock_minimo" %in% names(df))){
  stop("Para el modelo de inventario se necesitan stock_actual y stock_minimo.")
}

df <- df %>%
  mutate(
    stock_actual = as.numeric(stock_actual),
    stock_minimo = as.numeric(stock_minimo),
    margen_stock = stock_actual - stock_minimo,
    ratio_stock = ifelse(stock_minimo <= 0 | is.na(stock_minimo), NA, stock_actual / stock_minimo),
    stock_bajo = ifelse(stock_actual <= stock_minimo, "Si", "No"),

    # Objetivo 1: tiempo de entrega clasificado.
    clase_tiempo_entrega = crear_clase_terciles(dias_entrega, c("Rapido", "Medio", "Lento")),

    # Objetivo 2: costo de transporte clasificado.
    clase_costo_transporte = crear_clase_terciles(costo_transporte, c("Bajo", "Medio", "Alto")),

    # Objetivo 3: prioridad de inventario.
    # Se basa en stock minimo y exceso de inventario.
    prioridad_inventario = case_when(
      stock_actual <= stock_minimo ~ "Falta stock",
      ratio_stock <= 1.50 ~ "Atencion",
      ratio_stock >= 3.00 ~ "Sobre stock",
      TRUE ~ "Normal"
    ),

    # Objetivo 4: riesgo de retraso para rutas.
    riesgo_retraso_ruta = ifelse(pedido_retrasado %in% c(1, "1", "Si", "SI", "si", TRUE), "Si", "No")
  )

df$prioridad_inventario <- factor(
  df$prioridad_inventario,
  levels = c("Falta stock", "Atencion", "Normal", "Sobre stock")
)

df$riesgo_retraso_ruta <- factor(df$riesgo_retraso_ruta, levels = c("No", "Si"))

# 3. Limpieza de tipos
# Variables base disponibles antes de tomar decisiones operativas.
predictores_base <- c(
  "anio_pedido",
  "mes_pedido",
  "dia_semana_pedido",
  "km_recorridos",
  "municipio",
  "tipo_cliente",
  "categoria",
  "costo_unitario",
  "stock_actual",
  "stock_minimo",
  "margen_stock",
  "ratio_stock",
  "stock_bajo",
  "pais_proveedor",
  "ruta"
)

# Para costo NO usamos costo_transporte ni costo_por_km como predictores,
# porque costo_transporte es justamente la variable que se clasifica.
predictores_tiempo <- intersect(predictores_base, names(df))
predictores_costo  <- intersect(predictores_base, names(df))
predictores_rutas  <- intersect(predictores_base, names(df))

# Para inventario se dejan variables de stock porque la prioridad depende
# directamente de stock minimo, stock actual y exceso.
predictores_inventario <- intersect(
  c("categoria", "costo_unitario", "stock_actual", "stock_minimo", "margen_stock", "ratio_stock", "stock_bajo"),
  names(df)
)

variables_categoricas_global <- intersect(
  c("municipio", "tipo_cliente", "categoria", "stock_bajo", "pais_proveedor", "ruta"),
  names(df)
)

for(v in variables_categoricas_global){
  df[[v]] <- as.factor(df[[v]])
}

# Numericas que deben estar como numericas.
variables_numericas <- intersect(
  c("anio_pedido", "mes_pedido", "dia_semana_pedido", "km_recorridos", "costo_transporte",
    "costo_por_km", "dias_entrega", "costo_unitario", "stock_actual", "stock_minimo",
    "margen_stock", "ratio_stock"),
  names(df)
)

for(v in variables_numericas){
  df[[v]] <- as.numeric(df[[v]])
}

# Dataset para la app.
columnas_app <- unique(c(
  "pedido_id", "id_autoparte", "fecha_pedido", "fecha_entrega",
  "dias_entrega", "costo_transporte", "pedido_retrasado",
  "clase_tiempo_entrega", "clase_costo_transporte",
  "prioridad_inventario", "riesgo_retraso_ruta",
  predictores_tiempo, predictores_costo, predictores_inventario, predictores_rutas
))

dataset_app <- df %>%
  select(any_of(columnas_app)) %>%
  filter(!is.na(clase_tiempo_entrega),
         !is.na(clase_costo_transporte),
         !is.na(prioridad_inventario),
         !is.na(riesgo_retraso_ruta))

cat("\nDistribuciones de objetivos:\n")
print(table(dataset_app$clase_tiempo_entrega))
print(table(dataset_app$clase_costo_transporte))
print(table(dataset_app$prioridad_inventario))
print(table(dataset_app$riesgo_retraso_ruta))
# 4. Funciones de entrenamiento y evaluacion

preparar_modelo_df <- function(data, target, predictores){
  tmp <- data %>%
    select(all_of(target), all_of(predictores)) %>%
    na.omit()

  tmp[[target]] <- as.factor(tmp[[target]])

  for(v in predictores){
    if(is.character(tmp[[v]])){
      tmp[[v]] <- as.factor(tmp[[v]])
    }
  }

  tmp
}

split_estratificado <- function(data, target, prop = 0.75){
  idx <- unlist(tapply(seq_len(nrow(data)), data[[target]], function(i){
    n_train <- max(
      1,
      min(
        floor(prop * length(i)),
        length(i) - 1
      )
    )
    sample(i, size = n_train)
  }))

  list(
    train = data[idx, , drop = FALSE],
    test  = data[-idx, , drop = FALSE]
  )
}

evaluar_clasificador <- function(real, pred, modelo, objetivo){
  real <- factor(real, levels = levels(real))
  pred <- factor(pred, levels = levels(real))

  cm <- caret::confusionMatrix(pred, real)

  by_class <- cm$byClass
  if(is.null(dim(by_class))){
    precision <- unname(by_class["Precision"])
    recall    <- unname(by_class["Recall"])
    f1        <- unname(by_class["F1"])
  } else {
    precision <- mean(by_class[, "Precision"], na.rm = TRUE)
    recall    <- mean(by_class[, "Recall"], na.rm = TRUE)
    f1        <- mean(by_class[, "F1"], na.rm = TRUE)
  }

  data.frame(
    objetivo = objetivo,
    modelo = modelo,
    accuracy = round(unname(cm$overall["Accuracy"]), 4),
    precision_macro = round(precision, 4),
    recall_macro = round(recall, 4),
    f1_macro = round(f1, 4),
    stringsAsFactors = FALSE
  )
}

confusion_df <- function(real, pred){
  real <- as.factor(real)
  pred <- factor(pred, levels = levels(real))
  as.data.frame(table(Predicho = pred, Real = real))
}

importancia_rf <- function(modelo){
  imp <- randomForest::importance(modelo)

  if("MeanDecreaseGini" %in% colnames(imp)){
    valor <- imp[, "MeanDecreaseGini"]
  } else {
    valor <- imp[, 1]
  }

  data.frame(
    variable = rownames(imp),
    importancia = as.numeric(valor),
    row.names = NULL
  ) %>%
    arrange(desc(importancia))
}

importancia_arbol <- function(modelo){
  vi <- modelo$variable.importance
  if(is.null(vi)){
    return(data.frame(variable = character(), importancia = numeric()))
  }

  data.frame(
    variable = names(vi),
    importancia = as.numeric(vi),
    row.names = NULL
  ) %>%
    arrange(desc(importancia))
}

entrenar_rf <- function(data, target, predictores, objetivo, ntree = 500){
  datos <- preparar_modelo_df(data, target, predictores)
  partes <- split_estratificado(datos, target)

  formula_modelo <- as.formula(paste(target, "~", paste(predictores, collapse = " + ")))

  modelo <- randomForest(
    formula_modelo,
    data = partes$train,
    ntree = ntree,
    importance = TRUE
  )

  pred <- predict(modelo, newdata = partes$test, type = "class")

  list(
    tipo = "randomForest",
    nombre = "Random Forest",
    objetivo = objetivo,
    target = target,
    predictores = predictores,
    niveles = levels(datos[[target]]),
    modelo = modelo,
    metricas = evaluar_clasificador(partes$test[[target]], pred, "Random Forest", objetivo),
    confusion = confusion_df(partes$test[[target]], pred),
    importancia = importancia_rf(modelo),
    train = partes$train,
    test = partes$test
  )
}

entrenar_arbol <- function(data, target, predictores, objetivo){
  datos <- preparar_modelo_df(data, target, predictores)
  partes <- split_estratificado(datos, target)

  formula_modelo <- as.formula(paste(target, "~", paste(predictores, collapse = " + ")))

  modelo <- rpart(
    formula_modelo,
    data = partes$train,
    method = "class",
    control = rpart.control(cp = 0.005, maxdepth = 5, minbucket = 20)
  )

  pred <- predict(modelo, newdata = partes$test, type = "class")

  list(
    tipo = "rpart",
    nombre = "Arbol de decision",
    objetivo = objetivo,
    target = target,
    predictores = predictores,
    niveles = levels(datos[[target]]),
    modelo = modelo,
    metricas = evaluar_clasificador(partes$test[[target]], pred, "Arbol de decision", objetivo),
    confusion = confusion_df(partes$test[[target]], pred),
    importancia = importancia_arbol(modelo),
    train = partes$train,
    test = partes$test
  )
}

entrenar_xgb <- function(data, target, predictores, objetivo, nrounds = 120){
  datos <- preparar_modelo_df(data, target, predictores)
  partes <- split_estratificado(datos, target)

  niveles <- levels(datos[[target]])
  n_clases <- length(niveles)

  formula_x <- reformulate(predictores)

  x_train <- model.matrix(formula_x, data = partes$train)[, -1, drop = FALSE]
  x_test  <- model.matrix(formula_x, data = partes$test)[, -1, drop = FALSE]

  # Alinear columnas por si alguna categoria no aparece en test.
  faltantes_test <- setdiff(colnames(x_train), colnames(x_test))
  if(length(faltantes_test) > 0){
    for(col in faltantes_test){
      x_test <- cbind(x_test, rep(0, nrow(x_test)))
      colnames(x_test)[ncol(x_test)] <- col
    }
  }
  x_test <- x_test[, colnames(x_train), drop = FALSE]

  y_train <- as.integer(partes$train[[target]]) - 1

  if(n_clases == 2){
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      max_depth = 4,
      eta = 0.08,
      subsample = 0.85,
      colsample_bytree = 0.85
    )
  } else {
    params <- list(
      objective = "multi:softprob",
      eval_metric = "mlogloss",
      num_class = n_clases,
      max_depth = 4,
      eta = 0.08,
      subsample = 0.85,
      colsample_bytree = 0.85
    )
  }

  dtrain <- xgb.DMatrix(data = x_train, label = y_train)


  modelo <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
 )

  raw_pred <- predict(modelo, xgb.DMatrix(x_test))

  if(n_clases == 2){
    pred <- ifelse(raw_pred >= 0.50, niveles[2], niveles[1])
  } else {
    prob_mat <- matrix(raw_pred, ncol = n_clases, byrow = TRUE)
    pred <- niveles[max.col(prob_mat, ties.method = "first")]
  }

  pred <- factor(pred, levels = niveles)

  imp <- tryCatch({
    xgb.importance(feature_names = colnames(x_train), model = modelo) %>%
      as.data.frame() %>%
      transmute(variable = Feature, importancia = Gain) %>%
      arrange(desc(importancia))
  }, error = function(e){
    data.frame(variable = character(), importancia = numeric())
  })

  list(
    tipo = "xgboost",
    nombre = "XGBoost",
    objetivo = objetivo,
    target = target,
    predictores = predictores,
    niveles = niveles,
    modelo = modelo,
    xgb_formula = formula_x,
    xgb_cols = colnames(x_train),
    metricas = evaluar_clasificador(partes$test[[target]], pred, "XGBoost", objetivo),
    confusion = confusion_df(partes$test[[target]], pred),
    importancia = imp,
    train = partes$train,
    test = partes$test
  )
}

seleccionar_mejor <- function(lista_modelos){
  metricas <- bind_rows(lapply(lista_modelos, function(x) x$metricas)) %>%
    arrange(desc(f1_macro), desc(accuracy))

  mejor_nombre <- metricas$modelo[1]
  mejor <- lista_modelos[[which(sapply(lista_modelos, function(x) x$nombre) == mejor_nombre)[1]]]

  list(
    mejor = mejor,
    modelos = lista_modelos,
    metricas_comparativas = metricas
  )
}

alinear_factores_nuevo <- function(nuevo, niveles_factor){
  for(v in names(niveles_factor)){
    if(v %in% names(nuevo)){
      nuevo[[v]] <- factor(as.character(nuevo[[v]]), levels = niveles_factor[[v]])
    }
  }
  nuevo
}

predecir_modelo <- function(modelo_info, nuevo, niveles_factor){
  nuevo <- alinear_factores_nuevo(nuevo, niveles_factor)
  nuevo <- nuevo[, modelo_info$predictores, drop = FALSE]

  if(modelo_info$tipo %in% c("randomForest", "rpart")){
    probs <- predict(modelo_info$modelo, newdata = nuevo, type = "prob")
    if(is.null(dim(probs))){
      probs <- matrix(probs, ncol = length(modelo_info$niveles))
      colnames(probs) <- modelo_info$niveles
    }
    pred <- colnames(probs)[max.col(probs, ties.method = "first")]
    return(list(pred = factor(pred, levels = modelo_info$niveles), probs = as.data.frame(probs)))
  }

  if(modelo_info$tipo == "xgboost"){
    x_new <- model.matrix(modelo_info$xgb_formula, data = nuevo)[, -1, drop = FALSE]

    faltantes <- setdiff(modelo_info$xgb_cols, colnames(x_new))
    if(length(faltantes) > 0){
      for(col in faltantes){
        x_new <- cbind(x_new, rep(0, nrow(x_new)))
        colnames(x_new)[ncol(x_new)] <- col
      }
    }

    sobrantes <- setdiff(colnames(x_new), modelo_info$xgb_cols)
    if(length(sobrantes) > 0){
      x_new <- x_new[, setdiff(colnames(x_new), sobrantes), drop = FALSE]
    }

    x_new <- x_new[, modelo_info$xgb_cols, drop = FALSE]
    raw <- predict(modelo_info$modelo, xgb.DMatrix(x_new))

    niveles <- modelo_info$niveles
    n_clases <- length(niveles)

    
    if(n_clases == 2){
      probs <- data.frame(
        temp_no = 1 - raw,
        temp_si = raw
      )
      colnames(probs) <- niveles
    } else {
      probs <- as.data.frame(matrix(raw, ncol = n_clases, byrow = TRUE))
      colnames(probs) <- niveles
    }

    pred <- colnames(probs)[max.col(probs, ties.method = "first")]
    return(list(pred = factor(pred, levels = niveles), probs = probs))
  }

  stop("Tipo de modelo no reconocido.")
}
# 5. Entrenamiento de modelos
cat("\nEntrenando modelo de tiempo de entrega...\n")
modelo_tiempo_rf <- entrenar_rf(
  dataset_app,
  target = "clase_tiempo_entrega",
  predictores = predictores_tiempo,
  objetivo = "Tiempo de entrega"
)


tiempo <- list(
  mejor = modelo_tiempo_rf,
  modelos = list(modelo_tiempo_rf),
  metricas_comparativas = modelo_tiempo_rf$metricas
)

cat("\nEntrenando modelo de costos de transporte...\n")
modelo_costo_rf <- entrenar_rf(
  dataset_app,
  target = "clase_costo_transporte",
  predictores = predictores_costo,
  objetivo = "Costos de transporte"
)


costos <- list(
  mejor = modelo_costo_rf,
  modelos = list(modelo_costo_rf),
  metricas_comparativas = modelo_costo_rf$metricas
)

cat("\nEntrenando modelo de inventario...\n")
inventario <- entrenar_arbol(
  dataset_app,
  target = "prioridad_inventario",
  predictores = predictores_inventario,
  objetivo = "Inventario"
)

cat("\nEntrenando modelo de rutas...\n")
rutas <- entrenar_rf(
  dataset_app,
  target = "riesgo_retraso_ruta",
  predictores = predictores_rutas,
  objetivo = "Rutas"
)
# 6. Predicciones para dashboard
niveles_factor <- lapply(df[variables_categoricas_global], levels)

pred_tiempo <- predecir_modelo(tiempo$mejor, dataset_app, niveles_factor)
pred_costo  <- predecir_modelo(costos$mejor, dataset_app, niveles_factor)
pred_inv    <- predecir_modelo(inventario, dataset_app, niveles_factor)
pred_rutas  <- predecir_modelo(rutas, dataset_app, niveles_factor)

dataset_dashboard <- dataset_app %>%
  mutate(
    pred_tiempo_entrega = as.character(pred_tiempo$pred),
    pred_costo_transporte = as.character(pred_costo$pred),
    pred_prioridad_inventario = as.character(pred_inv$pred),
    pred_riesgo_ruta = as.character(pred_rutas$pred)
  )

# Probabilidades principales para ranking.
if("Lento" %in% colnames(pred_tiempo$probs)){
  dataset_dashboard$prob_entrega_lenta <- pred_tiempo$probs[["Lento"]]
}
if("Alto" %in% colnames(pred_costo$probs)){
  dataset_dashboard$prob_costo_alto <- pred_costo$probs[["Alto"]]
}
if("Falta stock" %in% colnames(pred_inv$probs)){
  dataset_dashboard$prob_falta_stock <- pred_inv$probs[["Falta stock"]]
}
if("Sobre stock" %in% colnames(pred_inv$probs)){
  dataset_dashboard$prob_sobre_stock <- pred_inv$probs[["Sobre stock"]]
}
if("Si" %in% colnames(pred_rutas$probs)){
  dataset_dashboard$prob_riesgo_ruta <- pred_rutas$probs[["Si"]]
}

ranking_rutas <- dataset_dashboard %>%
  group_by(ruta) %>%
  summarise(
    pedidos = n(),
    tasa_retraso_observada = mean(riesgo_retraso_ruta == "Si", na.rm = TRUE),
    prob_riesgo_promedio = mean(prob_riesgo_ruta, na.rm = TRUE),
    costo_promedio = mean(costo_transporte, na.rm = TRUE),
    km_promedio = mean(km_recorridos, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(prob_riesgo_promedio), desc(tasa_retraso_observada))

ranking_inventario <- dataset_dashboard %>%
  select(any_of(c(
    "id_autoparte", "categoria", "stock_actual", "stock_minimo",
    "margen_stock", "ratio_stock", "prioridad_inventario",
    "pred_prioridad_inventario", "prob_falta_stock", "prob_sobre_stock"
  ))) %>%
  distinct() %>%
  mutate(
    severidad_inventario = case_when(
      pred_prioridad_inventario == "Falta stock" ~ 4,
      pred_prioridad_inventario == "Atencion" ~ 3,
      pred_prioridad_inventario == "Sobre stock" ~ 2,
      TRUE ~ 1
    )
  ) %>%
  arrange(desc(severidad_inventario), margen_stock)

metricas_globales <- bind_rows(
  tiempo$metricas_comparativas,
  costos$metricas_comparativas,
  inventario$metricas,
  rutas$metricas
)

write_csv(metricas_globales, "metricas_modelos_qroparts.csv")
write_csv(ranking_rutas, "ranking_rutas_qroparts.csv")
write_csv(ranking_inventario, "ranking_inventario_qroparts.csv")
write_csv(dataset_dashboard, "dataset_dashboard_qroparts.csv")

# Graficas estaticas para reporte si se necesitan.
p_metricas <- ggplot(metricas_globales, aes(x = modelo, y = accuracy, fill = objetivo)) +
  geom_col(position = "dodge") +
  labs(
    title = "QroParts - Accuracy por modelo",
    x = "Modelo",
    y = "Accuracy"
  ) +
  theme_minimal(base_size = 13)

ggsave("metricas_modelos_qroparts.png", p_metricas, width = 9, height = 5)

p_rutas <- ggplot(head(ranking_rutas, 12), aes(x = reorder(ruta, prob_riesgo_promedio), y = prob_riesgo_promedio)) +
  geom_col(fill = "#0B3D5C") +
  coord_flip() +
  scale_y_continuous(labels = percent) +
  labs(
    title = "QroParts - Ranking de riesgo por ruta",
    x = "Ruta",
    y = "Probabilidad promedio de riesgo"
  ) +
  theme_minimal(base_size = 13)

ggsave("ranking_rutas_qroparts.png", p_rutas, width = 9, height = 5)

png("arbol_inventario_qroparts.png", width = 1300, height = 850)
rpart.plot(inventario$modelo, main = "QroParts - Arbol de decision para prioridad de inventario")
dev.off()
# 7. Guardar bundle para Shiny

rangos_numericos <- dataset_dashboard %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), list(min = ~min(.x, na.rm = TRUE),
                                      max = ~max(.x, na.rm = TRUE),
                                      mediana = ~median(.x, na.rm = TRUE))))

bundle <- list(
  empresa = "QroParts",
  objetivo = "Modelos predictivos para tiempo de entrega, costos, inventario y rutas",
  fecha_entrenamiento = as.character(Sys.Date()),

  dataset = dataset_dashboard,
  metricas_globales = metricas_globales,

  tiempo = tiempo,
  costos = costos,
  inventario = inventario,
  rutas = rutas,

  predictores_tiempo = predictores_tiempo,
  predictores_costo = predictores_costo,
  predictores_inventario = predictores_inventario,
  predictores_rutas = predictores_rutas,

  variables_categoricas_global = variables_categoricas_global,
  niveles_factor = niveles_factor,
  rangos_numericos = rangos_numericos,

  ranking_rutas = ranking_rutas,
  ranking_inventario = ranking_inventario
)

saveRDS(bundle, "qroparts_model_bundle.rds")

cat("\nEntrenamiento terminado.\n")
cat("Modelo elegido para tiempo de entrega: ", tiempo$mejor$nombre, "\n", sep = "")
cat("Modelo elegido para costos: ", costos$mejor$nombre, "\n", sep = "")
cat("Modelo de inventario: Arbol de decision\n")
cat("Modelo de rutas: Random Forest\n")

cat("\nArchivos generados:\n")
cat("- qroparts_model_bundle.rds\n")
cat("- metricas_modelos_qroparts.csv\n")
cat("- ranking_rutas_qroparts.csv\n")
cat("- ranking_inventario_qroparts.csv\n")
cat("- dataset_dashboard_qroparts.csv\n")
cat("- metricas_modelos_qroparts.png\n")
cat("- ranking_rutas_qroparts.png\n")
cat("- arbol_inventario_qroparts.png\n")
