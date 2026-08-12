# =============================================================
# QroParts - Dashboard Shiny de modelos logisticos predictivos
# Para correr:
# 1) Ejecuta: source("01_entrenar_modelos_qroparts.R")
# 2) Ejecuta esta app con: shiny::runApp()
# ==========================================

paquetes <- c(
  "shiny", "bslib", "dplyr", "ggplot2", "plotly", "DT", "scales",
  "rpart.plot", "randomForest", "xgboost", "tidyr"
)

instalar <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if(length(instalar) > 0){
  install.packages(instalar)
}

library(shiny)
library(bslib)
library(dplyr)
library(ggplot2)
library(plotly)
library(DT)
library(scales)
library(rpart.plot)
library(randomForest)
library(tidyr)


# 1. Cargar bundle entrenado

modelo_path <- "qroparts_model_bundle.rds"

if(!file.exists(modelo_path)){
  stop("No se encontro qroparts_model_bundle.rds. Primero ejecuta 01_entrenar_modelos_qroparts.R")
}

bundle <- readRDS(modelo_path)
datos <- bundle$dataset

# 2. Funciones auxiliares

choice_todos <- function(x){
  c("Todos", sort(unique(as.character(x))))
}

kpi_box <- function(titulo, valor, subtitulo = NULL){
  div(
    class = "kpi-card",
    div(class = "kpi-title", titulo),
    div(class = "kpi-value", valor),
    if(!is.null(subtitulo)) div(class = "kpi-subtitle", subtitulo)
  )
}

metricas_objetivo <- function(objetivo){
  bundle$metricas_globales %>%
    filter(.data$objetivo == objetivo)
}

matriz_confusion <- function(modelo_info){
  modelo_info$confusion %>%
    tidyr::pivot_wider(names_from = Real, values_from = Freq, values_fill = 0)
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
    probs <- as.data.frame(probs)
    pred <- colnames(probs)[max.col(probs, ties.method = "first")]
    return(list(pred = pred, probs = probs))
  }

  
  stop("Tipo de modelo no reconocido.")
}

prob_maxima <- function(pred_result){
  max(as.numeric(pred_result$probs[1, ]), na.rm = TRUE)
}

texto_prob <- function(pred_result){
  percent(prob_maxima(pred_result), accuracy = 0.1)
}

clase_badge <- function(valor){
  valor <- as.character(valor)
  if(valor %in% c("Lento", "Alto", "Falta stock", "Si")){
    return("badge-alto")
  }
  if(valor %in% c("Medio", "Atencion", "Sobre stock")){
    return("badge-medio")
  }
  "badge-bajo"
}

crear_nuevo_pedido <- function(input){
  data.frame(
    anio_pedido = as.integer(input$sim_anio),
    mes_pedido = as.integer(input$sim_mes),
    dia_semana_pedido = as.integer(input$sim_dia),
    km_recorridos = as.numeric(input$sim_km),
    municipio = input$sim_municipio,
    tipo_cliente = input$sim_tipo_cliente,
    categoria = input$sim_categoria,
    costo_unitario = as.numeric(input$sim_costo_unitario),
    stock_actual = as.numeric(input$sim_stock_actual),
    stock_minimo = as.numeric(input$sim_stock_minimo),
    margen_stock = as.numeric(input$sim_stock_actual) - as.numeric(input$sim_stock_minimo),
    ratio_stock = as.numeric(input$sim_stock_actual) / max(as.numeric(input$sim_stock_minimo), 1),
    stock_bajo = ifelse(as.numeric(input$sim_stock_actual) <= as.numeric(input$sim_stock_minimo), "Si", "No"),
    pais_proveedor = input$sim_pais_proveedor,
    ruta = input$sim_ruta,
    stringsAsFactors = FALSE
  )
}

recomendacion_simulador <- function(tiempo, costo, inventario, ruta){
  recs <- c()

  if(tiempo == "Lento"){
    recs <- c(recs, "Revisar promesa de entrega y dar seguimiento antes del despacho.")
  }
  if(costo == "Alto"){
    recs <- c(recs, "Validar si la ruta o el proveedor puede optimizar el costo de transporte.")
  }
  if(inventario %in% c("Falta stock", "Atencion")){
    recs <- c(recs, "Priorizar compra o reposicion de inventario para evitar faltantes.")
  }
  if(inventario == "Sobre stock"){
    recs <- c(recs, "Revisar exceso de inventario para liberar capital y espacio.")
  }
  if(ruta == "Si"){
    recs <- c(recs, "Asignar monitoreo operativo a la ruta porque tiene riesgo de retraso.")
  }

  if(length(recs) == 0){
    recs <- "Operacion normal. No se detectan alertas fuertes con los datos ingresados."
  }

  tags$ul(lapply(recs, tags$li))
}
# 3. Interfaz

ui <- navbarPage(
  title = div(
    class = "brand-title",
    span("QroParts"),
    tags$small(" | Dashboard ML logistico")
  ),
  theme = bs_theme(version = 5, bootswatch = "flatly", primary = "#0B3D5C"),

  header = tags$head(
    tags$style(HTML("
      body { background-color: #F5F7FA; }
      .navbar-brand { font-weight: 800; letter-spacing: .3px; }
      .brand-title small { font-weight: 400; margin-left: 6px; color: #E8EEF3; }
      .qro-hero {
        background: linear-gradient(135deg, #0B3D5C, #126782);
        color: white;
        padding: 24px;
        border-radius: 18px;
        margin: 18px 0 18px 0;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
      }
      .qro-hero h2 { margin-top: 0; font-weight: 800; }
      .qro-hero p { margin-bottom: 0; font-size: 16px; }
      .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,.08);
        margin-bottom: 15px;
        border-left: 6px solid #0B3D5C;
        min-height: 125px;
      }
      .kpi-title { color: #6B7280; font-size: 13px; text-transform: uppercase; letter-spacing: .5px; }
      .kpi-value { color: #0B3D5C; font-size: 30px; font-weight: 800; margin-top: 8px; }
      .kpi-subtitle { color: #6B7280; font-size: 13px; margin-top: 4px; }
      .box-card {
        background: white;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,.08);
        margin-bottom: 18px;
      }
      .section-title { color: #0B3D5C; font-weight: 800; margin-top: 0; }
      .badge-result {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 13px;
        color: white;
        font-weight: 800;
        margin-top: 6px;
      }
      .badge-alto { background-color: #B91C1C; }
      .badge-medio { background-color: #B45309; }
      .badge-bajo { background-color: #047857; }
      .small-note { color: #6B7280; font-size: 13px; }
    "))
  ),

  tabPanel(
    "Dashboard",
    fluidPage(
      div(
        class = "qro-hero",
        h2("Resumen ejecutivo para QroParts"),
        p("Vista general de predicciones para entregas, costos, inventario y rutas. Los modelos se entrenan una vez y Shiny solo presenta resultados y simulaciones.")
      ),
      fluidRow(
        column(3, uiOutput("kpi_pedidos")),
        column(3, uiOutput("kpi_entrega_lenta")),
        column(3, uiOutput("kpi_costo_alto")),
        column(3, uiOutput("kpi_stock_critico"))
      ),
      fluidRow(
        column(6, div(class = "box-card", h4(class = "section-title", "Ranking de rutas con mayor riesgo"), plotlyOutput("plot_dash_rutas", height = "360px"))),
        column(6, div(class = "box-card", h4(class = "section-title", "Distribucion de alertas de inventario"), plotlyOutput("plot_dash_inventario", height = "360px")))
      ),
      fluidRow(
        column(6, div(class = "box-card", h4(class = "section-title", "Modelos usados"), DTOutput("tabla_modelos_dashboard"))),
        column(6, div(class = "box-card", h4(class = "section-title", "Lectura ejecutiva"), uiOutput("lectura_dashboard")))
      )
    )
  ),

  tabPanel(
    "Tiempo de entrega",
    sidebarLayout(
      sidebarPanel(
        h4("Filtros"),
        selectInput("t_ruta", "Ruta", choices = choice_todos(datos$ruta)),
        selectInput("t_municipio", "Municipio", choices = choice_todos(datos$municipio)),
        selectInput("t_categoria", "Categoria", choices = choice_todos(datos$categoria)),
        selectInput("t_pred", "Prediccion", choices = choice_todos(datos$pred_tiempo_entrega))
      ),
      mainPanel(
        fluidRow(
          column(6, div(class = "box-card", h4(class = "section-title", "Metricas del modelo Random Forest"), DTOutput("tabla_metricas_tiempo"))),
          column(6, div(class = "box-card", h4(class = "section-title", "Importancia de variables"), plotlyOutput("plot_importancia_tiempo", height = "330px")))
        ),
        fluidRow(
          column(6, div(class = "box-card", h4(class = "section-title", "Real vs predicho"), plotlyOutput("plot_tiempo_real_pred", height = "330px"))),
          column(6, div(class = "box-card", h4(class = "section-title", "Matriz de confusion del mejor modelo"), DTOutput("tabla_conf_tiempo")))
        ),
        fluidRow(
          column(12, div(class = "box-card", h4(class = "section-title", "Pedidos con entrega lenta o riesgo operativo"), DTOutput("tabla_tiempo")))
        )
      )
    )
  ),

  tabPanel(
    "Costos de transporte",
    sidebarLayout(
      sidebarPanel(
        h4("Filtros"),
        selectInput("c_ruta", "Ruta", choices = choice_todos(datos$ruta)),
        selectInput("c_municipio", "Municipio", choices = choice_todos(datos$municipio)),
        selectInput("c_categoria", "Categoria", choices = choice_todos(datos$categoria)),
        selectInput("c_pred", "Prediccion", choices = choice_todos(datos$pred_costo_transporte))
      ),
      mainPanel(
        fluidRow(
          column(6, div(class = "box-card", h4(class = "section-title", "Metricas del modelo Random Forest"), DTOutput("tabla_metricas_costo"))),
          column(6, div(class = "box-card", h4(class = "section-title", "Importancia de variables"), plotlyOutput("plot_importancia_costo", height = "330px")))
        ),
        fluidRow(
          column(6, div(class = "box-card", h4(class = "section-title", "Km recorridos vs costo"), plotlyOutput("plot_costo_km", height = "330px"))),
          column(6, div(class = "box-card", h4(class = "section-title", "Matriz de confusion del mejor modelo"), DTOutput("tabla_conf_costo")))
        ),
        fluidRow(
          column(12, div(class = "box-card", h4(class = "section-title", "Pedidos con costo alto"), DTOutput("tabla_costo")))
        )
      )
    )
  ),

  tabPanel(
    "Inventario",
    fluidPage(
      div(
        class = "qro-hero",
        h2("Prioridad de inventario"),
        p("Modelo con arbol de decision para clasificar piezas en Falta stock, Atencion, Normal o Sobre stock con base en stock minimo y exceso.")
      ),
      fluidRow(
        column(3, uiOutput("kpi_falta_stock")),
        column(3, uiOutput("kpi_atencion")),
        column(3, uiOutput("kpi_normal")),
        column(3, uiOutput("kpi_sobre_stock"))
      ),
      fluidRow(
        column(6, div(class = "box-card", h4(class = "section-title", "Distribucion de prioridad"), plotlyOutput("plot_inv_dist", height = "340px"))),
        column(6, div(class = "box-card", h4(class = "section-title", "Metricas del arbol"), DTOutput("tabla_metricas_inv")))
      ),
      fluidRow(
        column(12, div(class = "box-card", h4(class = "section-title", "Arbol de decision"), plotOutput("plot_arbol_inv", height = "600px")))
      ),
      fluidRow(
        column(12, div(class = "box-card", h4(class = "section-title", "Ranking de piezas a revisar"), DTOutput("tabla_inv")))
      )
    )
  ),

  tabPanel(
    "Rutas",
    fluidPage(
      div(
        class = "qro-hero",
        h2("Riesgo operativo por ruta"),
        p("Random Forest para estimar riesgo de retraso por ruta y priorizar seguimiento logistico.")
      ),
      fluidRow(
        column(6, div(class = "box-card", h4(class = "section-title", "Metricas del Random Forest"), DTOutput("tabla_metricas_rutas"))),
        column(6, div(class = "box-card", h4(class = "section-title", "Importancia de variables"), plotlyOutput("plot_importancia_rutas", height = "330px")))
      ),
      fluidRow(
        column(6, div(class = "box-card", h4(class = "section-title", "Riesgo promedio predicho por ruta"), plotlyOutput("plot_rutas_riesgo", height = "360px"))),
        column(6, div(class = "box-card", h4(class = "section-title", "Tasa observada vs riesgo predicho"), plotlyOutput("plot_rutas_obs_pred", height = "360px")))
      ),
      fluidRow(
        column(12, div(class = "box-card", h4(class = "section-title", "Ranking operativo de rutas"), DTOutput("tabla_rutas")))
      )
    )
  ),

  tabPanel(
    "Simulador",
    sidebarLayout(
      sidebarPanel(
        h4("Datos del pedido"),
        selectInput("sim_municipio", "Municipio", choices = sort(unique(as.character(datos$municipio)))),
        selectInput("sim_ruta", "Ruta", choices = sort(unique(as.character(datos$ruta)))),
        selectInput("sim_tipo_cliente", "Tipo de cliente", choices = sort(unique(as.character(datos$tipo_cliente)))),
        selectInput("sim_categoria", "Categoria", choices = sort(unique(as.character(datos$categoria)))),
        selectInput("sim_pais_proveedor", "Pais proveedor", choices = sort(unique(as.character(datos$pais_proveedor)))),
        hr(),
        sliderInput("sim_km", "Kilometros recorridos",
                    min = floor(min(datos$km_recorridos, na.rm = TRUE)),
                    max = ceiling(max(datos$km_recorridos, na.rm = TRUE)),
                    value = round(median(datos$km_recorridos, na.rm = TRUE))),
        sliderInput("sim_costo_unitario", "Costo unitario",
                    min = floor(min(datos$costo_unitario, na.rm = TRUE)),
                    max = ceiling(max(datos$costo_unitario, na.rm = TRUE)),
                    value = round(median(datos$costo_unitario, na.rm = TRUE))),
        sliderInput("sim_stock_actual", "Stock actual",
                    min = floor(min(datos$stock_actual, na.rm = TRUE)),
                    max = ceiling(max(datos$stock_actual, na.rm = TRUE)),
                    value = round(median(datos$stock_actual, na.rm = TRUE))),
        sliderInput("sim_stock_minimo", "Stock minimo",
                    min = floor(min(datos$stock_minimo, na.rm = TRUE)),
                    max = ceiling(max(datos$stock_minimo, na.rm = TRUE)),
                    value = round(median(datos$stock_minimo, na.rm = TRUE))),
        hr(),
        sliderInput("sim_anio", "Anio del pedido",
                    min = min(datos$anio_pedido, na.rm = TRUE),
                    max = max(datos$anio_pedido, na.rm = TRUE),
                    value = max(datos$anio_pedido, na.rm = TRUE),
                    step = 1, sep = ""),
        sliderInput("sim_mes", "Mes", min = 1, max = 12, value = 6, step = 1),
        sliderInput("sim_dia", "Dia de semana", min = 0, max = 6, value = 1, step = 1),
        actionButton("simular", "Calcular predicciones", class = "btn-primary")
      ),
      mainPanel(
        div(class = "box-card", h3(class = "section-title", "Resultado operativo"), uiOutput("resultado_simulador")),
        div(class = "box-card", h4(class = "section-title", "Recomendaciones"), uiOutput("recomendacion_simulador")),
        div(class = "box-card", h4(class = "section-title", "Registro usado por los modelos"), DTOutput("tabla_simulador"))
      )
    )
  )
)
# 4. Servidor
server <- function(input, output, session){

  # Dashboard
  output$kpi_pedidos <- renderUI({
    kpi_box("Registros analizados", comma(nrow(datos)), paste0("Entrenamiento: ", bundle$fecha_entrenamiento))
  })

  output$kpi_entrega_lenta <- renderUI({
    valor <- mean(datos$pred_tiempo_entrega == "Lento", na.rm = TRUE)
    kpi_box("Entregas lentas predichas", percent(valor, accuracy = 0.1), "Clase Lento del modelo de tiempo")
  })

  output$kpi_costo_alto <- renderUI({
    valor <- mean(datos$pred_costo_transporte == "Alto", na.rm = TRUE)
    kpi_box("Costos altos predichos", percent(valor, accuracy = 0.1), "Clase Alto del modelo de costos")
  })

  output$kpi_stock_critico <- renderUI({
    valor <- mean(datos$pred_prioridad_inventario %in% c("Falta stock", "Atencion"), na.rm = TRUE)
    kpi_box("Inventario a revisar", percent(valor, accuracy = 0.1), "Falta stock o Atencion")
  })

  output$plot_dash_rutas <- renderPlotly({
    top <- bundle$ranking_rutas %>%
      slice_head(n = 12)

    p <- ggplot(top, aes(x = reorder(ruta, prob_riesgo_promedio), y = prob_riesgo_promedio,
                         text = paste("Ruta:", ruta,
                                      "<br>Pedidos:", pedidos,
                                      "<br>Riesgo promedio:", percent(prob_riesgo_promedio, accuracy = 0.1),
                                      "<br>Tasa observada:", percent(tasa_retraso_observada, accuracy = 0.1)))) +
      geom_col(fill = "#0B3D5C") +
      coord_flip() +
      scale_y_continuous(labels = percent) +
      labs(x = "Ruta", y = "Riesgo promedio") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$plot_dash_inventario <- renderPlotly({
    res <- datos %>%
      count(pred_prioridad_inventario, name = "piezas")

    p <- ggplot(res, aes(x = reorder(pred_prioridad_inventario, piezas), y = piezas,
                         text = paste("Prioridad:", pred_prioridad_inventario, "<br>Piezas:", piezas))) +
      geom_col(fill = "#126782") +
      coord_flip() +
      labs(x = "Prioridad", y = "Piezas") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$tabla_modelos_dashboard <- renderDT({
    data.frame(
      Area = c("Tiempo de entrega", "Costos de transporte", "Inventario", "Rutas"),
      Modelo = c(bundle$tiempo$mejor$nombre, bundle$costos$mejor$nombre, "Arbol de decision", "Random Forest"),
      Objetivo = c("Rapido / Medio / Lento", "Bajo / Medio / Alto", "Falta stock / Atencion / Normal / Sobre stock", "Riesgo de retraso: No / Si")
    ) %>%
      datatable(options = list(dom = "t", scrollX = TRUE), rownames = FALSE)
  })

  output$lectura_dashboard <- renderUI({
    ruta_top <- bundle$ranking_rutas %>% slice(1)
    inv_top <- datos %>% count(pred_prioridad_inventario, sort = TRUE) %>% slice(1)
    modelo_t <- bundle$tiempo$mejor$nombre
    modelo_c <- bundle$costos$mejor$nombre

    tags$ul(
      tags$li(paste0("El mejor modelo para tiempo de entrega fue ", modelo_t, ".")),
      tags$li(paste0("El mejor modelo para costos de transporte fue ", modelo_c, ".")),
      tags$li(paste0("La ruta con mayor riesgo promedio predicho es ", ruta_top$ruta, " con ", percent(ruta_top$prob_riesgo_promedio, accuracy = 0.1), ".")),
      tags$li(paste0("La prioridad de inventario mas frecuente es ", inv_top$pred_prioridad_inventario, " con ", inv_top$n, " registros.")),
      tags$li("El simulador permite probar un pedido nuevo antes de despacharlo.")
    )
  })

  # Tiempo de entrega
  datos_tiempo <- reactive({
    df <- datos
    if(input$t_ruta != "Todos") df <- df %>% filter(ruta == input$t_ruta)
    if(input$t_municipio != "Todos") df <- df %>% filter(municipio == input$t_municipio)
    if(input$t_categoria != "Todos") df <- df %>% filter(categoria == input$t_categoria)
    if(input$t_pred != "Todos") df <- df %>% filter(pred_tiempo_entrega == input$t_pred)
    df
  })

  output$tabla_metricas_tiempo <- renderDT({
    datatable(metricas_objetivo("Tiempo de entrega"), options = list(dom = "t", scrollX = TRUE), rownames = FALSE)
  })

  output$plot_importancia_tiempo <- renderPlotly({
    imp <- bundle$tiempo$mejor$importancia %>% slice_head(n = 12)
    p <- ggplot(imp, aes(x = reorder(variable, importancia), y = importancia,
                         text = paste("Variable:", variable, "<br>Importancia:", round(importancia, 4)))) +
      geom_col(fill = "#126782") +
      coord_flip() +
      labs(x = "Variable", y = "Importancia") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$plot_tiempo_real_pred <- renderPlotly({
    df <- datos_tiempo()
    req(nrow(df) > 0)

    res <- df %>%
      count(clase_tiempo_entrega, pred_tiempo_entrega, name = "pedidos")

    p <- ggplot(res, aes(x = clase_tiempo_entrega, y = pedidos, fill = pred_tiempo_entrega,
                         text = paste("Real:", clase_tiempo_entrega,
                                      "<br>Predicho:", pred_tiempo_entrega,
                                      "<br>Pedidos:", pedidos))) +
      geom_col(position = "dodge") +
      labs(x = "Clase real", y = "Pedidos", fill = "Prediccion") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$tabla_conf_tiempo <- renderDT({
    datatable(matriz_confusion(bundle$tiempo$mejor), options = list(dom = "t"), rownames = FALSE)
  })

  output$tabla_tiempo <- renderDT({
    df <- datos_tiempo() %>%
      arrange(desc(prob_entrega_lenta)) %>%
      select(any_of(c("pedido_id", "ruta", "municipio", "categoria", "km_recorridos",
                      "dias_entrega", "clase_tiempo_entrega", "pred_tiempo_entrega",
                      "prob_entrega_lenta")))

    datatable(df, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
  })

  # Costos
  datos_costo <- reactive({
    df <- datos
    if(input$c_ruta != "Todos") df <- df %>% filter(ruta == input$c_ruta)
    if(input$c_municipio != "Todos") df <- df %>% filter(municipio == input$c_municipio)
    if(input$c_categoria != "Todos") df <- df %>% filter(categoria == input$c_categoria)
    if(input$c_pred != "Todos") df <- df %>% filter(pred_costo_transporte == input$c_pred)
    df
  })

  output$tabla_metricas_costo <- renderDT({
    datatable(metricas_objetivo("Costos de transporte"), options = list(dom = "t", scrollX = TRUE), rownames = FALSE)
  })

  output$plot_importancia_costo <- renderPlotly({
    imp <- bundle$costos$mejor$importancia %>% slice_head(n = 12)
    p <- ggplot(imp, aes(x = reorder(variable, importancia), y = importancia,
                         text = paste("Variable:", variable, "<br>Importancia:", round(importancia, 4)))) +
      geom_col(fill = "#126782") +
      coord_flip() +
      labs(x = "Variable", y = "Importancia") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$plot_costo_km <- renderPlotly({
    df <- datos_costo()
    req(nrow(df) > 0)

    p <- ggplot(df, aes(x = km_recorridos, y = costo_transporte, color = pred_costo_transporte,
                        text = paste("Pedido:", pedido_id,
                                     "<br>Ruta:", ruta,
                                     "<br>Prediccion:", pred_costo_transporte,
                                     "<br>Costo:", dollar(costo_transporte, prefix = "$")))) +
      geom_point(alpha = 0.7) +
      labs(x = "Km recorridos", y = "Costo de transporte", color = "Prediccion") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$tabla_conf_costo <- renderDT({
    datatable(matriz_confusion(bundle$costos$mejor), options = list(dom = "t"), rownames = FALSE)
  })

  output$tabla_costo <- renderDT({
    df <- datos_costo() %>%
      arrange(desc(prob_costo_alto)) %>%
      select(any_of(c("pedido_id", "ruta", "municipio", "categoria", "km_recorridos",
                      "costo_transporte", "clase_costo_transporte", "pred_costo_transporte",
                      "prob_costo_alto")))

    datatable(df, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
  })

  # Inventario
  output$kpi_falta_stock <- renderUI({
    n <- sum(datos$pred_prioridad_inventario == "Falta stock", na.rm = TRUE)
    kpi_box("Falta stock", comma(n), "Maxima prioridad")
  })

  output$kpi_atencion <- renderUI({
    n <- sum(datos$pred_prioridad_inventario == "Atencion", na.rm = TRUE)
    kpi_box("Atencion", comma(n), "Stock cercano al minimo")
  })

  output$kpi_normal <- renderUI({
    n <- sum(datos$pred_prioridad_inventario == "Normal", na.rm = TRUE)
    kpi_box("Normal", comma(n), "Sin alerta principal")
  })

  output$kpi_sobre_stock <- renderUI({
    n <- sum(datos$pred_prioridad_inventario == "Sobre stock", na.rm = TRUE)
    kpi_box("Sobre stock", comma(n), "Exceso de inventario")
  })

  output$plot_inv_dist <- renderPlotly({
    res <- datos %>%
      count(pred_prioridad_inventario, name = "registros")

    p <- ggplot(res, aes(x = reorder(pred_prioridad_inventario, registros), y = registros,
                         text = paste("Prioridad:", pred_prioridad_inventario, "<br>Registros:", registros))) +
      geom_col(fill = "#0B3D5C") +
      coord_flip() +
      labs(x = "Prioridad", y = "Registros") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$tabla_metricas_inv <- renderDT({
    datatable(bundle$inventario$metricas, options = list(dom = "t", scrollX = TRUE), rownames = FALSE)
  })

  output$plot_arbol_inv <- renderPlot({
    rpart.plot(bundle$inventario$modelo, main = "QroParts - Arbol de decision para inventario")
  })

  output$tabla_inv <- renderDT({
    df <- bundle$ranking_inventario %>%
      select(any_of(c("id_autoparte", "categoria", "stock_actual", "stock_minimo",
                      "margen_stock", "ratio_stock", "prioridad_inventario",
                      "pred_prioridad_inventario", "prob_falta_stock", "prob_sobre_stock")))

    datatable(df, options = list(pageLength = 12, scrollX = TRUE), rownames = FALSE)
  })

  # Rutas
  output$tabla_metricas_rutas <- renderDT({
    datatable(bundle$rutas$metricas, options = list(dom = "t", scrollX = TRUE), rownames = FALSE)
  })

  output$plot_importancia_rutas <- renderPlotly({
    imp <- bundle$rutas$importancia %>% slice_head(n = 12)
    p <- ggplot(imp, aes(x = reorder(variable, importancia), y = importancia,
                         text = paste("Variable:", variable, "<br>Importancia:", round(importancia, 4)))) +
      geom_col(fill = "#126782") +
      coord_flip() +
      labs(x = "Variable", y = "Importancia") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$plot_rutas_riesgo <- renderPlotly({
    top <- bundle$ranking_rutas %>%
      slice_head(n = 15)

    p <- ggplot(top, aes(x = reorder(ruta, prob_riesgo_promedio), y = prob_riesgo_promedio,
                         text = paste("Ruta:", ruta,
                                      "<br>Pedidos:", pedidos,
                                      "<br>Riesgo predicho:", percent(prob_riesgo_promedio, accuracy = 0.1)))) +
      geom_col(fill = "#0B3D5C") +
      coord_flip() +
      scale_y_continuous(labels = percent) +
      labs(x = "Ruta", y = "Riesgo promedio predicho") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$plot_rutas_obs_pred <- renderPlotly({
    rr <- bundle$ranking_rutas

    p <- ggplot(rr, aes(x = tasa_retraso_observada, y = prob_riesgo_promedio,
                        text = paste("Ruta:", ruta,
                                     "<br>Tasa observada:", percent(tasa_retraso_observada, accuracy = 0.1),
                                     "<br>Riesgo predicho:", percent(prob_riesgo_promedio, accuracy = 0.1)))) +
      geom_point(size = 3, alpha = 0.75, color = "#126782") +
      scale_x_continuous(labels = percent) +
      scale_y_continuous(labels = percent) +
      labs(x = "Tasa observada de retraso", y = "Riesgo promedio predicho") +
      theme_minimal(base_size = 13)

    ggplotly(p, tooltip = "text")
  })

  output$tabla_rutas <- renderDT({
    datatable(bundle$ranking_rutas, options = list(pageLength = 12, scrollX = TRUE), rownames = FALSE)
  })

  # Simulador
  simulacion <- eventReactive(input$simular, {
    nuevo <- crear_nuevo_pedido(input)

    pred_t <- predecir_modelo(bundle$tiempo$mejor, nuevo, bundle$niveles_factor)
    pred_c <- predecir_modelo(bundle$costos$mejor, nuevo, bundle$niveles_factor)
    pred_i <- predecir_modelo(bundle$inventario, nuevo, bundle$niveles_factor)
    pred_r <- predecir_modelo(bundle$rutas, nuevo, bundle$niveles_factor)

    list(
      nuevo = nuevo,
      tiempo = pred_t,
      costo = pred_c,
      inventario = pred_i,
      ruta = pred_r
    )
  }, ignoreInit = FALSE)

  output$resultado_simulador <- renderUI({
    sim <- simulacion()

    val_t <- sim$tiempo$pred[1]
    val_c <- sim$costo$pred[1]
    val_i <- sim$inventario$pred[1]
    val_r <- sim$ruta$pred[1]

    fluidRow(
      column(3,
             div(class = "kpi-card",
                 div(class = "kpi-title", "Tiempo de entrega"),
                 h3(val_t),
                 span(class = paste("badge-result", clase_badge(val_t)), texto_prob(sim$tiempo)))
      ),
      column(3,
             div(class = "kpi-card",
                 div(class = "kpi-title", "Costo transporte"),
                 h3(val_c),
                 span(class = paste("badge-result", clase_badge(val_c)), texto_prob(sim$costo)))
      ),
      column(3,
             div(class = "kpi-card",
                 div(class = "kpi-title", "Inventario"),
                 h3(val_i),
                 span(class = paste("badge-result", clase_badge(val_i)), texto_prob(sim$inventario)))
      ),
      column(3,
             div(class = "kpi-card",
                 div(class = "kpi-title", "Riesgo ruta"),
                 h3(ifelse(val_r == "Si", "Con riesgo", "Sin riesgo")),
                 span(class = paste("badge-result", clase_badge(val_r)), texto_prob(sim$ruta)))
      )
    )
  })

  output$recomendacion_simulador <- renderUI({
    sim <- simulacion()
    recomendacion_simulador(
      tiempo = sim$tiempo$pred[1],
      costo = sim$costo$pred[1],
      inventario = sim$inventario$pred[1],
      ruta = sim$ruta$pred[1]
    )
  })

  output$tabla_simulador <- renderDT({
    sim <- simulacion()
    datatable(sim$nuevo, options = list(dom = "t", scrollX = TRUE), rownames = FALSE)
  })
}

shinyApp(ui, server)
