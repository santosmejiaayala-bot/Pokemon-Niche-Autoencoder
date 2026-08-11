library(keras3)

py_require(
  packages = "tensorflow",
  python_version = ">=3.9,<3.11"
)
library(tidyverse)
library(caret)
library(keras3)
library(plotly)
library(cluster)
library(htmlwidgets)

pokemon <- read.csv("pokemon.csv", stringsAsFactors = FALSE)

features <- pokemon %>%
  select(
    hp,
    attack,
    defense,
    sp_attack,
    sp_defense,
    speed,
    generation,
    legendary,
    type1,
    type2
  )

features$type2[is.na(features$type2)] <- "None"
features$type2[features$type2 == ""] <- "None"

features$type1 <- as.factor(features$type1)
features$type2 <- as.factor(features$type2)

features$legendary <- as.integer(features$legendary)

dummy_encoder <-
  dummyVars(
    ~ .,
    data = features
  )

x <- predict(
  dummy_encoder,
  newdata = features
)

x_raw <- predict(
  dummy_encoder,
  newdata = features
)

bad_values <- apply(
  x_raw,
  2,
  function(z) any(!is.finite(z))
)

cat(
  "Columns containing non-finite values:",
  sum(bad_values),
  "\n"
)

if (any(bad_values)) {
  print(colnames(x_raw)[bad_values])
}

x_raw <- x_raw[, !bad_values, drop = FALSE]

zero_variance <- apply(
  x_raw,
  2,
  function(z) sd(z) == 0
)

cat(
  "Zero-variance columns:",
  sum(zero_variance),
  "\n"
)

if (any(zero_variance)) {
  print(colnames(x_raw)[zero_variance])
}

x_raw <- x_raw[, !zero_variance, drop = FALSE]

x <- scale(x_raw)
cat(
  "Invalid input values:",
  sum(!is.finite(x)),
  "\n"
)

if (any(!is.finite(x))) {
  stop("x still contains NA, NaN, or infinite values.")
}

input_dim <- ncol(x)

cat("Input Features:", input_dim, "\n")

input <-
  layer_input(
    shape = input_dim
  )

encoded <-
  input %>%
  layer_dense(
    units = 64,
    activation = "relu"
  ) %>%
  layer_dense(
    units = 32,
    activation = "relu"
  ) %>%
  layer_dense(
    units = 8,
    activation = "relu",
    name = "latent"
  )

decoded <-
  encoded %>%
  layer_dense(
    units = 32,
    activation = "relu"
  ) %>%
  layer_dense(
    units = 64,
    activation = "relu"
  ) %>%
  layer_dense(
    units = input_dim,
    activation = "linear"
  )

autoencoder <-
  keras_model(
    inputs = input,
    outputs = decoded
  )
autoencoder %>% compile(
  
  optimizer = "adam",
  
  loss = "mse"
)

callback <-
  callback_early_stopping(
    
    monitor = "val_loss",
    
    patience = 10,
    
    restore_best_weights = TRUE
  )

history <-
  autoencoder %>% fit(
    
    x,
    x,
    
    epochs = 200,
    
    batch_size = 32,
    
    validation_split = 0.20,
    
    callbacks = callback,
    
    verbose = 1
  )

encoder <-
  keras_model(
    
    inputs = input,
    
    outputs = get_layer(
      autoencoder,
      "latent"
    )$output
  )

latent <- predict(
  encoder,
  x
)

latent_sd <- apply(
  latent,
  2,
  sd
)

keep_latent <- latent_sd > 0

latent_clean <- latent[
  ,
  keep_latent,
  drop = FALSE
]

cat(
  "Latent dimensions retained:",
  ncol(latent_clean),
  "\n"
)

latent_scaled <- scale(latent_clean)

if (any(!is.finite(latent_scaled))) {
  stop("Latent scaling produced invalid values.")
}

pca <- prcomp(
  latent_scaled,
  center = FALSE,
  scale. = FALSE
)

embedding <- as.data.frame(
  pca$x[, 1:2]
)

colnames(embedding) <- c(
  "PC1",
  "PC2"
)
plot_data <-
  bind_cols(
    pokemon,
    embedding
  )
plot_data$hover <-
  paste0(
    "<b>", plot_data$name, "</b><br>",
    "Type: ", plot_data$type1,
    
    ifelse(
      plot_data$type2 == "" |
        is.na(plot_data$type2),
      "",
      paste0("/", plot_data$type2)
    ),
    
    "<br><br>",
    
    "HP: ", plot_data$hp,
    "<br>Attack: ", plot_data$attack,
    "<br>Defense: ", plot_data$defense,
    "<br>Sp. Attack: ", plot_data$sp_attack,
    "<br>Sp. Defense: ", plot_data$sp_defense,
    "<br>Speed: ", plot_data$speed,
    "<br>Generation: ", plot_data$generation,
    "<br>Legendary: ",
    ifelse(plot_data$legendary, "Yes", "No")
  )

fig <-
  
  plot_ly(
    
    data = plot_data,
    
    x = ~PC1,
    y = ~PC2,
    
    type = "scatter",
    mode = "markers",
    
    color = ~type1,
    
    text = ~hover,
    
    hoverinfo = "text",
    
    marker = list(
      size = 8,
      opacity = 0.8
    )
  )

fig <-
  
  fig %>%
  
  layout(
    
    title = "Pokémon Stat Clusters Learned by a Deep Autoencoder",
    
    xaxis =
      list(
        title = "Principal Component 1"
      ),
    
    yaxis =
      list(
        title = "Principal Component 2"
      )
  )

fig

set.seed(42)

k_values <- 2:12

silhouette_scores <- numeric(length(k_values))

for(i in seq_along(k_values)){
  
  km <- kmeans(
    latent_scaled,
    centers = k_values[i],
    nstart = 25
  )
  
  sil <- silhouette(
    km$cluster,
    dist(latent_scaled)
  )
  
  silhouette_scores[i] <- mean(sil[,3])
  
}
best_k <- k_values[which.max(silhouette_scores)]

cat("Optimal Number of Clusters:", best_k)

kmeans_fit <- kmeans(
  
  latent_scaled,
  
  centers = best_k,
  
  nstart = 50
  
)

plot_data$Cluster <- factor(kmeans_fit$cluster)

cluster_summary <-
  
  plot_data %>%
  
  group_by(Cluster) %>%
  
  summarise(
    
    Count = n(),
    
    HP = mean(hp),
    
    Attack = mean(attack),
    
    Defense = mean(defense),
    
    SpAttack = mean(sp_attack),
    
    SpDefense = mean(sp_defense),
    
    Speed = mean(speed),
    
    Legendary = mean(legendary)
    
  )

cluster_summary

assign_niche <- function(hp, atk, def, spa, spd, spe) {
  
  if (atk >= 110 & spe >= 100) {
    return("Physical Sweeper")
  }
  
  if (spa >= 110 & spe >= 100) {
    return("Special Sweeper")
  }
  
  if (def >= 110 & hp >= 90) {
    return("Physical Wall")
  }
  
  if (spd >= 110 & hp >= 90) {
    return("Special Wall")
  }
  
  if (hp >= 110 & atk >= 90 & def >= 90) {
    return("Bulky Tank")
  }
  
  if (spe >= 115) {
    return("Fast Utility")
  }
  
  return("Balanced")
}

cluster_summary$Niche <- mapply(
  assign_niche,
  cluster_summary$HP,
  cluster_summary$Attack,
  cluster_summary$Defense,
  cluster_summary$SpAttack,
  cluster_summary$SpDefense,
  cluster_summary$Speed
)
plot_data <- plot_data %>%
  left_join(
    cluster_summary %>%
      select(Cluster, Niche),
    by = "Cluster"
  )

plot_data$hover <- paste0(
  "<b>", plot_data$name, "</b><br>",
  "<b>Cluster:</b> ", plot_data$Cluster, "<br>",
  "<b>Niche:</b> ", plot_data$Niche, "<br>",
  "<b>Type:</b> ", plot_data$type1,
  ifelse(
    plot_data$type2 == "" | is.na(plot_data$type2),
    "",
    paste0("/", plot_data$type2)
  ),
  "<br><br>",
  "HP: ", plot_data$hp,
  "<br>Attack: ", plot_data$attack,
  "<br>Defense: ", plot_data$defense,
  "<br>Sp. Attack: ", plot_data$sp_attack,
  "<br>Sp. Defense: ", plot_data$sp_defense,
  "<br>Speed: ", plot_data$speed
)

# Plot the discovered clusters
fig <- plot_ly(
  data = plot_data,
  x = ~PC1,
  y = ~PC2,
  type = "scatter",
  mode = "markers",
  color = ~Cluster,
  text = ~hover,
  hoverinfo = "text",
  marker = list(
    size = 8,
    opacity = 0.8
  )
)

fig <- fig %>%
  layout(
    title = "Pokémon Clusters in Autoencoder Latent Space",
    xaxis = list(
      title = "PC1"
    ),
    yaxis = list(
      title = "PC2"
    ),
    legend = list(
      title = list(
        text = "Cluster"
      )
    )
  )

fig