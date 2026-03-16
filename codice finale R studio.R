#COMPETIZIONE PER PREVEDERE PREZZI DELLE CASE (KAGGLE COMPETITION)--------------

library(dplyr)
library(corrgram)
library(mctest)
library(caret)
library(rpart)
library(rpart.plot)
library(plyr)
library(car)
library(xgboost)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(glmnet)
library(gbm)
library(ranger)
library(nnet)
library(xgboost)
library(plyr)
library(mice)

library(psych)
library(VIM)
library(funModeling)
library(factoextra)
library(factoextra)
library(GGally)
library(rgl)

library(gvlma)
library(MASS)
library(car)
library(lmtest)
library(sandwich)
library(mgcv)
library(ggeffects)
library(gratia)
library(stringr)
library(ppcor)
library(corrplot)
library(plyr)

library(leaps)
library(robustbase)
library(forestmodel)
library(ggplot2)
library(MASS)
library(caret)


library(factorMerger)

library(Boruta)
library(pROC)
library(caret)
library(rpart)
library(rpart.plot)
library(gbm)
library(nnet)



#-------------------------------------------------------------------------------
options(scipen=999)
train <- read.csv("train.csv", sep="," , dec = ".",stringsAsFactors=TRUE,na.strings=c("NA","N/A", "Unknown ", "Unknown", "?",""))
test <- read.csv("test.csv", sep="," , dec = ".",stringsAsFactors=TRUE,na.strings=c("NA","N/A", "Unknown ", "Unknown", "?",""))


# PANORAMICA PREZZI 
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "black", alpha = 0.7) +
  scale_x_continuous(labels = scales::comma) +
  labs(
    title = "Distribuzione dei Prezzi delle case",
    x = "Prezzo (in Euro)",
    y = "Numero di case"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

# TRASFOMRAZIONI VARIABILI √ ---------------------------------------------------
train$id <- NULL
test$id <- NULL


# MISSING  √ -------------------------------------------------------------------
y <- log(train$SalePrice)
train_x <- subset(train, select = -SalePrice)
full <- rbind(train_x, test) # unisco train e test per fare trasformazioni identiche

# converto factor -> character per imputare senza problemi
char_cols <- names(full)[sapply(full, is.factor)]
for (col in char_cols) {
  full[[col]] <- as.character(full[[col]])}

# -------------------------
# 1) NA = assenza reale
# -------------------------
none_cols <- c(
  "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
  "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
  "PoolQC", "Fence", "MiscFeature", "MasVnrType"
)

for (col in none_cols) {
  if (col %in% names(full)) {
    full[[col]][is.na(full[[col]])] <- "None"
  }
}

# basement numeriche: assenza seminterrato -> 0
bsmt_num_cols <- c("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                   "BsmtFullBath", "BsmtHalfBath")

for (col in bsmt_num_cols) {
  if (col %in% names(full)) {
    full[[col]][is.na(full[[col]])] <- 0
  }
}

# garage numeriche: assenza garage -> 0
garage_num_cols <- c("GarageYrBlt", "GarageCars", "GarageArea")

for (col in garage_num_cols) {
  if (col %in% names(full)) {
    full[[col]][is.na(full[[col]])] <- 0
  }
}

# masonry veneer
if ("MasVnrArea" %in% names(full)) {
  full$MasVnrArea[is.na(full$MasVnrArea)] <- 0
}

# -------------------------
# 2) imputazione mirata
# -------------------------

# LotFrontage: meglio mediana per quartiere
if ("LotFrontage" %in% names(full) && "Neighborhood" %in% names(full)) {
  full$LotFrontage <- ave(
    full$LotFrontage,
    full$Neighborhood,
    FUN = function(x) {
      x[is.na(x)] <- median(x, na.rm = TRUE)
      x
    }
  )
}

Mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

# categoriche residue -> moda
cat_cols <- names(full)[sapply(full, is.character)]

for (col in cat_cols) {
  if (any(is.na(full[[col]]))) {
    full[[col]][is.na(full[[col]])] <- Mode(full[[col]])
  }
}

# numeriche residue -> mediana
num_cols <- names(full)[sapply(full, is.numeric)]

for (col in num_cols) {
  if (any(is.na(full[[col]]))) {
    full[[col]][is.na(full[[col]])] <- median(full[[col]], na.rm = TRUE)
  }
}

# riconverto character -> factor
cat_cols <- names(full)[sapply(full, is.character)]
for (col in cat_cols) {
  full[[col]] <- as.factor(full[[col]])
}

# -------------------------
# 3) ricreo train e test
# -------------------------
train_clean <- full[1:nrow(train_x), ]
test_clean  <- full[(nrow(train_x) + 1):nrow(full), ]

train <- train_clean
train$SalePrice <- y
test  <- test_clean

# controllo finale
sum(is.na(train))
sum(is.na(test))


#-------------------------------------------------------------------------------
# -------------------------
# 1) feature engineering PRIMA dei log
# -------------------------

# superficie totale
if (all(c("TotalBsmtSF","X1stFlrSF","X2ndFlrSF") %in% names(train))) {
  train$TotalSF <- train$TotalBsmtSF + train$X1stFlrSF + train$X2ndFlrSF
  test$TotalSF  <- test$TotalBsmtSF + test$X1stFlrSF + test$X2ndFlrSF
}

# bagni totali pesati
if (all(c("FullBath","HalfBath","BsmtFullBath","BsmtHalfBath") %in% names(train))) {
  train$TotalBathrooms <- train$FullBath + 0.5 * train$HalfBath +
    train$BsmtFullBath + 0.5 * train$BsmtHalfBath
  test$TotalBathrooms  <- test$FullBath + 0.5 * test$HalfBath +
    test$BsmtFullBath + 0.5 * test$BsmtHalfBath
}

# età casa
if (all(c("YrSold","YearBuilt") %in% names(train))) {
  train$HouseAge <- train$YrSold - train$YearBuilt
  test$HouseAge  <- test$YrSold - test$YearBuilt
}

# età ristrutturazione
if (all(c("YrSold","YearRemodAdd") %in% names(train))) {
  train$RemodAge <- train$YrSold - train$YearRemodAdd
  test$RemodAge  <- test$YrSold - test$YearRemodAdd
}

# casa ristrutturata
if (all(c("YearBuilt","YearRemodAdd") %in% names(train))) {
  train$Remodeled <- ifelse(train$YearBuilt != train$YearRemodAdd, 1, 0)
  test$Remodeled  <- ifelse(test$YearBuilt != test$YearRemodAdd, 1, 0)
}

# qualità * area abitabile
if (all(c("OverallQual","GrLivArea") %in% names(train))) {
  train$QualLivArea <- train$OverallQual * train$GrLivArea
  test$QualLivArea  <- test$OverallQual * test$GrLivArea
}

# spazio esterno totale
if (all(c("OpenPorchSF","EnclosedPorch","X3SsnPorch","ScreenPorch","WoodDeckSF") %in% names(train))) {
  train$TotalPorchSF <- train$OpenPorchSF + train$EnclosedPorch +
    train$X3SsnPorch + train$ScreenPorch + train$WoodDeckSF
  test$TotalPorchSF  <- test$OpenPorchSF + test$EnclosedPorch +
    test$X3SsnPorch + test$ScreenPorch + test$WoodDeckSF
}

# indicatori presenza
if ("GarageArea" %in% names(train)) {
  train$HasGarage <- ifelse(train$GarageArea > 0, 1, 0)
  test$HasGarage  <- ifelse(test$GarageArea > 0, 1, 0)
}

if ("TotalBsmtSF" %in% names(train)) {
  train$HasBsmt <- ifelse(train$TotalBsmtSF > 0, 1, 0)
  test$HasBsmt  <- ifelse(test$TotalBsmtSF > 0, 1, 0)
}

if ("Fireplaces" %in% names(train)) {
  train$HasFireplace <- ifelse(train$Fireplaces > 0, 1, 0)
  test$HasFireplace  <- ifelse(test$Fireplaces > 0, 1, 0)
}

if ("PoolArea" %in% names(train)) {
  train$HasPool <- ifelse(train$PoolArea > 0, 1, 0)
  test$HasPool  <- ifelse(test$PoolArea > 0, 1, 0)
}

if ("LowQualFinSF" %in% names(train)) {
  train$HasLowQual <- ifelse(train$LowQualFinSF > 0, 1, 0)
  test$HasLowQual  <- ifelse(test$LowQualFinSF > 0, 1, 0)
}

# -------------------------
# 2) variabili ordinali -> numeriche ordinate
# -------------------------
qual_levels <- c("None","Po","Fa","TA","Gd","Ex")

ord_cols <- c(
  "ExterQual","ExterCond",
  "BsmtQual","BsmtCond",
  "HeatingQC","KitchenQual",
  "FireplaceQu","GarageQual",
  "GarageCond","PoolQC"
)

for (col in ord_cols) {
  if (col %in% names(train)) {
    train[[col]] <- as.numeric(factor(train[[col]], levels = qual_levels))
    test[[col]]  <- as.numeric(factor(test[[col]], levels = qual_levels))
  }
}

# BsmtExposure: None < No < Mn < Av < Gd
if ("BsmtExposure" %in% names(train)) {
  expo_levels <- c("None","No","Mn","Av","Gd")
  train$BsmtExposure <- as.numeric(factor(train$BsmtExposure, levels = expo_levels))
  test$BsmtExposure  <- as.numeric(factor(test$BsmtExposure, levels = expo_levels))
}

# BsmtFinType: None < Unf < LwQ < Rec < BLQ < ALQ < GLQ
bsmtfin_levels <- c("None","Unf","LwQ","Rec","BLQ","ALQ","GLQ")
for (col in c("BsmtFinType1","BsmtFinType2")) {
  if (col %in% names(train)) {
    train[[col]] <- as.numeric(factor(train[[col]], levels = bsmtfin_levels))
    test[[col]]  <- as.numeric(factor(test[[col]], levels = bsmtfin_levels))
  }
}

# GarageFinish: None < Unf < RFn < Fin
if ("GarageFinish" %in% names(train)) {
  garagefin_levels <- c("None","Unf","RFn","Fin")
  train$GarageFinish <- as.numeric(factor(train$GarageFinish, levels = garagefin_levels))
  test$GarageFinish  <- as.numeric(factor(test$GarageFinish, levels = garagefin_levels))
}

# -------------------------
# 3) log transform su variabili skewed
# -------------------------
skew_cols <- c(
  "LotArea","GrLivArea","TotalBsmtSF","X1stFlrSF","X2ndFlrSF",
  "MasVnrArea","GarageArea","WoodDeckSF","OpenPorchSF",
  "EnclosedPorch","X3SsnPorch","ScreenPorch","LotFrontage",
  "TotalSF","QualLivArea","TotalPorchSF"
)

for (col in skew_cols) {
  if (col %in% names(train)) {
    train[[col]] <- log1p(train[[col]])
    test[[col]]  <- log1p(test[[col]])
  }
}

# -------------------------
# 4) pulizia colonne poco utili / molto sparse
# -------------------------
drop_cols <- c("Utilities", "PoolQC", "MiscFeature")

train <- train[, !(names(train) %in% drop_cols)]
test  <- test[, !(names(test) %in% drop_cols)]

# rimuovo variabili grezze sostituite da flag
if ("PoolArea" %in% names(train)) {
  train$PoolArea <- NULL
  test$PoolArea  <- NULL
}

if ("LowQualFinSF" %in% names(train)) {
  train$LowQualFinSF <- NULL
  test$LowQualFinSF  <- NULL
}

# -------------------------
# 5) target transform
# -------------------------
train$SalePrice <- log1p(train$SalePrice)
#-------------------------------------------------------------------------------

#################### COLLINEARITA' #############################################
# matrice correlazioni numeriche
numdata <- train %>% dplyr::select(where(is.numeric))

R <- cor(numdata, use = "pairwise.complete.obs")

# rimuove solo correlazioni estreme
high_corr <- caret::findCorrelation(R, cutoff = 0.95)

train <- train[, -high_corr]
test  <- test[, -high_corr]

# --- ONE HOT ENCODING --------------------------------------------------------
library(dplyr)

# salvo target
y <- train$SalePrice

# tolgo target dal train
train_x <- train %>% dplyr::select(-SalePrice)

# unisco train e test
full_data <- bind_rows(train_x, test)

# one-hot encoding
full_data_encoded <- model.matrix(~ . - 1, data = full_data)
full_data_encoded <- as.data.frame(full_data_encoded)

# separo train e test
train <- full_data_encoded[1:nrow(train_x), ]
test  <- full_data_encoded[(nrow(train_x) + 1):nrow(full_data_encoded), ]

# riattacco target
train$SalePrice <- y


# OUTLIERS  √ ------------------------------------------------------------------
train <- train[!(train$GrLivArea > 4000 & train$SalePrice < 300000), ]

# ------------------------------------------------------------------------------
######################## TUNING DEI MODELLI ####################################
# ------------------------------------------------------------------------------

############################# LASSO ############################################
################################################################################
set.seed(1)
cvCtrl <- trainControl(method = "repeatedcv",number = 10, repeats = 5)

grid <- expand.grid(
  alpha = seq(0,1,0.1),
  lambda = 10^seq(-5, 1, length = 100)
)

tune_glmnet <- train(
  SalePrice ~ .,
  data = train,
  method = "glmnet",
  metric = "RMSE",
  trControl = cvCtrl,
  tuneGrid = grid,
  preProcess = c("center", "scale")
)

print(tune_glmnet)
print(tune_glmnet$bestTune)

best_results <- tune_glmnet$results[order(tune_glmnet$results$RMSE), ]
print(head(best_results, 10))

plot(tune_glmnet)

# RMSE 0.009632666


############################## TREE ############################################
################################################################################
set.seed(123)

cvCtrl <- trainControl(
  method = "cv",
  number = 10
)

tree <- train(
  SalePrice ~ .,
  data = train,
  method = "rpart",
  tuneLength = 10,
  metric = "RMSE",
  trControl = cvCtrl
)

print(tree$results)
print(tree$bestTune)
getTrainPerf(tree)
plot(tree)

# RMSE 0.01562594

############################ RANDOM FOREST #####################################
################################################################################
set.seed(123)
cvCtrl <- trainControl(method = "cv",number = 5)

rf_grid <- expand.grid(
  mtry = c(5, 12, 20),
  splitrule = "variance",
  min.node.size = c(3, 5)
)

tune_rf <- train(
  SalePrice ~ .,
  data = train,
  method = "ranger",
  metric = "RMSE",
  trControl = cvCtrl,
  tuneGrid = rf_grid,
  importance = "impurity",
  num.trees = 300
)

print(tune_rf$results)
print(tune_rf$bestTune)

# RMSE 0.01041764

############################ GRADIENT BOOST ####################################
################################################################################
set.seed(123)
cvCtrl <- trainControl(method = "cv",number = 5)

gb_grid <- expand.grid(
  interaction.depth = c(2, 3),
  n.trees = c(100, 200),
  shrinkage = c(0.05, 0.1),
  n.minobsinnode = 10
)

gb <- train(
  SalePrice ~ .,
  data = train,
  method = "gbm",
  tuneGrid = gb_grid,
  metric = "RMSE",
  trControl = cvCtrl,
  verbose = FALSE
)

print(gb$results)
print(gb$bestTune)
plot(gb)

# RMSE 0.009563829

############################# NEURAL NETWORK ###################################
################################################################################
set.seed(123)
grid <- expand.grid(size = c(3, 5, 7),decay = c(0.001, 0.01))

nnet_fit <- train(
  SalePrice ~ .,
  data = train,
  method = "nnet",
  preProcess = c("center", "scale", "nzv"),
  metric = "RMSE",
  trControl = cvCtrl,
  tuneGrid = grid,
  maxit = 200,
  trace = FALSE,
  linout = TRUE
)

print(nnet_fit$results)
print(nnet_fit$bestTune)

plot(nnet_fit)

# RMSE 0.01052248

############################# XGBOOST ##########################################
################################################################################
set.seed(123)
cvCtrl <- trainControl(method = "cv",number = 5)

xgb_grid <- expand.grid(
  nrounds = c(300, 500),
  max_depth = c(3, 4),
  eta = c(0.03, 0.05),
  gamma = 0,
  colsample_bytree = c(0.7, 0.9),
  min_child_weight = c(1, 3),
  subsample = c(0.8, 1.0)
)

xgb_model <- train(
  SalePrice ~ .,
  data = train,
  method = "xgbTree",
  metric = "RMSE",
  trControl = cvCtrl,
  tuneGrid = xgb_grid,
  verbose = FALSE
)

print(xgb_model$results)
print(xgb_model$bestTune)

# RMSE 0.009020550

#-------------------------------------------------------------------------------
# =========================
# ENSEMBLE SEMPLICE
# =========================
set.seed(123)
library(caret)

set.seed(123)

folds <- createFolds(train$SalePrice, k = 5)

oof_pred <- rep(NA, nrow(train))

for(i in seq_along(folds)) {
  
  val_idx <- folds[[i]]
  train_idx <- setdiff(seq_len(nrow(train)), val_idx)
  
  train_fold <- train[train_idx, ]
  val_fold <- train[val_idx, ]
  
  # XGBOOST
  xgb_fit <- train(
    SalePrice ~ .,
    data = train_fold,
    method = "xgbTree",
    trControl = trainControl(method="cv", number=5),
    tuneGrid = xgb_model$bestTune
  )
  
  # GBM
  gbm_fit <- train(
    SalePrice ~ .,
    data = train_fold,
    method = "gbm",
    trControl = trainControl(method="cv", number=5),
    tuneGrid = gb$bestTune,
    verbose = FALSE
  )
  
  # LASSO
  lasso_fit <- train(
    SalePrice ~ .,
    data = train_fold,
    method = "glmnet",
    trControl = trainControl(method="cv", number=5),
    tuneGrid = tune_glmnet$bestTune,
    preProcess = c("center","scale")
  )
  
  # predictions
  p1 <- predict(xgb_fit, val_fold)
  p2 <- predict(gbm_fit, val_fold)
  p3 <- predict(lasso_fit, val_fold)
  
  # ensemble
  oof_pred[val_idx] <- 0.6*p1 + 0.25*p2 + 0.15*p3
}

# RMSE reale
rmse_oof <- sqrt(mean((train$SalePrice - oof_pred)^2))

rmse_oof

# RMSE 0.009252208

#################### STAKING ###################################################

library(caret)

set.seed(123)

# ==========================================================
# FOLDS OUT-OF-FOLD
# ==========================================================
folds <- createFolds(train$SalePrice, k = 5, returnTrain = FALSE)

oof_xgb   <- rep(NA, nrow(train))
oof_gbm   <- rep(NA, nrow(train))
oof_lasso <- rep(NA, nrow(train))

# ==========================================================
# STACKING LEVEL 1
# uso bestTune già trovati -> molto più veloce
# ==========================================================
for (i in seq_along(folds)) {
  
  cat("Fold:", i, "\n")
  
  val_idx <- folds[[i]]
  tr_idx  <- setdiff(seq_len(nrow(train)), val_idx)
  
  train_fold <- train[tr_idx, ]
  val_fold   <- train[val_idx, ]
  
  # ---------------- XGBOOST ----------------
  fit_xgb <- train(
    SalePrice ~ .,
    data = train_fold,
    method = "xgbTree",
    trControl = trainControl(method = "none"),
    tuneGrid = xgb_model$bestTune,
    verbose = FALSE
  )
  
  oof_xgb[val_idx] <- predict(fit_xgb, newdata = val_fold)
  
  # ---------------- GBM ----------------
  fit_gbm <- train(
    SalePrice ~ .,
    data = train_fold,
    method = "gbm",
    trControl = trainControl(method = "none"),
    tuneGrid = gb$bestTune,
    verbose = FALSE
  )
  
  oof_gbm[val_idx] <- predict(fit_gbm, newdata = val_fold)
  
  # ---------------- LASSO ----------------
  fit_lasso <- train(
    SalePrice ~ .,
    data = train_fold,
    method = "glmnet",
    trControl = trainControl(method = "none"),
    tuneGrid = tune_glmnet$bestTune,
    preProcess = c("center", "scale")
  )
  
  oof_lasso[val_idx] <- predict(fit_lasso, newdata = val_fold)
}

# ==========================================================
# DATASET META-MODEL
# ==========================================================
stack_train <- data.frame(
  xgb = oof_xgb,
  gbm = oof_gbm,
  lasso = oof_lasso,
  SalePrice = train$SalePrice
)

# ==========================================================
# META MODEL
# lm = veloce e spesso ottimo in stacking
# ==========================================================
meta_model <- lm(SalePrice ~ ., data = stack_train)

# RMSE stacking realistico
pred_stack_oof <- predict(meta_model, newdata = stack_train)
rmse_stack <- sqrt(mean((train$SalePrice - pred_stack_oof)^2))
print(rmse_stack)

# ==========================================================
# REFIT MODELLI FULL TRAIN
# ==========================================================
final_xgb <- train(
  SalePrice ~ .,
  data = train,
  method = "xgbTree",
  trControl = trainControl(method = "none"),
  tuneGrid = xgb_model$bestTune,
  verbose = FALSE
)

final_gbm <- train(
  SalePrice ~ .,
  data = train,
  method = "gbm",
  trControl = trainControl(method = "none"),
  tuneGrid = gb$bestTune,
  verbose = FALSE
)

final_lasso <- train(
  SalePrice ~ .,
  data = train,
  method = "glmnet",
  trControl = trainControl(method = "none"),
  tuneGrid = tune_glmnet$bestTune,
  preProcess = c("center", "scale")
)

# ==========================================================
# PREDIZIONI TEST
# ==========================================================
pred_xgb_test   <- predict(final_xgb, newdata = test)
pred_gbm_test   <- predict(final_gbm, newdata = test)
pred_lasso_test <- predict(final_lasso, newdata = test)

stack_test <- data.frame(
  xgb = pred_xgb_test,
  gbm = pred_gbm_test,
  lasso = pred_lasso_test
)

pred_stack_log <- predict(meta_model, newdata = stack_test)
pred_stack <- expm1(pred_stack_log)


# -------------------------
# SUBMISSION FINALE XGBOOST
# -------------------------

# Id originali
test_ids <- read.csv("test.csv", stringsAsFactors = FALSE)$Id

# predizione del modello
pred_loglog <- predict(xgb_model, newdata = test)

# inversione corretta: log1p(log(price)) -> price
pred_price <- exp(expm1(pred_loglog))

# sicurezza
pred_price[pred_price < 0] <- 0

submission <- data.frame(
  Id = test_ids,
  SalePrice = pred_price
)

head(submission)
summary(submission$SalePrice)
nrow(submission)

write.csv(submission, "submission_xgboost.csv", row.names = FALSE)





