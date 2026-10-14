library(dplyr)
library(ggplot2)
library(GGally)
library(janitor)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(stringr)

# Define a custom function for mode (this must be run first!)
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# 1. Load the dataset into R, explicitly handling 'NAN' and '\N' strings found in the CSV
# NOTE: Replace the placeholders below with your Google Drive File IDs after uploading
basics_file_id  <- "1qQPWCOAubb5cnHPTsirIgQz0HdpuvaYM"
ratings_file_id <- "1sV68QFXXp7c6qNZ6aR-hGhnGoKPDhCwV"

basics_url  <- paste0("https://drive.google.com/uc?export=download&id=", basics_file_id)
ratings_url <- paste0("https://drive.google.com/uc?export=download&id=", ratings_file_id)

# If running locally before uploading to Google Drive, replace basics_url with "title.basics.csv"
# and ratings_url with "title.ratings.csv"
df_basics <- read.csv(
  basics_url,
  header = TRUE,
  na.strings = c("NAN", "          NAN", "   NAN", "\\N", "NA", ""),
  stringsAsFactors = FALSE
)

df_ratings <- read.csv(
  ratings_url,
  header = TRUE,
  na.strings = c("NAN", "          NAN", "   NAN", "\\N", "NA", ""),
  stringsAsFactors = FALSE
)

# Merge the basics and ratings datasets by 'tconst'
df <- inner_join(df_basics, df_ratings, by = "tconst")

# Standardize names so the rest of the workflow works seamlessly
df <- df %>%
  clean_names() %>%
  mutate(primary_genre = word(genres, 1, sep = ",")) %>%
  rename(
    Title_ID       = tconst,
    Title_Type     = title_type,
    Primary_Title  = primary_title,
    Original_Title = original_title,
    Is_Adult       = is_adult,
    Start_Year     = start_year,
    End_Year       = end_year,
    Runtime        = runtime_minutes,
    Genre          = primary_genre,
    Rating         = average_rating,
    Votes          = num_votes
  ) %>%
  filter(Title_Type == "movie") %>%
  select(Title_ID, Primary_Title, Title_Type, Is_Adult, Start_Year, Runtime, Genre, Rating, Votes)

# Convert numeric columns explicitly
df$Is_Adult   <- as.numeric(df$Is_Adult)
df$Start_Year <- as.numeric(df$Start_Year)
df$Runtime    <- as.numeric(df$Runtime)
df$Rating     <- as.numeric(df$Rating)
df$Votes      <- as.numeric(df$Votes)

# Handle Missing Values: Create a new dataframe removing rows with NA (na.omit)
df_clean <- na.omit(df)

# 2. Display the first few rows of the dataset
print("First few rows of the data:")
head(df_clean)

# 3. Show shape (rows x columns)
cat("\nShape of the dataset (Rows, Columns):")
print(dim(df))

# 4. Display data types of each column (Structure of the data frame)
cat("\nStructure of the dataset:")
str(df)

# 5. Create a summary table for Votes (and Rating)
votes_stats <- df_clean %>%
  summarise(
    Count   = n(),
    Mean    = mean(Votes, na.rm = TRUE),
    Median  = median(Votes, na.rm = TRUE),
    Mode    = get_mode(Votes),
    Std_Dev = sd(Votes, na.rm = TRUE),
    Min     = min(Votes, na.rm = TRUE),
    Max     = max(Votes, na.rm = TRUE)
  )
cat("\nDescriptive Statistics for Votes:\n")
print(votes_stats)

cat("\nSummary statistics for ALL variables:\n")
print(summary(df))

cat("\nSummary statistics for NUMERIC variables:\n")
print(df %>% select(where(is.numeric)) %>% summary())

# Identify categorical and numerical features
numerical_features <- names(df)[sapply(df, is.numeric)]
categorical_features <- names(df)[sapply(df, is.character) | sapply(df, is.factor)]

print("--- Identified Features ---")
print(paste("Numerical Features:", paste(numerical_features, collapse = ", ")))
print(paste("Categorical Features:", paste(categorical_features, collapse = ", ")))

# --- 1. Univariate Analysis ---

ggplot(df_clean, aes(x = Rating)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of IMDb Movie Ratings", x = "Rating")

ggplot(df_clean, aes(y = Runtime)) +
  geom_boxplot(fill = "tomato") +
  labs(title = "Boxplot of Movie Runtime (Minutes)", y = "Runtime (Minutes)")

# Keep Top 10 Genres for clean bar chart visualization
top_10_genres <- names(sort(table(df_clean$Genre), decreasing = TRUE))[1:10]
df_clean <- df_clean %>%
  mutate(Genre_Group = ifelse(Genre %in% top_10_genres, Genre, "Other"))

ggplot(df_clean, aes(x = Genre_Group, fill = Genre_Group)) +
  geom_bar() +
  coord_flip() +
  labs(title = "Movie Frequency by Primary Genre", x = "Genre", y = "Count")

cat("\nFrequency of Movies by Genre:")
genre_freq <- df_clean %>%
  group_by(Genre_Group) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count))
print(genre_freq)

cat("\nFrequency of Movies by Adult Category:")
adult_freq <- df_clean %>%
  group_by(Is_Adult) %>%
  summarise(Count = n())
print(adult_freq)

# --- 2. Bivariate Analysis ---

numerical_df <- df_clean %>%
  select(where(is.numeric))

ggcorr(numerical_df, label = TRUE)

ggplot(df_clean, aes(x = Runtime, y = Votes)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Scatter Plot: Runtime vs. Votes", x = "Runtime (Minutes)", y = "Total Votes")

ggplot(df_clean, aes(x = Runtime, y = Rating)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "lm", se = FALSE, color = "orange") +
  labs(title = "Scatter Plot: Runtime vs. Rating", x = "Runtime (Minutes)", y = "IMDb Rating")

# Create IMDb Bayesian Weighted Score for Best/Worst Movie Calculation
C_mean <- mean(df_clean$Rating, na.rm = TRUE)
m_vote <- quantile(df_clean$Votes, 0.90, na.rm = TRUE)

df_clean <- df_clean %>%
  mutate(Weighted_Score = (Votes / (Votes + m_vote)) * Rating + (m_vote / (Votes + m_vote)) * C_mean)

ggplot(df_clean, aes(x = Genre_Group, y = Weighted_Score, fill = Genre_Group)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Boxplot: Weighted Score by Genre", x = "Genre", y = "Weighted Score") +
  theme(legend.position = "none")

ggplot(df_clean, aes(x = as.factor(Is_Adult), y = Rating, fill = as.factor(Is_Adult))) +
  geom_boxplot() +
  labs(title = "Boxplot: Movie Rating Distribution by Adult Status", x = "Is Adult (0 = No, 1 = Yes)", y = "IMDb Rating") +
  theme(legend.position = "none")

ggplot(df_clean, aes(x = Start_Year, y = Rating)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Scatter Plot: Release Year vs. Rating (Check Patterns)", x = "Release Year", y = "Rating")

ggplot(df_clean, aes(x = Runtime)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Runtime (Check Skewness)", x = "Runtime (Minutes)")

ggplot(df_clean, aes(y = Votes)) +
  geom_boxplot(fill = "tomato") +
  scale_y_log10() +
  labs(title = "Boxplot of Votes on Log10 Scale (Check Outliers)", y = "Votes (Log10)")

# --- Find Most-Watched Movie (Before Outlier Filtering) ---
cat("\n=== MOST-WATCHED MOVIES IN DATASET (BY TOTAL VOTES) ===\n")
most_watched_movies <- df_clean %>%
  arrange(desc(Votes)) %>%
  select(Title_ID, Primary_Title, Start_Year, Genre, Rating, Votes) %>%
  head(5)
print(most_watched_movies)

# Data Processing

cat("\nMissing Values (NA Count) per Column:")
print(colSums(is.na(df)))

cat("\nTotal NA values after cleaning:")
print(sum(is.na(df_clean)))

# --- Outlier Handling (IQR Method on Runtime) ---

df_clean <- df_clean[!duplicated(df_clean$Title_ID), ]

Q1_r <- quantile(df_clean$Runtime, 0.25, na.rm = TRUE)
Q3_r <- quantile(df_clean$Runtime, 0.75, na.rm = TRUE)
IQR_r <- Q3_r - Q1_r

lower_bound_r <- Q1_r - 1.5 * IQR_r
upper_bound_r <- Q3_r + 1.5 * IQR_r

cat("\nOutlier bounds for Runtime (Lower, Upper): ", lower_bound_r, ", ", upper_bound_r, "\n")

df_final <- df_clean %>%
  filter(Runtime >= lower_bound_r & Runtime <= upper_bound_r)

cat("\nOriginal Rows (Cleaned Data):", nrow(df_clean))
cat("\nRows after Outlier Removal:", nrow(df_final), "\n")

# --- 3. Data Conversion (Label Encoding) ---

# Create a 3-class Rating Tier for Classification Evaluation (Low, Medium, High)
df_final <- df_final %>%
  mutate(
    Rating_Class = case_when(
      Rating < 5.5 ~ "Low_Rated",
      Rating >= 5.5 & Rating < 7.0 ~ "Medium_Rated",
      Rating >= 7.0 ~ "High_Rated"
    )
  )

df_converted <- df_final %>%
  mutate(
    Title_Type_Encoded   = as.numeric(as.factor(Title_Type)),
    Genre_Encoded        = as.numeric(as.factor(Genre_Group)),
    Is_Adult_Encoded     = as.numeric(as.factor(Is_Adult)),
    Rating_Class_Encoded = as.numeric(as.factor(Rating_Class))
  )

cat("\nStructure after Label Encoding:\n")
str(df_converted)

# --- 4. Data Transformation (Normalization and Log Transformation) ---

min_max_normalize <- function(x) {
  x <- as.numeric(x)
  mn <- min(x, na.rm = TRUE)
  mx <- max(x, na.rm = TRUE)
  if (isTRUE(all.equal(mx, mn))) return(rep(0, length(x)))
  (x - mn) / (mx - mn)
}

df_transformed <- df_converted %>%
  mutate(
    Start_Year_Scaled     = min_max_normalize(Start_Year),
    Runtime_Scaled        = min_max_normalize(Runtime),
    Rating_Scaled         = min_max_normalize(Rating),
    Weighted_Score_Scaled = min_max_normalize(Weighted_Score),
    Votes_Log             = log10(Votes + 1)
  )

cat("\nStructure after Transformation/Scaling:\n")
str(df_transformed)

# --- 5. Feature Selection ---

df_selected <- df_transformed %>%
  select(
    Genre_Encoded,
    Is_Adult_Encoded,
    Rating_Class_Encoded,
    Start_Year_Scaled,
    Runtime_Scaled,
    Rating_Scaled,
    Weighted_Score_Scaled,
    Votes_Log
  )

cor_matrix_final <- cor(df_selected, use = "complete.obs")
print("\nFinal Correlation Matrix (for Feature Selection):")
print(round(cor_matrix_final, 2))

variance_check <- sapply(df_selected, var, na.rm = TRUE)
low_variance_features <- names(variance_check)[variance_check < 0.001]

cat("\nFeatures with Variance < 0.001 (Low Variance Threshold):\n")
print(low_variance_features)

df_final_project <- df_selected %>%
  select(-any_of(low_variance_features))

cat("\nStructure of the FINAL Project Data Frame (Ready for Modeling):\n")
str(df_final_project)
head(df_final_project)

# --- 6. Model Evaluation (75% Train / 25% Test Split) ---

set.seed(123)

# ==============================================================================
# PART A: Classification Evaluation (Decision Tree & Random Forest on Rating_Class)
# ==============================================================================
target_col <- "Rating_Class_Encoded"

# Remove direct Rating leak columns when predicting Rating_Class_Encoded
model_df <- df_final_project %>%
  select(-Rating_Scaled, -Weighted_Score_Scaled)

model_df[[target_col]] <- as.factor(model_df[[target_col]])
model_df <- na.omit(model_df)

cat("\nNumber of classes in target:", nlevels(model_df[[target_col]]), "\n")
print(table(model_df[[target_col]]))

# 75% Train / 25% Test Split
trainIndex <- createDataPartition(model_df[[target_col]], p = 0.75, list = FALSE)
trainData  <- model_df[trainIndex, ]
testData   <- model_df[-trainIndex, ]

cat("\nTraining Set Rows (75%):", nrow(trainData))
cat("\nTesting Set Rows  (25%):", nrow(testData), "\n")

form <- as.formula(paste(target_col, "~ ."))

# --- 1. Decision Tree Model ---
model_dt <- rpart(form, data = trainData, method = "class", control = rpart.control(cp = 0.005))
rpart.plot(model_dt, box.palette = "BuGn", main = "Decision Tree Visualization (75% Train Set)")

pred_dt <- predict(model_dt, newdata = testData, type = "class")
pred_dt <- factor(pred_dt, levels = levels(testData[[target_col]]))

conf_mat_dt <- confusionMatrix(pred_dt, testData[[target_col]])
cat("\n=== DECISION TREE CONFUSION MATRIX ===\n")
print(conf_mat_dt)

accuracy_dt  <- conf_mat_dt$overall["Accuracy"]
byClass_dt   <- conf_mat_dt$byClass
precision_dt <- mean(byClass_dt[, "Pos Pred Value"], na.rm = TRUE)
recall_dt    <- mean(byClass_dt[, "Sensitivity"], na.rm = TRUE)
f1_dt        <- mean(byClass_dt[, "F1"], na.rm = TRUE)

print(paste("Decision Tree Accuracy: ", round(accuracy_dt, 4)))
print(paste("Decision Tree Precision:", round(precision_dt, 4)))
print(paste("Decision Tree Recall:   ", round(recall_dt, 4)))
print(paste("Decision Tree F1 Score: ", round(f1_dt, 4)))

# Visualize Decision Tree Confusion Matrix
cm_df_dt <- as.data.frame(conf_mat_dt$table)

ggplot(cm_df_dt, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#e7f0fa", high = "#2c7bb6") +
  theme_minimal() +
  labs(
    title = "Decision Tree Confusion Matrix Heatmap (25% Test Set)",
    x = "Actual Class",
    y = "Predicted Class",
    fill = "Frequency"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 14)
  )

# --- 2. Random Forest Model (With Colorful Error Plot & Option B ggplot2 Feature Importance) ---
set.seed(123)
model_rf <- randomForest(form, data = trainData, ntree = 200, importance = TRUE)

# Colorful Random Forest Error Rate Plot
rf_colors <- c("black", "#e41a1c", "#377eb8", "#4daf4a")
plot(model_rf, main = "Random Forest Error Rate Across Trees", col = rf_colors, lwd = 2)
legend(
  "topright",
  legend = colnames(model_rf$err.rate),
  col = rf_colors,
  lty = 1:4,
  lwd = 2,
  cex = 0.85,
  bg = "white"
)

# Option B: Full Colorful ggplot2 Feature Importance Bar Chart
imp_df <- as.data.frame(importance(model_rf))
imp_df$Feature <- rownames(imp_df)

ggplot(imp_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini, fill = Feature)) +
  geom_col(width = 0.65, color = "black", show.legend = FALSE) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  labs(
    title = "Random Forest Feature Importance (Mean Decrease Gini)",
    x = "Features",
    y = "Importance Score (Mean Decrease Gini)"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 13)
  )

# Predictions & Confusion Matrix
pred_rf <- predict(model_rf, newdata = testData, type = "class")
pred_rf <- factor(pred_rf, levels = levels(testData[[target_col]]))

conf_mat_rf <- confusionMatrix(pred_rf, testData[[target_col]])
cat("\n=== RANDOM FOREST CONFUSION MATRIX ===\n")
print(conf_mat_rf)

accuracy_rf  <- conf_mat_rf$overall["Accuracy"]
byClass_rf   <- conf_mat_rf$byClass
precision_rf <- mean(byClass_rf[, "Pos Pred Value"], na.rm = TRUE)
recall_rf    <- mean(byClass_rf[, "Sensitivity"], na.rm = TRUE)
f1_rf        <- mean(byClass_rf[, "F1"], na.rm = TRUE)

print(paste("Random Forest Accuracy: ", round(accuracy_rf, 4)))
print(paste("Random Forest Precision:", round(precision_rf, 4)))
print(paste("Random Forest Recall:   ", round(recall_rf, 4)))
print(paste("Random Forest F1 Score: ", round(f1_rf, 4)))

# Colorful Random Forest Confusion Matrix Heatmap
cm_df_rf <- as.data.frame(conf_mat_rf$table)

ggplot(cm_df_rf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = Freq), size = 6, fontface = "bold", color = "black") +
  scale_fill_gradientn(colors = c("#f7fbff", "#6baed6", "#08519c")) +
  theme_minimal() +
  labs(
    title = "Random Forest Confusion Matrix Heatmap (25% Test Set)",
    x = "Actual Class (Reference)",
    y = "Predicted Class",
    fill = "Count"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.text = element_text(size = 12, face = "bold"),
    axis.title = element_text(size = 14)
  )

# ==============================================================================
# PART B: Regression ML Models to Find Best, Worst, Highest-Rated & Lowest-Rated Movies
# ==============================================================================
set.seed(123)
reg_train_idx <- createDataPartition(df_clean$Weighted_Score, p = 0.75, list = FALSE)
reg_train     <- df_clean[reg_train_idx, ]
reg_test      <- df_clean[-reg_train_idx, ]

# 1. Train Decision Tree & Random Forest to Predict Weighted_Score (Best & Worst Movie)
dt_best_model <- rpart(
  Weighted_Score ~ Start_Year + Runtime + Is_Adult + Votes + Genre_Group,
  data = reg_train, method = "anova"
)
rf_best_model <- randomForest(
  Weighted_Score ~ Start_Year + Runtime + Is_Adult + Votes + Genre_Group,
  data = reg_train, ntree = 100
)

# 2. Train Decision Tree & Random Forest to Predict Rating (Highest & Lowest Rated Movie)
dt_rate_model <- rpart(
  Rating ~ Start_Year + Runtime + Is_Adult + Votes + Genre_Group,
  data = reg_train, method = "anova"
)
rf_rate_model <- randomForest(
  Rating ~ Start_Year + Runtime + Is_Adult + Votes + Genre_Group,
  data = reg_train, ntree = 100
)

# Predict across the full cleaned dataset to rank movies
df_clean$Pred_Weighted_RF <- predict(rf_best_model, newdata = df_clean)
df_clean$Pred_Weighted_DT <- predict(dt_best_model, newdata = df_clean)
df_clean$Pred_Rating_RF   <- predict(rf_rate_model, newdata = df_clean)
df_clean$Pred_Rating_DT   <- predict(dt_rate_model, newdata = df_clean)

cat("\n=== 1. BEST MOVIE USING ML MODELS (BY PREDICTED WEIGHTED SCORE) ===\n")
best_movies_ml <- df_clean %>%
  arrange(desc(Pred_Weighted_RF), desc(Votes)) %>%
  select(Title_ID, Primary_Title, Start_Year, Rating, Votes, Weighted_Score, Pred_Weighted_RF, Pred_Weighted_DT) %>%
  head(5)
print(best_movies_ml)

cat("\n=== 2. WORST MOVIE USING ML MODELS (BY PREDICTED WEIGHTED SCORE) ===\n")
worst_movies_ml <- df_clean %>%
  arrange(Pred_Weighted_RF, desc(Votes)) %>%
  select(Title_ID, Primary_Title, Start_Year, Rating, Votes, Weighted_Score, Pred_Weighted_RF, Pred_Weighted_DT) %>%
  head(5)
print(worst_movies_ml)

cat("\n=== 3. HIGHEST-RATED MOVIE USING ML MODELS (BY PREDICTED RATING) ===\n")
highest_rated_ml <- df_clean %>%
  arrange(desc(Pred_Rating_RF), desc(Votes)) %>%
  select(Title_ID, Primary_Title, Start_Year, Rating, Votes, Pred_Rating_RF, Pred_Rating_DT) %>%
  head(5)
print(highest_rated_ml)

cat("\n=== 4. LOWEST-RATED MOVIE USING ML MODELS (BY PREDICTED RATING) ===\n")
lowest_rated_ml <- df_clean %>%
  arrange(Pred_Rating_RF, desc(Votes)) %>%
  select(Title_ID, Primary_Title, Start_Year, Rating, Votes, Pred_Rating_RF, Pred_Rating_DT) %>%
  head(5)
print(lowest_rated_ml)

# Visualize Actual vs. Predicted Scores for Random Forest and Decision Tree (25% Test Set)
reg_test$RF_Pred <- predict(rf_best_model, newdata = reg_test)
reg_test$DT_Pred <- predict(dt_best_model, newdata = reg_test)

ggplot(reg_test) +
  geom_point(aes(x = Weighted_Score, y = DT_Pred, color = "Decision Tree"), alpha = 0.4) +
  geom_point(aes(x = Weighted_Score, y = RF_Pred, color = "Random Forest"), alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  scale_color_manual(values = c("Decision Tree" = "#d95f02", "Random Forest" = "#1b9e77")) +
  labs(
    title = "Random Forest vs. Decision Tree: Actual vs. Predicted Movie Score (25% Test Set)",
    x = "Actual IMDb Weighted Score",
    y = "ML Predicted Weighted Score",
    color = "ML Model"
  ) +
  theme_minimal()