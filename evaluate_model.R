args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args_all, value = TRUE)
if (!length(script_arg)) {
  stop("Run this script with Rscript so its project paths can be resolved.")
}

script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/", mustWork = TRUE)
project_dir <- dirname(script_path)
input_args <- commandArgs(trailingOnly = TRUE)
prediction_path <- if (length(input_args) >= 1) {
  input_args[[1]]
} else {
  file.path(project_dir, "artifacts", "test_predictions.csv")
}
manifest_path <- if (length(input_args) >= 2) {
  input_args[[2]]
} else {
  file.path(project_dir, "data", "split_manifest.csv")
}
output_dir <- if (length(input_args) >= 3) {
  input_args[[3]]
} else {
  file.path(project_dir, "artifacts", "r_evaluation")
}

if (!file.exists(prediction_path)) {
  stop("Test predictions CSV not found: ", prediction_path)
}
if (!file.exists(manifest_path)) {
  stop("Split manifest CSV not found: ", manifest_path)
}

classes <- c("No_Accident", "Minor_Accident", "Severe_Accident")
probability_columns <- paste0("prob_", classes)
predictions <- read.csv(prediction_path, stringsAsFactors = FALSE, check.names = FALSE)
manifest <- read.csv(manifest_path, stringsAsFactors = FALSE, check.names = FALSE)

prediction_columns <- c("relative_path", "true", "pred", probability_columns)
missing_prediction_columns <- setdiff(prediction_columns, names(predictions))
if (length(missing_prediction_columns)) {
  stop(
    "Predictions CSV is missing columns: ",
    paste(missing_prediction_columns, collapse = ", ")
  )
}
manifest_columns <- c("relative_path", "label", "split")
missing_manifest_columns <- setdiff(manifest_columns, names(manifest))
if (length(missing_manifest_columns)) {
  stop(
    "Split manifest is missing columns: ",
    paste(missing_manifest_columns, collapse = ", ")
  )
}
if (!nrow(predictions)) {
  stop("Predictions CSV contains no rows.")
}
if (anyNA(predictions[, prediction_columns]) || anyNA(manifest[, manifest_columns])) {
  stop("Predictions or split manifest contain missing values in required columns.")
}
if (anyDuplicated(predictions$relative_path)) {
  stop("Predictions CSV contains duplicate image paths.")
}
if (anyDuplicated(manifest$relative_path)) {
  stop("Split manifest contains duplicate image paths.")
}
if (any(!predictions$true %in% classes) || any(!predictions$pred %in% classes)) {
  stop("Prediction labels must be one of: ", paste(classes, collapse = ", "))
}

manifest_index <- match(predictions$relative_path, manifest$relative_path)
if (anyNA(manifest_index)) {
  stop("Some prediction image paths do not appear in the split manifest.")
}
if (any(manifest$split[manifest_index] != "test")) {
  stop("Predictions contain images not assigned to the held-out test split.")
}
expected_test_paths <- manifest$relative_path[manifest$split == "test"]
if (length(expected_test_paths) != nrow(predictions) ||
    !setequal(expected_test_paths, predictions$relative_path)) {
  stop("Predictions do not contain exactly the images assigned to the test split.")
}
if (any(predictions$true != manifest$label[manifest_index])) {
  stop("Prediction true labels do not match the split manifest.")
}

probabilities <- as.matrix(predictions[, probability_columns, drop = FALSE])
if (!is.numeric(probabilities) || any(!is.finite(probabilities))) {
  stop("Class probability columns must contain finite numeric values.")
}
if (any(probabilities < 0 | probabilities > 1)) {
  stop("Class probabilities must be between 0 and 1.")
}
if (any(abs(rowSums(probabilities) - 1) > 1e-3)) {
  stop("Class probabilities must sum to 1 for every prediction.")
}
probability_predictions <- classes[max.col(probabilities, ties.method = "first")]
if (any(predictions$pred != probability_predictions)) {
  stop("Predicted class does not match the largest saved class probability.")
}

top_probabilities <- t(apply(probabilities, 1, function(row) {
  sort(row, decreasing = TRUE)[1:2]
}))
predicted_confidence <- top_probabilities[, 1]
true_probability <- probabilities[cbind(
  seq_len(nrow(predictions)),
  match(predictions$true, classes)
)]
is_error <- predictions$true != predictions$pred
error_analysis <- data.frame(
  relative_path = predictions$relative_path,
  group_id = if ("group_id" %in% names(manifest)) {
    manifest$group_id[manifest_index]
  } else {
    NA_character_
  },
  true = predictions$true,
  pred = predictions$pred,
  correct = !is_error,
  predicted_confidence = predicted_confidence,
  true_class_probability = true_probability,
  top_two_margin = top_probabilities[, 1] - top_probabilities[, 2],
  error_type = ifelse(
    !is_error, "Correct",
    ifelse(predicted_confidence >= 0.8, "High-confidence error", "Lower-confidence error")
  ),
  stringsAsFactors = FALSE
)
error_analysis <- error_analysis[
  order(error_analysis$correct, -error_analysis$predicted_confidence),
]

actual <- factor(predictions$true, levels = classes)
predicted <- factor(predictions$pred, levels = classes)
confusion <- table(Actual = actual, Predicted = predicted)
true_positive <- diag(confusion)
support <- rowSums(confusion)
predicted_count <- colSums(confusion)

safe_divide <- function(numerator, denominator) {
  result <- numeric(length(numerator))
  valid <- denominator > 0
  result[valid] <- numerator[valid] / denominator[valid]
  result
}

roc_curve <- function(labels, scores) {
  positive_count <- sum(labels)
  negative_count <- length(labels) - positive_count
  if (!positive_count || !negative_count) {
    return(list(fpr = numeric(), tpr = numeric(), auc = NA_real_))
  }

  ordering <- order(scores, decreasing = TRUE)
  sorted_scores <- scores[ordering]
  sorted_labels <- labels[ordering]
  group_starts <- which(!duplicated(sorted_scores))
  group_sizes <- diff(c(group_starts, length(sorted_labels) + 1))
  group_positive <- vapply(seq_along(group_starts), function(i) {
    start <- group_starts[i]
    sum(sorted_labels[start:(start + group_sizes[i] - 1)])
  }, numeric(1))
  group_negative <- group_sizes - group_positive

  tpr <- c(0, cumsum(group_positive) / positive_count)
  fpr <- c(0, cumsum(group_negative) / negative_count)
  list(
    fpr = fpr,
    tpr = tpr,
    auc = sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
  )
}

roc_curves <- lapply(seq_along(classes), function(i) {
  roc_curve(actual == classes[i], probabilities[, i])
})
auc_by_class <- vapply(roc_curves, `[[`, numeric(1), "auc")
precision <- safe_divide(true_positive, predicted_count)
recall <- safe_divide(true_positive, support)
f1 <- safe_divide(2 * precision * recall, precision + recall)
specificity <- vapply(seq_along(classes), function(i) {
  true_negative <- sum(confusion) - support[i] - predicted_count[i] + true_positive[i]
  safe_divide(true_negative, sum(confusion) - support[i])
}, numeric(1))

calibration_rows <- list()
calibration_error <- numeric(length(classes))
for (i in seq_along(classes)) {
  scores <- probabilities[, i]
  observed <- as.numeric(actual == classes[i])
  bins <- cut(scores, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE, labels = FALSE)
  present_bins <- sort(unique(bins))
  class_rows <- lapply(present_bins, function(bin) {
    in_bin <- bins == bin
    data.frame(
      class = classes[i],
      bin = bin,
      confidence_lower = (bin - 1) / 10,
      confidence_upper = bin / 10,
      count = sum(in_bin),
      mean_confidence = mean(scores[in_bin]),
      observed_frequency = mean(observed[in_bin]),
      stringsAsFactors = FALSE
    )
  })
  class_calibration <- do.call(rbind, class_rows)
  calibration_rows[[i]] <- class_calibration
  calibration_error[i] <- sum(
    class_calibration$count / nrow(predictions) *
      abs(class_calibration$mean_confidence - class_calibration$observed_frequency)
  )
}
calibration_bins <- do.call(rbind, calibration_rows)

one_hot <- sapply(classes, function(class_name) as.numeric(actual == class_name))
multiclass_brier <- mean(rowSums((probabilities - one_hot)^2))
accuracy <- sum(diag(confusion)) / sum(confusion)
macro_precision <- mean(precision)
macro_recall <- mean(recall)
macro_f1 <- mean(f1)
weighted_f1 <- weighted.mean(f1, support)
macro_auc <- mean(auc_by_class)
macro_calibration_error <- mean(calibration_error)

set.seed(42)
bootstrap_replicates <- 1000L
indices_by_class <- lapply(classes, function(class_name) which(actual == class_name))
bootstrap_metrics <- matrix(NA_real_, nrow = bootstrap_replicates, ncol = 2L)
colnames(bootstrap_metrics) <- c("accuracy", "macro_f1")
for (replicate in seq_len(bootstrap_replicates)) {
  sampled_indices <- unlist(lapply(indices_by_class, function(indices) {
    sample(indices, size = length(indices), replace = TRUE)
  }), use.names = FALSE)
  sampled_confusion <- table(
    Actual = factor(actual[sampled_indices], levels = classes),
    Predicted = factor(predicted[sampled_indices], levels = classes)
  )
  sampled_tp <- diag(sampled_confusion)
  sampled_precision <- safe_divide(sampled_tp, colSums(sampled_confusion))
  sampled_recall <- safe_divide(sampled_tp, rowSums(sampled_confusion))
  sampled_f1 <- safe_divide(
    2 * sampled_precision * sampled_recall,
    sampled_precision + sampled_recall
  )
  bootstrap_metrics[replicate, ] <- c(
    sum(diag(sampled_confusion)) / length(sampled_indices),
    mean(sampled_f1)
  )
}

accuracy_ci <- quantile(bootstrap_metrics[, "accuracy"], c(0.025, 0.975), names = FALSE)
macro_f1_ci <- quantile(bootstrap_metrics[, "macro_f1"], c(0.025, 0.975), names = FALSE)
summary_metrics <- data.frame(
  metric = c(
    "test_samples", "accuracy", "accuracy_bootstrap_95ci_lower",
    "accuracy_bootstrap_95ci_upper", "macro_precision", "macro_recall",
    "macro_f1", "macro_f1_bootstrap_95ci_lower", "macro_f1_bootstrap_95ci_upper",
    "weighted_f1", "macro_auc_ovr", "multiclass_brier_score",
    "macro_expected_calibration_error"
  ),
  value = c(
    nrow(predictions), accuracy, accuracy_ci, macro_precision, macro_recall,
    macro_f1, macro_f1_ci, weighted_f1, macro_auc, multiclass_brier,
    macro_calibration_error
  ),
  stringsAsFactors = FALSE
)

class_metrics <- data.frame(
  class = classes,
  support = as.integer(support),
  precision = precision,
  recall = recall,
  specificity = specificity,
  f1 = f1,
  auc_ovr = auc_by_class,
  expected_calibration_error = calibration_error,
  stringsAsFactors = FALSE
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(summary_metrics, file.path(output_dir, "summary_metrics.csv"), row.names = FALSE)
write.csv(class_metrics, file.path(output_dir, "class_metrics.csv"), row.names = FALSE)
write.csv(calibration_bins, file.path(output_dir, "calibration_bins.csv"), row.names = FALSE)
write.csv(as.data.frame.matrix(confusion), file.path(output_dir, "confusion_matrix.csv"))
write.csv(error_analysis, file.path(output_dir, "error_analysis.csv"), row.names = FALSE)

if ("group_id" %in% names(manifest)) {
  group_data <- data.frame(
    group_id = manifest$group_id[manifest_index],
    true = predictions$true,
    pred = predictions$pred,
    correct = !is_error,
    stringsAsFactors = FALSE
  )
  valid_groups <- !is.na(group_data$group_id) & nzchar(group_data$group_id)
  group_data <- group_data[valid_groups, , drop = FALSE]
  if (nrow(group_data)) {
    group_rows <- lapply(split(group_data, group_data$group_id), function(group) {
      errors <- group[!group$correct, , drop = FALSE]
      confusion_pair <- if (nrow(errors)) {
        pair_counts <- sort(table(paste(errors$true, "->", errors$pred)), decreasing = TRUE)
        names(pair_counts)[1]
      } else {
        ""
      }
      data.frame(
        group_id = group$group_id[1],
        samples = nrow(group),
        errors = nrow(errors),
        error_rate = mean(!group$correct),
        most_common_error = confusion_pair,
        stringsAsFactors = FALSE
      )
    })
    group_error_analysis <- do.call(rbind, group_rows)
    group_error_analysis <- group_error_analysis[
      order(-group_error_analysis$error_rate, -group_error_analysis$errors),
    ]
    write.csv(
      group_error_analysis,
      file.path(output_dir, "group_error_analysis.csv"),
      row.names = FALSE
    )
  } else {
    write.csv(
      data.frame(
        group_id = character(),
        samples = integer(),
        errors = integer(),
        error_rate = numeric(),
        most_common_error = character()
      ),
      file.path(output_dir, "group_error_analysis.csv"),
      row.names = FALSE
    )
  }
}

plot_classes <- c("No accident", "Minor", "Severe")
plot_colors <- c("#2c7fb8", "#f28e2b", "#d62728")
png(file.path(output_dir, "evaluation_report.png"), width = 1800, height = 1400, res = 150)
par(mfrow = c(2, 2), mar = c(6, 6, 4, 2))

plot.new()
plot.window(xlim = c(0, length(classes)), ylim = c(0, length(classes)))
title("Test confusion matrix", xlab = "Predicted class", ylab = "Actual class")
heat_colors <- colorRampPalette(c("#f7fbff", "#2171b5"))(100)
max_count <- max(confusion)
for (row in seq_along(classes)) {
  for (column in seq_along(classes)) {
    value <- confusion[row, column]
    color_index <- if (max_count == 0) 1 else max(1, ceiling(value / max_count * 100))
    y <- length(classes) - row
    rect(column - 1, y, column, y + 1, col = heat_colors[color_index], border = "white")
    text(column - 0.5, y + 0.5, labels = value)
  }
}
axis(1, at = seq_along(classes) - 0.5, labels = plot_classes, las = 2)
axis(2, at = length(classes) - seq_along(classes) + 0.5, labels = plot_classes, las = 2)

plot(
  c(0, 1), c(0, 1), type = "n", xlim = c(0, 1), ylim = c(0, 1),
  xaxs = "i", yaxs = "i", asp = 1,
  xlab = "False positive rate", ylab = "True positive rate",
  main = "One-vs-rest ROC"
)
abline(0, 1, col = "gray60", lty = 2)
for (i in seq_along(classes)) {
  lines(roc_curves[[i]]$fpr, roc_curves[[i]]$tpr, col = plot_colors[i], lwd = 2)
}
legend(
  "bottomright",
  legend = paste0(plot_classes, " (AUC ", sprintf("%.3f", auc_by_class), ")"),
  col = plot_colors, lwd = 2, cex = 0.8
)

plot(
  c(0, 1), c(0, 1), type = "n", xlim = c(0, 1), ylim = c(0, 1),
  xaxs = "i", yaxs = "i", xlab = "Mean predicted probability",
  ylab = "Observed frequency", main = "One-vs-rest calibration"
)
abline(0, 1, col = "gray60", lty = 2)
for (i in seq_along(classes)) {
  class_bins <- calibration_bins[calibration_bins$class == classes[i], ]
  lines(
    class_bins$mean_confidence,
    class_bins$observed_frequency,
    type = "b", col = plot_colors[i], lwd = 2, pch = 16
  )
}
legend("topleft", legend = plot_classes, col = plot_colors, lwd = 2, pch = 16, cex = 0.8)

bar_positions <- barplot(
  f1, names.arg = plot_classes, ylim = c(0, 1),
  col = plot_colors, ylab = "F1 score", main = "Per-class test F1"
)
text(bar_positions, f1, labels = sprintf("%.3f", f1), pos = 3, cex = 0.85)
dev.off()

cat("R evaluation complete for", nrow(predictions), "held-out test images.\n")
cat("Split manifest check: all predictions match the test split.\n")
cat("Accuracy:", sprintf("%.2f%%", 100 * accuracy),
    "(stratified bootstrap 95% CI", sprintf("%.2f%%–%.2f%%", 100 * accuracy_ci[1], 100 * accuracy_ci[2]), ")\n")
cat("Macro F1:", sprintf("%.4f", macro_f1),
    "(stratified bootstrap 95% CI", sprintf("%.4f–%.4f", macro_f1_ci[1], macro_f1_ci[2]), ")\n")
cat("Macro one-vs-rest AUC:", sprintf("%.4f", macro_auc), "\n")
cat("Evaluation files saved under:", normalizePath(output_dir, winslash = "/"), "\n")
