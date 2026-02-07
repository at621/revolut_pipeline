
library(PDtoolkit)
library(jsonlite)

# Read the shared dataset
df <- read.csv("C:/Users/Watson/AppData/Local/Temp/claude/c--projects-revolut-2/f14df8fe-6892-4d2b-a79a-20b7157297e3/scratchpad/cross_val_data.csv")
cat("R: Loaded data with", nrow(df), "rows\n")
cat("R: Default rate:", mean(df$target), "\n")
cat("R: Columns:", paste(names(df), collapse=", "), "\n")
cat("R: f1 bins:", paste(sort(unique(df$f1)), collapse=", "), "\n")
cat("R: f2 bins:", paste(sort(unique(df$f2)), collapse=", "), "\n")

# The data already has:
#   - target: binary 0/1
#   - pred: model predicted probabilities
#   - f1, f2: bin labels (character)

# Make sure risk factors are character type (PDtoolkit requirement)
df$f1 <- as.character(df$f1)
df$f2 <- as.character(df$f2)

# We need to create a "current model" that produces the pred column.
# Since we already have the predictions, we'll use the internal miv() function
# by providing the model formula and letting it predict.
# But we need to trick it -- the miv() function calls glm() internally.

# Instead, let's call woe.tbl and compute MIV manually, matching the
# PDtoolkit miv() function logic exactly.

results <- list()

for (rf_name in c("f1", "f2")) {
    cat("\n--- Computing MIV for", rf_name, "---\n")

    # Observed WoE table (using actual target)
    observed <- woe.tbl(tbl = df, x = rf_name, y = "target", y.check = TRUE)
    cat("Observed WoE table:\n")
    print(observed[, c("bin", "no", "ng", "nb", "woe")])

    # Expected WoE table (using predicted probabilities as target)
    expected <- woe.tbl(tbl = df, x = rf_name, y = "pred", y.check = FALSE)
    cat("Expected WoE table:\n")
    print(expected[, c("bin", "no", "ng", "nb", "woe")])

    # Merge observed and expected
    comm.cols <- c("bin", "no", "ng", "nb", "woe")
    miv.tbl <- merge(observed[, comm.cols],
                     expected[, comm.cols],
                     by = "bin",
                     all = TRUE,
                     suffixes = c(".o", ".e"))

    cat("Merged MIV table:\n")
    print(miv.tbl)

    # Compute MIV exactly as PDtoolkit does
    miv.tbl$delta <- miv.tbl$woe.o - miv.tbl$woe.e
    miv.val.g <- sum(miv.tbl$ng.o * miv.tbl$delta) / sum(miv.tbl$ng.o)
    miv.val.b <- sum(miv.tbl$nb.o * miv.tbl$delta) / sum(miv.tbl$nb.o)
    miv.val <- miv.val.g - miv.val.b

    cat("MIV components: miv.val.g =", miv.val.g, ", miv.val.b =", miv.val.b, "\n")
    cat("MIV =", miv.val, "\n")

    # Chi-square test
    m.chiq.g <- miv.tbl$ng.o * log(miv.tbl$ng.o / miv.tbl$ng.e)
    m.chiq.b <- miv.tbl$nb.o * log(miv.tbl$nb.o / miv.tbl$nb.e)
    m.chiq.gb <- m.chiq.g + m.chiq.b
    m.chiq.stat <- 2 * sum(m.chiq.gb)
    p.val <- pchisq(m.chiq.stat, nrow(miv.tbl) - 1, lower.tail = FALSE)

    cat("Chi-square stat =", m.chiq.stat, ", p-value =", p.val, "\n")

    results[[rf_name]] <- list(
        miv = miv.val,
        miv_g = miv.val.g,
        miv_b = miv.val.b,
        chisq_stat = m.chiq.stat,
        p_value = p.val,
        n_bins = nrow(miv.tbl)
    )
}

# Save results as JSON
json_out <- toJSON(results, auto_unbox = TRUE, pretty = TRUE)
writeLines(json_out, "C:/Users/Watson/AppData/Local/Temp/claude/c--projects-revolut-2/f14df8fe-6892-4d2b-a79a-20b7157297e3/scratchpad/r_miv_results.json")
cat("\nResults saved to JSON\n")
