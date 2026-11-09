packages <- c("tidyverse", "here", "viridis", "patchwork")
missing <- packages[!packages %in% rownames(installed.packages())]
if (length(missing) > 0) install.packages(missing)
message("R analysis packages ready. Consider renv::init() once plotting is stable.")
