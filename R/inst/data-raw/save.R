cocaine <- read.table("./inst/data-raw/cocaine.txt", header = TRUE, 
                      stringsAsFactors = FALSE)
save(cocaine, file = "./data/cocaine.rda")

prostate <- read.table("./inst/data-raw/prostate.txt", header = TRUE, 
                      stringsAsFactors = FALSE)
save(prostate, file = "./data/prostate.rda")
