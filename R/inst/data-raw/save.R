cocaine_dependence <- read.table("./inst/data-raw/cocaine_dependence.txt", header = TRUE, 
                                 stringsAsFactors = FALSE)
save(cocaine_dependence, file = "./data/cocaine_dependence.rda")

prostate <- read.table("./inst/data-raw/prostate.txt", header = TRUE, 
                      stringsAsFactors = FALSE)
save(prostate, file = "./data/prostate.rda")
