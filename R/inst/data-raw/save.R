# cocaine dependence
cocaine_dependence <- read.table("./inst/data-raw/cocaine_dependence.txt", header = TRUE, 
                                 stringsAsFactors = FALSE)
colnames(cocaine_dependence) <- tolower(colnames(cocaine_dependence))
save(cocaine_dependence, file = "./data/cocaine_dependence.rda")

# prostate
prostate <- read.table("./inst/data-raw/prostate.txt", header = TRUE, 
                      stringsAsFactors = FALSE)
save(prostate, file = "./data/prostate.rda")
