# 1. Initalisieren ---------------------------------
# 1a) Clear Enviorment ---------------------------------
rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc()

# 1b) Options ---------------------------------
options(java.parameters = c("-XX:+UseConcMarkSweepGC", "-Xmx100g"))
options(scipen=999)
options(useFancyQuotes = FALSE)

# 1c) Packages ---------------------------------
library(jsonlite)
library(data.table)
library(rstudioapi)
library(lubridate)
library(magrittr)
library(openxlsx)

# 1d) Directories und Output---------------------------------
# working_dir <- dirname(getActiveDocumentContext()$path)
git <- "C:/Users/CH00HAG/git/parcaster/data/scraping/parking-data-sg/"
path <- paste0("O:/Group/TC/Daten_Programme/USERS/HAG/ML4SE/Praxisprojekt/Parkplatzdaten/")

# Rohdaten
# file_2022.rohdaten <- paste0(path, "rohdaten/PLS_2022.csv")
# file_2023.rohdaten <- paste0(path, "rohdaten/PLS_2023.csv")

# Input
inputpath <- paste0(path, "input/")
file_2019_2020 <- paste0(inputpath, "PLS_2019_2020.txt")
file_2021 <- paste0(inputpath, "PLS_2021.csv")
file_2022 <- paste0(inputpath, "pp_2022.csv")
file_2023 <- paste0(inputpath, "pp_2023.csv")

# Live-Daten
file_2023_aktuell <- paste0(inputpath, "pp_2023_aktuell.csv")

# Output
outputpath <- paste0(git, "final/")

# Features
calendar <- paste0(path, "features/features.xlsx")

# 2. JSON to Table -----------------------------------------
# 2.1 Jahr 2022  -----------------------------------------
# data_2022 <- read.csv2(file_2022.roh, sep = ",", header = FALSE, stringsAsFactors = FALSE, encoding="UTF-8")
# 
# # Transform JSON into a list
# data_2022.list <- lapply(data_2022$V1, function(x) fromJSON(x))
# 
# for(i in 1:length(data_2022.list)){
#   tmp <- as.data.table(data_2022.list[[i]][["records"]])
#   
#   if(i == 1){pp_2022 <- tmp} else {pp_2022 <- rbind(pp_2022, tmp)}
# }
# 
# pp_2022.final <- data.frame(lapply(pp_2022, as.character), stringsAsFactors=FALSE)
# 
# write.csv(pp_2022.final, file=paste0(outputpath, "pp_2022.csv"), row.names=FALSE, fileEncoding = "UTF-8")

# 2.2 Jahr 2023  -----------------------------------------
# data_2023 <- read.csv2(file_2023.rohdaten, sep = ",", header = FALSE, stringsAsFactors = FALSE, encoding="UTF-8")
# 
# # Transform JSON into a list
# data_2023.list <- lapply(data_2023$V1, function(x) fromJSON(x))
# 
# for(i in 1:length(data_2023.list)){
#   tmp <- as.data.table(data_2023.list[[i]][["records"]])
#   
#   if(i == 1){pp_2023 <- tmp} else {pp_2023 <- rbind(pp_2023, tmp)}
# }
# 
# pp_2023.final <- data.frame(lapply(pp_2023, as.character), stringsAsFactors=FALSE)
# 
# write.csv(pp_2023.final, file=paste0(outputpath, "pp_2023.csv"), row.names=FALSE, fileEncoding = "UTF-8")

# 3. Load Data -----------------------------------------
# 3.1 Jahre 2019 bis 2020 -----------------------------------------
pp_2019_2020 <- data.table(read.table(file_2019_2020, sep=";", header = TRUE, encoding="UTF-8"))
pp_2019_2020_00 <- pp_2019_2020[, datetime := as_datetime(Datum, format = "%d/%m/%Y %H:%M")]

# 3.2 Jahr 2021 -----------------------------------------
pp_2021 <- data.table(read.table(file_2021, sep=",", header = TRUE, encoding="UTF-8"))
pp_2021_00 <- pp_2021[, datetime := as_datetime(Datum, format = "%Y-%m-%d %H:%M:%S")]

# 3.2 Jahr 2022 -----------------------------------------
pp_2022 <- data.table(read.table(file_2022, sep=",", header = TRUE, encoding="UTF-8"))
pp_2022_00 <- as.data.table(pp_2022)[, datetime := as_datetime(fields.zeitpunkt)]

# 3.4 Jahr 2023  -----------------------------------------
pp_2023 <- data.table(read.table(file_2023, sep=",", header = TRUE, encoding="UTF-8"))
pp_2023_00 <- as.data.table(pp_2023)[, datetime := as_datetime(fields.zeitpunkt)]

# 3.5 Jahr 2023 aktuell -----------------------------------------
pp_2023_aktuell <- data.table(read.table(file_2023_aktuell, sep=",", header = TRUE, encoding="UTF-8"))
pp_2023_aktuell_00 <- as.data.table(pp_2023_aktuell)[, datetime := as_datetime(date)] %>% 
  .[, -c("date")]

# 4. Code-Tabelle Parkhaeuser -----------------------------------------
code_pp <- unique(pp_2023_00[, .(fields.phid, fields.phname)], by = c("fields.phid", "fields.phname"))

#Alte Jahre umbenennen
setnames(pp_2019_2020_00, sub(" ", ".", code_pp$fields.phname), code_pp$fields.phid)
setnames(pp_2021_00, sub(" ", ".", code_pp$fields.phname), code_pp$fields.phid)


# 5. Duplikate entfernen -----------------------------------------
## Ueber gesamten Datensatz
pp_2019_2020_01 <- unique(pp_2019_2020_00)
pp_2021_01 <- unique(pp_2021_00)
pp_2022_01_a <- setorder(unique(pp_2022_00[, -c("record_timestamp")]), cols = "fields.shortfree")
pp_2023_01_a <- setorder(unique(pp_2023_00[, -c("record_timestamp")]), cols = "fields.shortfree")

## Ueber gesamten timestamp und fields.phid (behalte kleiner Anzahl frei)
pp_2022_01 <- unique(pp_2022_01_a, by = c("datetime", "fields.phid"))
pp_2023_01 <- unique(pp_2023_01_a, by = c("datetime", "fields.phid"))
pp_2023_aktuell_01 <- unique(pp_2023_aktuell_00, by = c("datetime"))

# 6. Mergen -----------------------------------------
## Neue Daten transponieren
pp_2022_02 <- dcast(pp_2022_01, datetime ~ fields.phid, value.var = "fields.shortfree", fill=0)
pp_2023_02 <- dcast(pp_2023_01, datetime ~ fields.phid, value.var = "fields.shortfree", fill=0)

# Drop Cols
pp_2019_2020_02 <- pp_2019_2020_01[, -c("Summe", "X..frei", "X.belegt", "Datum")]
pp_2021_02 <- pp_2021_01[, -c("Datum")]

pp <- rbind(pp_2019_2020_02
          , pp_2021_02
          , pp_2022_02
          , pp_2023_02
          , pp_2023_aktuell_01)

# 7. Calendar-Features -----------------------------------------
calendar.features <- data.table(read.xlsx(calendar, colNames = TRUE)) %>% 
  .[, date := as.Date(date, origin = "1899-12-30")]

# Mergen
pp_00 <- pp[, date := as.Date(datetime)] %>% 
  .[calendar.features, `:=` (ferien = i.ferien
                          , feiertag = i.feiertag
                          , covid_19 = i.covid_19
                          , olma_offa = i.olma_offa), on = .(date = date)] %>% 
  .[, -c("date")]

# 8. Weather -----------------------------------------
pp.final <- pp_00

# 9. Final Dataset -----------------------------------------
write.csv(pp.final, file=paste0(outputpath, "pp_sg.csv"), row.names=FALSE, fileEncoding = "UTF-8")
