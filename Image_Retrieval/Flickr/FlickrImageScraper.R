############################################################################################
#                                                                                          #
#  Script to read in a search csv, and look for images on Flickr that match the query      #
#                                                                                          #
############################################################################################
#
# setwd("W:/PYWELL_SHARED/Pywell Projects/BRC/Tom August/DEFRA_image_recognition")

source(file.path('.','Image_Retrieval','Flickr','FlickR.R'))
library(devtools)
install_github('FrancescaMancini/FlickrAPI_EABhackathon')
library(flickr)
library(httr)
library(RCurl)

# Set save directory
savelocation <- file.path('.','Image_Retrieval','Flickr')

# Set a maximum number of images: e.g. maximages = 100 OR maximages = NA
maximages = Inf

# Set the year range to search for images.  This year can be entered as:
#    format(Sys.Date(), "%Y")
startYear <- 1990
endYear   <- format(Sys.Date(), "%Y")
yearRange <- c(startYear,endYear)

# Set where you're searching for images.  Good options are Scotland, UK, Europe, or World
location <- 'UK'

# Authenticate with Flickr.  This is needed to allow you to query the Flickr database
# Run the below command, and a browser should launch for you to enter your Flickr details
# This only needs to be done once
# Currently this is commented out to prevent it running every time this script is run
# authFlickr()

# This section finds the relevant ID number for location chosen, and outputs the name
woeResult <- findPlaces(location)
cat(paste0('Location chosen: ',woeResult$woe_name[1]))

# Read in the search csv
pathtocsvs <- file.path('.','Image_Retrieval','Flickr')
csvs <- list.files(pathtocsvs,'\\.csv')

FileError <- c()
if(length(csvs)==0){
  cat('This script requires a search csv. Please create one and try again\n')
} else if(length(csvs)>1){
  cat('This script requires exactly 1 search csv. Please delete extra csvs and try again\n')
} else {
  searchDF <- read.csv(file.path(pathtocsvs,csvs),stringsAsFactors = FALSE)
  FileError <- download.flickr(searchDF,savelocation,maximages,
                               woeResult = NULL,yearRange,FileError)
}

# Print a completion message
if(is.null(FileError)){
  cat('All files downloaded successfully\n')
} else {
  cat('Some files failed.  Have a look at FileFailure for failed URLs\n')
}
