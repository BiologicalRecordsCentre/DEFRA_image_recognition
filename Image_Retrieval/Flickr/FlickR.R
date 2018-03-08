############################################################################################
#                                                                                          #
#  Script to scrape photos from Flickr                                                     #
#                                                                                          #
#  First, the libraries:                                                                   #
library(devtools)
install_github('FrancescaMancini/FlickrAPI_EABhackathon')
library(flickr)
source('./ImageDownloader.R')

# Then set save directory
savelocation <- '.'

# Set a maximum number of images: e.g. maximages = 100 OR maximages = NA
maximages = NA

# Set your search terms here, as a list
searchterms <- c('birds+foot+trefoil')

# Set where you're searching for images.  Good options are UK, Europe, or World
location <- 'Scotland'

# Authenticate with Flickr.  This is needed to allow you to query the Flickr database
# Run the below command, and a browser should launch for you to enter your Flickr details
authFlickr()

# This section finds the relevant ID number for location chosen, and outputs the name
woeResult <- findPlaces(location)
cat(paste0('Location chosen: ',woeResult$woe_name[1]))

# Now the script loops through each search term, downloading as it goes
FileFailure <- c()
counter <- 1
for(i in searchterms){
  cat(paste0('Searching for ',i,', Search term ',counter,' of ',length(searchterms),'\n'))
  counter <- counter+1
  # Call the photosSearch function from the flickr package
  photoDF <- photosSearch(year_range = c(1970, format(Sys.Date(), "%Y")),
                          text = i,
                          woe_id = as.character(woeResult$woe_id[1]))
  
  # Set the save directory
  NewDir <- file.path(savelocation,paste0(gsub('\\+',' ',i),' - Flickr'))
  #NewDir <- file.path(savelocation,paste0('Flickr ',i))
  dir.create(NewDir,showWarnings = FALSE)
  
  # Generate a list of URLs based on the highest resolution file first, then working down
  # the list
  oList   <- photoDF[!is.na(photoDF$url_o),]$url_o
  lList   <- photoDF[is.na(photoDF$url_o)&(!is.na(photoDF$url_l)),]$url_l
  mList   <- photoDF[is.na(photoDF$url_l)&(!is.na(photoDF$url_m)),]$url_m
  sList   <- photoDF[is.na(photoDF$url_m)&(!is.na(photoDF$url_s)),]$url_s
  imgURLs <- c(oList,lList,mList,sList)
  
  # Set up the file names based on everything after the last / symbol
  imgName <- unlist(regmatches(imgURLs,regexpr('\\/[^\\/]*$',imgURLs,perl=TRUE)))
  imgName <- substr(imgName,2,length(imgName))
  
  # Save these URLs and names as a dataframe for sending to the image downloader function
  imgDF <- data.frame(imgURLs,imgName,stringsAsFactors = FALSE)
  
  # Now, download the files
  TmpFileFailure <- download.images(imgDF,NewDir,maximages,FileFailure)
  FileFailure    <- c(FileFailure,TmpFileFailure)
}

# Print a completion message
if(is.null(FileFailure)){
  cat('All files downloaded successfully')
} else {
  cat('Some files failed.  Have a look at FileFailure for failed URLs')
}