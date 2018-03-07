############################################################################################
#                                                                                          #
# This script is to scrape images from google image search                                 #
#                                                                                          #
# To use it, you need to:                                                                  #
#   Open up a browser                                                                      #
#   Search for whatever you want on google                                                 #
#   Click the image option at the top to see image results                                 #
#   Scroll down to the very bottom, clicking on 'Load More Images' as you go               #
#   Save as 'Webpage, Complete' (it must be this format to preserve full URL links         #
#     NOTE: DO NOT rename the file.  Leave it as the default                               #
#   Repeat for as many searches as you wish                                                #
#   Enter the location of your saved files here:                                           #

savelocation <- '.'

#   Run the code.                                                                          #
#                                                                                          #
# This script will create subfolders for each search, and drop the images in there         #
#                                                                                          #
############################################################################################
library('RCurl')
library('httr')

# First, find all the html files
htmls <- list.files(savelocation,pattern='\\.html')

# Set up a couple of files to track failed URLs
URLFailure <- FileFailure <- c()

# Loop through the html files, downloading all images to separate folders
for(i in htmls){
  # Read in the html file
  tmp     <- readLines(i)
  # All URLs start 'imgres\\imgurl=' and end '&amp'.
  # These bookends are used to find the URLs.
  imgres  <- unlist(regmatches(tmp,
                               gregexpr('(?<=imgres\\?imgurl=).+?(?=\\&amp)',
                                        tmp,
                                        perl=TRUE)))
  # Keep only jpg files.  There's a few other formats which just don't work.
  imgres  <- imgres[grepl('(?i).+?\\.jp[e]?g',imgres,perl=TRUE)]
  # Decode the html to a link that the file downloader function can understand
  imgURLs <- unlist(lapply(imgres, function(x) URLdecode(x)))
 
  # Create filenames using the html full name, stripping off everything after .jpg
  # Some files have pixel sizes or other info after the .jpg extension, which mess
  # up Windows' understanding of what the file type is
  imgName <- unlist(regmatches(imgres,regexpr('(?i).+?\\.jp[e]?g',imgres,perl=TRUE)))

  # Create a directory for these images using the search term from the file name
  NewDir  <- unlist(regmatches(i,gregexpr('.+(?= - Google Search)',i,perl=TRUE)))
  dir.create(file.path(savelocation,NewDir),showWarnings = FALSE)
  
  # Strip down file names (including absolute path) greater than 250 characters,
  # as we're getting close to max file length in Windows
  loclength <- nchar(file.path(normalizePath(savelocation),NewDir))
  longlist  <- imgName[(nchar(imgName)+loclength+1)>250]
  imgName[(nchar(imgName)+loclength+1)>250] <-
    substr(longlist,nchar(longlist)+loclength-249,nchar(longlist))
  
  # Now, download the files
  for(j in 1:length(imgURLs)){
    cat(paste0('File ',j,' of ',length(imgURLs),'\n'))
    # Check if the file exists already.  If it does, no need to try a download
    if(!file.exists(file.path(savelocation,NewDir,imgName[j]))){
      # The downloading sometimes fails, with an error.  Therefore, the download command
      # is wrapped in a tryCatch, to prevent exiting on error.
      tryCatch({
        if(http_status(GET(imgURLs[j]))$category=='Success'){
          download.file(imgURLs[j],file.path(savelocation,NewDir,imgName[j]),mode = 'wb')  
        } else {
          # There is a problem with the URL.  Save the URL to the URLFailure list
          URLFailure <- c(URLFailure,imgURLs[j])
          cat(paste0('Could not find URL: ',imgURLs[j],'\n'))
        }
      },error=function(e){
        # The code failed with an error.  This is probably due to the URL,
        # but could be a save issue.
        FileFailure <- c(FileFailure,imgURLs[j])
        cat(paste0('Could not save from URL: ',imgURLs[j],'\n\n'))
      })
    }
  }
}