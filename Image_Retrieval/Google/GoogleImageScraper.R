############################################################################################
#
# This script is to scrape images from google image search
#
# To use it, you need to:
#   Open up a browser
#   Search for whatever you want on google
#   Click the image option at the top to see image results
#   Scroll down to the very bottom, clicking on 'Load More Images' as you go
#   Save as 'Webpage, Complete' (it must be this format to preserve full URL links
#     NOTE: DO NOT rename the file.  Leave it as the default
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

# The code grabs the files in your save folder here, and finds all the html formatted files
filelist <- list.files(savelocation)
htmls <- filelist[grepl('\\.html',filelist)]

URLFailure <- FileFailure <- c()
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
  imgres  <- imgres[grepl('.+?\\.jp[e]?g',imgres,perl=TRUE)]
  # Decode the html to a link that the file downloader function can understand
  imgURLs <- unlist(lapply(imgres, function(x) URLdecode(x)))
  # Strip down file names greater than 250 characters, as Windows can't handle it
  longlist <- imgres[nchar(imgres)>250]
  imgres[nchar(imgres)>250] <- substr(longlist,nchar(longlist)-249,nchar(longlist))

  # Create filenames using the html full name, stripping off everything after .jpg
  # Some files have pixel sizes or other info after the .jpg extension, which mess
  # up Windows' understanding of what the file type is
  imgName <- unlist(regmatches(imgres,regexpr('.+?\\.jp[e]?g',imgres,perl=TRUE)))
  
  # Create a directory for these images using the search term from the file name
  NewDir  <- unlist(regmatches(i,gregexpr('.+(?= - Google Search)',i,perl=TRUE)))
  if(!dir.exists(file.path(savelocation,NewDir))){
    dir.create(file.path(savelocation,NewDir))
  }
  
  # Now, download the files
  for(j in 1:length(imgURLs)){
    print(paste0('File ',j,' of ',length(imgURLs)))
    # The downloading quite often fails, with an error.  Therefore, the download command
    # is wrapped in a tryCatch, to prevent exiting on error.
    tryCatch({
      if(http_status(GET(imgURLs[j]))$category=='Success'){
        if(!file.exists(file.path(savelocation,NewDir,imgres[j]))){
          download.file(imgURLs[j],file.path(savelocation,NewDir,imgres[j]),mode = 'wb')
        }
      } else {
        # There is a problem with the URL.  Save the URL to the URLFailure list
        URLFailure <- c(URLFailure,imgURLs[j])
        print(paste0('Could not find URL: ',imgres[j]))
      }
    },error=function(e){
      # The code failed with an error.  This is probably due to the URL,
      # but could be a save issue.
      FileFailure <- c(URLFailure,imgURLs[j])
      print(paste0('Could not save from URL: ',imgURLs[j]))
    })
  }
}
