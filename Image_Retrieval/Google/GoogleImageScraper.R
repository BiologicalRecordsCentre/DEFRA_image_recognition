############################################################################################
#                                                                                          #
# This script is to scrape images from google image search                                 #
#                                                                                          #
# To use it, you need to:                                                                  #
#   Open up a browser                                                                      #
#   Search for whatever you want on google                                                 #
#   Click the image option at the top to see image results                                 #
#   Scroll down to the very bottom, clicking on 'Load More Images' as you go               #
#   Save as 'Webpage, Complete' (it must be this format to preserve full URL links)        #
#     NOTE: DO NOT rename the file.  Leave it as the default                               #
#   Repeat for as many searches as you wish                                                #
#   Enter the location of your saved files here:                                           #

savelocation <- file.path('.','Image_Retrieval','Google')

# Set a maximum number of images: e.g. maximages = 100 OR maximages = NA                   #

maximages = NA

#   Run the code.                                                                          #
#                                                                                          #
# This script will create subfolders for each search, and drop the images in there         #
#                                                                                          #
############################################################################################
library('RCurl')
library('httr')
source(file.path('.','Image_Retrieval','ImageDownloader.R'))

# First, find all the html files
htmls <- list.files(savelocation,pattern='\\.html')

# Set up a list to track failed URLs
FileError <- c()

# Loop through the html files, downloading all images to separate folders
for(i in htmls){
  # First, delete the html files folder, as we don't need it
  unlink(file.path(savelocation,gsub('\\.html','_files',i)), recursive = TRUE)

  # Read in the html file
  tmp     <- readLines(file.path(savelocation,i))
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
  NewDir  <- unlist(regmatches(i,gregexpr('.+(?= Search)',i,perl=TRUE)))
  NewDir  <- file.path(savelocation,NewDir)
  dir.create(NewDir,showWarnings = FALSE)
  
  # Strip down file names (including absolute path) greater than 250 characters,
  # as we're getting close to max file length in Windows
  loclength <- nchar(file.path(normalizePath(savelocation),NewDir))
  longlist  <- imgName[(nchar(imgName)+loclength+1)>250]
  imgName[(nchar(imgName)+loclength+1)>250] <-
    substr(longlist,nchar(longlist)+loclength-249,nchar(longlist))
  
  # Save these URLs and names as a dataframe for sending to the image downloader function
  imgDF <- data.frame(imgURLs,imgName,stringsAsFactors = FALSE)
  
  # Now, download the files
  TmpFileError <- download.images(imgDF,NewDir,maximages,FileError)
  FileError    <- c(FileError,TmpFileError)
  
  # Correctly saved htmls will still throw a warning, so just let the user know this isn't
  # anything to worry about
  cat("Don't worry about",'"incomplete final line" error messages\n\n')
}

# Print a completion message
if(is.null(FileError)){
  cat('All files downloaded successfully\n')
} else {
  cat('Some files failed.  Have a look at FileError for failed URLs\n')
}
