###########################################################################################
#                                                                                         #
#  This module downloads files from the internet and saves them                           #
#    Inputs:                                                                              #
#      imgDF - a dataframe containing                                                     #
#        $imgURLs - a list of URLs to download images from                                #
#        $imgName - a list of names to save the files as                                  #
#      NewDir - directory to save images to                                               #
#      maximages - maximum number of images to download.  Can be a numeric or NA          #
#      FileFailure - a list of failed URLs, for error tracking                            #
#    Outputs:                                                                             #
#      FileFailure - all failed URLs, appended to input list                              #
#                                                                                         #
###########################################################################################
download.images <- function(imgDF,NewDir,maximages,FileFailure){
  # Check if max images is set and, if so, take a random sample from the input
  if(!is.na(maximages)&&nrow(imgDF)>maximages){
    imgDF <- imgDF[sample(nrow(imgDF), maximages), ]
    rownames(imgDF) <- seq(length=nrow(imgDF))
  }
  for(j in 1:nrow(imgDF)){
    cat(paste0('File ',j,' of ',nrow(imgDF),'\n'))
    # Check if the file exists already.  If it does, no need to try a download
    if(!file.exists(file.path(NewDir,imgDF$imgName[j]))){
      # The downloading sometimes fails, with an error.  Therefore, the download command
      # is wrapped in a tryCatch, to prevent the script stopping on error.
      # All subsections end by returning the current file failure list, with the current
      # URL being appended to the failure list in the event of an error.
      FileFailure <- tryCatch({
        if(http_status(GET(imgDF$imgURLs[j]))$category=='Success'){
          download.file(imgDF$imgURLs[j],
                        file.path(NewDir,imgDF$imgName[j]),
                        mode = 'wb')
          FileFailure
        } else {
          # There is a problem with the URL.  Save the URL to the URLFailure list
          FileFailure <- c(FileFailure,imgDF$imgURLs[j])
          cat(paste0('Could not find URL: ',imgDF$imgURLs[j],'\n'))
          FileFailure
        }
      },error=function(e){
        # The code failed with an error.  This is probably due to the URL,
        # but could be a save issue.
        cat(paste0('Could not save from URL: ',imgDF$imgURLs[j],'\n\n'))
        c(FileFailure,imgDF$imgURLs[j])
      })
    }
  }
  # return the failure list
  FileFailure
}