############################################################################################
#                                                                                          #
#  Script to scrape photos from Flickr.  Called by ImageRetriever.R                        #
#    Inputs:                                                                               #
#      searchDF - a dataframe containing                                                   #
#        $Target - the general name for the thing you're searching for                     #
#        $SearchTerm1,$SearchTerm2,...,$SearchTerm10 - the terms used to search flickr     #
#      maximages - maximum number of images to download.  Can be a numeric or NA           #
#      woeResult - the DF resulting from calling findPlaces in the                         #
#                  FlickrAPI_EABhackathon/flickr package                                   #
#      FileError - a list of failed URLs, for error tracking                               #
#    Outputs:                                                                              #
#      FileError - all failed URLs, appended to input list                                 #
#                                                                                          #
############################################################################################
download.flickr <- function(searchDF,savelocation,maximages,woeResult,yearRange,FileError){
  #  Include the downloader module                                                         #
  source(file.path('.','Image_Retrieval','ImageDownloader.R'))

  # Count the number of search terms for outputting to screen
  numSearchTerms <- sum(sapply(searchDF[,-1], function(x) sum(!is.na(x)&x!='')))
  counter <- 1
  
  # Loop through targets using all the search terms, putting all images in one folder
  for(j in 1:nrow(searchDF)){
    searchterms <- searchDF[searchDF$Target==searchDF$Target[j],2:ncol(searchDF)]
    searchterms <- searchterms[!is.na(searchterms)&searchterms!='']

    # Set the save directory for this target
    NewDir <- file.path(savelocation,paste0(searchDF$Target[j],' - Flickr'))
    dir.create(NewDir,showWarnings = FALSE)
    
    # Now the script loops through each search term for the target, downloading as it goes
    for(i in searchterms){
      cat(paste0('Searching for ',searchDF$Target[j],': ',i,
                 ', Search term ',counter,' of ',numSearchTerms,'\n'))
      counter <- counter+1
      # Call the photosSearch function from the flickr package
      photoDF <- photosSearch(year_range = yearRange,
                              text = gsub(' ','\\+',i), 
                              woe_id = as.character(woeResult$woe_id[1]))
      if(is.null(photoDF)){
        cat('No images for',i,'for year range',yearRange[1],'-',yearRange[2],
            'in',as.character(woeResult$woe_name[1]),'\n\n')
      } else {
        # Generate a list of URLs based on the highest resolution file first, then working
        # down the list
        oList   <- photoDF[!is.na(photoDF$url_o),]$url_o
        lList   <- photoDF[is.na(photoDF$url_o)&(!is.na(photoDF$url_l)),]$url_l
        mList   <- photoDF[is.na(photoDF$url_l)&(!is.na(photoDF$url_m)),]$url_m
        sList   <- photoDF[is.na(photoDF$url_m)&(!is.na(photoDF$url_s)),]$url_s
        imgURLs <- c(oList,lList,mList,sList)
        
        # Set up the file names based on everything after the last / symbol
        imgName <- unlist(regmatches(imgURLs,
                                     regexpr('(?<=\\/)[^\\/]*$',imgURLs,perl=TRUE)))

        # Save URLs and names as a dataframe for sending to the image downloader function
        imgDF <- data.frame(imgURLs,imgName,stringsAsFactors = FALSE)
        
        # Now, download the files
        TmpFileError <- download.images(imgDF,NewDir,maximages,FileError)
        FileError    <- c(FileError,TmpFileError)
      }
    }
  }
  # Return the failure list
  FileError
}