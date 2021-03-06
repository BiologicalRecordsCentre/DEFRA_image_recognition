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
#                  This can be NULL if all images, with and without location data          #
#      FileError - a list of failed URLs, for error tracking                               #
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
                              woe_id = switch(is.null(woeResult) + 1,
                                              as.character(woeResult$woe_id[1]),
                                              NULL))
      if(is.null(photoDF)){
        cat('No images for',i,'for year range',yearRange[1],'-',yearRange[2],
            ifelse(is.null(woeResult), '', paste('in',as.character(woeResult$woe_name[1]))),'\n\n')
      } else {
        # Now, download the files
        if(!is.na(maximages) & maximages != Inf){
          photoDF <- photoDF[sample(x = 1:nrow(photoDF),
                             size =  maximages,
                             replace = FALSE), ]
        }
        
        downloadImages(photoSearch_results = photoDF,
                       licenses = 0:10,
                       max_quality = 2,
                       saveDir = NewDir)
      }
    }
  }
}