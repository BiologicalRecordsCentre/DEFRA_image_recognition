#########################################################################################
#                                                                                       #
#  Code for collating all images from verified and unverified subfolders, and putting   #
#    them in one place, with a prefix of the taxa name                                  #
#  This code uses list.files and if statements rather than a recursive list.dirs        #
#    function. This is because the recursive function takes much, much longer           #
#                                                                                       #
#########################################################################################
collate.images <- function(inputfolder,collatefolder){
  ## First, find the list of files and directories in the input folder
  FileList <- list.files(inputfolder)
  if(!dir.exists(file.path('.',collatefolder))){
    dir.create(file.path('.',collatefolder))
  }
  
  ## Set variables for list of folders to delete when done with the files
  DeleteList   <- c()
  indeletelist <- FALSE
  
  ## Remove all non-folders
  FileList <- FileList[which(!grepl('\\.',FileList))]
  ## Loop through file and directory list
  for(i in FileList){
    ## We have a directory, find the sub-files
    SubFileList <- list.files(file.path(inputfolder,i))
    for(j in SubFileList){
      if(grepl('erified$',j)){
        ## This is an Unverified or Verified subfolder.  Find the images and move them.
        ImageList <- list.files(file.path(inputfolder,i,j))
        if(is.na(ImageList[1])){
          print(paste0('Cannot find images in folder ',file.path(inputfolder,i,j)))
        } else {
          print(paste0('Moving files from taxa folder ',file.path(inputfolder,i,j),', ',
                       match(i,FileList),' of ',length(FileList)))
          file.rename(file.path(inputfolder,i,j,ImageList),
                      file.path(collatefolder,paste0(i,'_',ImageList)))
        }
        ## Add directory to delete list, but only if we haven't done so already
        if(!indeletelist){
          DeleteList <- c(DeleteList,file.path(inputfolder,i))
          indeletelist <- TRUE
        }
      }
    }
    ## Reset delete list boolean
    indeletelist <- FALSE
  }
  ## Delete all the folders in the delete list
  print(paste('Now removing all folders.  There are',length(DeleteList),
              'to remove, and each 1000 folders takes about a minute'))
  unlink(DeleteList,recursive=T)
}
