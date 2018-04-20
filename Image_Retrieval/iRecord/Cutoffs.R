## Function to remove images over the max cutoff
cutoff.max <- function(AllTaxaDF){
  ## Sort AllTaxa by species name, then by media count to get the widest range of
  ## images, and finally by a random value to avoid biases
  AllTaxaDF$random <- runif(length(AllTaxaDF$preferred_taxon), 0, 100)
  AllTaxaDF <- AllTaxaDF[order(AllTaxaDF$preferred_taxon,
                               AllTaxaDF$media_count,
                               AllTaxaDF$random),]
  AllTaxaDF$random <- NULL

  ## Set up the variables
  TmpSpecies  <- AllTaxaDF$preferred_taxon[1]
  TmpOccCount <- 0
  removelist  <- c()
  TmpLength <- length(AllTaxaDF$id)

  ## Loop through records and generate a list to get rid of unnecessary records
  for(i in 1:TmpLength){
    if(TmpSpecies == AllTaxaDF$preferred_taxon[i]){
      TmpOccCount <- TmpOccCount+1
    } else {
      TmpSpecies <- AllTaxaDF$preferred_taxon[i]
      TmpOccCount <- 1
    }
    if(TmpOccCount > CutoffmaxNum){
      ## We're over the max cutoff, or we're under the max cutoff
      removelist <- c(removelist,i)
    }
    print(paste('Checking',i,'of',TmpLength,'for images over cutoff'))
  }
  print(paste0('Pre-cutoff length = ',TmpLength,
               ', post cutoff length = ',TmpLength - length(removelist)))
  ## Return the remove list
  removelist
}
