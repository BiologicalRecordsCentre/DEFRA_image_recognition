## Function to reformat the AllTaxa dataframe for use in generating jpgs
all.taxa.reformat <- function(AllTaxaDF){
  ## First, replace awkward / symbols with underscores
  AllTaxaDF$preferred_taxon <- str_replace_all(AllTaxaDF$preferred_taxon,"\\/","_")
  ## Remove anything in ()'s
  AllTaxaDF$preferred_taxon <- gsub("\\([^\\)]*\\) ","",
                                    AllTaxaDF$preferred_taxon,perl=TRUE)
  
  ## Create a vector with TRUE for all preferred_taxon names of number of words != 1
  GenusLevel <- sapply(strsplit(AllTaxaDF$preferred_taxon,' '),
                       FUN = function(x){length(x)!=1})
  ## Keep just entries, removing all entries with only one word (genus level entries)
  AllTaxaDF <- AllTaxaDF[GenusLevel,]
  
  ## Strip out spaces and replace with %20 for URL paths
  AllTaxaDF$path <- str_replace_all(AllTaxaDF$path,' ','%20')
  
  ## Create a vector of all preferred_taxon names, returning only the first two words
  ##  Note: this is a bit messy.  It works fine for butterfly taxa in iRecord, but 
  ##  there's no gurantee this will always work, if name format varies significantly
  SpeciesLevel <- sapply(strsplit(AllTaxaDF$preferred_taxon,' '),FUN = function(x){
    paste(x[1:2],collapse = ' ')
  })
  ## Add this to the AllTaxaDF data frame to use for saving files
  AllTaxaDF <- data.frame(AllTaxaDF,SpeciesLevel)
  
  ## Create a new column which concatenates Species and Status
  AllTaxaDF$Species_Status <- paste0(AllTaxaDF$SpeciesLevel,'_',AllTaxaDF$record_status)
  
  ## Remove pesky .'s from preferred taxon names
  AllTaxaDF$preferred_taxon <- gsub('\\.','',AllTaxaDF$preferred_taxon)
  AllTaxaDF$preferred_taxon <- gsub('\\ ','_',AllTaxaDF$preferred_taxon)
  
  ## Pass back the dataframe
  AllTaxaDF
}