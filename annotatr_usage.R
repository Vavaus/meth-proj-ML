library('annotatr')
library('GenomicRanges')

#-------Data for genomic ranges conversion-------#
df <- read.csv('for_granges.csv')
df <- df[c(2,3,4)]

#-------Annotatr magic-------#
annotatr::builtin_annotations() # All possible annotations

# List of annotations that we are interested in
annots = c('mm10_cpgs','mm10_basicgenes', "mm10_enhancers_fantom") 

# Building annotations
anns <- build_annotations(genome = "mm10", annotations = annots)
anns
# Convert our CpGs as GRanges object
df_ranges <- as(df,"GRanges")

# Intersection of annotations
result <- annotate_regions(regions = df_ranges, annotations = anns, quiet = F)

# Write to .csv
write.csv(result, 'res_annotatr.csv')

# Plotting annotation results
plot_annotation(annotated_regions = result)
