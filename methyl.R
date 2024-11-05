library('tidyverse')
library('ggplot2')
library("genomation")
library("GenomicRanges")

df <- read.csv('./for_granges.csv')
df <- df[c(2,3,4)]
df

refseq_anot <- readTranscriptFeatures("./hgTables.txt")
refseq_anot

myDiff25p.hyper.anot <- annotateWithGeneParts(target = as(df,"GRanges"),
                                              feature = refseq_anot)

myDiff25p.hyper.anot

dist_tss <- getAssociationWithTSS(myDiff25p.hyper.anot)
head(dist_tss)

write.csv(dist_tss, './tss_dist.csv')

# See whether the differentially methylated CpGs are within promoters,introns or exons; the order is the same as the target set
inclusion <- getMembers(myDiff25p.hyper.anot)

write.csv(inclusion, './regions.csv')

# This can also be summarized for all differentially methylated CpGs
plotTargetAnnotation(myDiff25p.hyper.anot, main = "Differential Methylation Annotation")

cpg_anot <- readFeatureFlank("./cpg.txt", feature.flank.name = c("CpGi", "shores"), flank=2000)
diffCpGann <- annotateWithFeatureFlank(as(df,"GRanges"), feature = cpg_anot$CpGi, flank = cpg_anot$shores, feature.name = "CpGi", flank.name = "shores")

# See whether the CpG in myDiff25p belong to a CpG Island or Shore
head(getMembers(diffCpGann))

df_cpg <- getMembers(diffCpGann)
write.csv(df_cpg, './cpg_ann.csv')
# This can also be summarized for all differentially methylated CpGs
plotTargetAnnotation(diffCpGann, main = "Differential Methylation Annotation")
diffCpGann

dist_tss
