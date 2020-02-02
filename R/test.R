library(tumopp)
library(tidyverse)
library(gplots)

tumoppArgs<-"-D3　-Chex　-N10000　-k10　-Lconst　-Pmindrag"
numberOfRegion<-8
numberOfCell<-100
result <- tumopp(tumoppArgs,numberOfRegion)
population = result$population[[1]]
extant = population %>% tumopp::filter_extant()
graph = tumopp::make_igraph(population)
regions = tumopp::sample_uniform_regions(extant,numberOfRegion, numberOfCell)
subgraph = tumopp::subtree(graph, purrr::flatten_chr(regions$id))
mutated = tumopp::mutate_clades(subgraph, mu = 1)
vaf = tally_vaf(regions$id, mutated$carriers) %>% print()
