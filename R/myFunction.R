
library(tumopp)
library(tidyverse)
library(gplots)

my.dist<-function(x){
  dist(x, method = "manhattan")
}
my.hclust<-function(x){
  #hclust(x,method="ward")
  hclust(x,method="ward.D2")
}
my.col <- function(n){
  colorpanel(n,high="red",low="gray95")
}
sortMat<-function(M){
  M2<-M
  M2[M2>0]<-1  
  mutCount<-apply(M2,1,sum)
  founder<-sort(names(which(mutCount==ncol(M))) )
  shared<-names(which(mutCount>1 & mutCount<ncol(M))) 
  unique<-names(which(mutCount==1))
  
  if(length(shared) == 0 & length(unique) == 0){
    return(M) 
  }
  
  Fo<-M[founder,,drop=FALSE]
  S<-M[shared,,drop=FALSE]
  U<-M[unique,,drop=FALSE]
  
  #sort founder row
  if(nrow(Fo) > 1){
    fo0<-rownames(Fo)[apply(Fo,1,mean)>=0.99]
    fo1<-rownames(Fo)[apply(Fo,1,mean)<0.99]
    Fo<-M[names(rev(sort(apply(Fo,1,mean)))),]
    Fo0 <- Fo[fo0,,drop=FALSE]
    if(length(fo1) > 1){
      Fo1<-M[names(rev(sort(apply(Fo[fo1,],1,mean)))),]
    }else if (length(fo1) == 1) {
      Fo1<-Fo[fo1,,drop=FALSE]
    }else{
      Fo1 <- NULL
    }
    Fo<-rbind(Fo0,Fo1)
  }
  
  #sort shared row
  
  if(nrow(S) > 1){
    h<-my.hclust(my.dist(S))
    tmp<-(h$labels)[h$order]
    S<-S[tmp,]
    #if(mean(apply(S[1:as.integer(nrow(S)/2),],1,mean)) <  mean(apply(S[(nrow(S)-as.integer(nrow(S)/2)):nrow(S),],1,mean))){
    if(mean(S[1,]) <  mean(S[nrow(S),])){
      S<-S[rev(tmp),]
    }
  }
  
  #get sorted col 
  M3<-rbind(Fo,S,U)
  h<-heatmap(M3, scale="none", col=c("gray95", my.col(100)), Rowv=NA, distfun=my.dist, hclustfun = my.hclust)
  sortCol<-colnames(M3)[h$colInd]
  
  Fo<-Fo[,sortCol,drop=FALSE]
  S<-S[,sortCol,drop=FALSE]
  U<-U[,sortCol,drop=FALSE]
  
  #sort unique row
  if(nrow(U) > 1){
    U<-as.matrix(U[do.call(order, as.data.frame(U)),]) 
    U<-U[nrow(U):1,]
  }
  
  M4<-rbind(Fo,S,U)
  #M4<-M4[nrow(M4):1,]
  return(M4)
}

getVafMat<-function(result, numberOfRegion=8, numberOfCell=100){
  population = result$population[[1]]
  extant = population %>% tumopp::filter_extant()
  graph = tumopp::make_igraph(population)
  regions = tumopp::sample_uniform_regions(extant,numberOfRegion, numberOfCell)
  subgraph = tumopp::subtree(graph, purrr::flatten_chr(regions$id))
  mutated = tumopp::mutate_clades(subgraph, mu = 1)
  vaf = tally_vaf(regions$id, mutated$carriers)
  vaf_tidy = vaf %>%
    tumopp::filter_detectable(0.05) %>%
    tumopp::sort_vaf() %>%
    tumopp::tidy_vaf() 
  n_row<-length(levels(factor(vaf_tidy$site)))
  n_col<-length(levels(factor(vaf_tidy$sample)))
  vaf_mat = matrix(0, nrow=n_row, ncol=n_col)
  for(i in 1:nrow(vaf_tidy)){
    vaf_mat[vaf_tidy$site[i],vaf_tidy$sample[i]] <- vaf_tidy$frequency[i]
  }
  rownames(vaf_mat)<-levels(factor(vaf_tidy$site))
  colnames(vaf_mat)<-levels(factor(vaf_tidy$sample))
  vaf_mat<-sortMat(vaf_mat)
  return(vaf_mat)
}


resizeImageMatrix<- function(M,X=8,Y=256) {
  if(ncol(M) > X){
    M<-sortMat(M[,sample(1:ncol(M),X)])
  }
  N <- matrix(rep(0,X*Y), nrow =Y, ncol=X )
  y<- 1:(Y+1)
  y2 = 0.5 + nrow(M)*y/(Y+1)
  N <- M[round(y2[1:Y]), ]
  if(ncol(N) < X){
    N <- cbind(N, matrix(0, ncol = X-ncol(N), nrow = nrow(N)))
  }
  return(N)
}