scriptDir<-dirname(sys.frame(1)$ofile)
setwd(scriptDir)
latentFile <-"../result/latent.tab"
prmFile <-"../result/prm.tab"
outDir <-"../Rplot"
dir.create(outDir)

L<-as.matrix(read.table(latentFile))
L<-L[sample(rownames(L), 1000),]
P<-read.table(prmFile)
P<-P[rownames(P),]


library("Rtsne")
tsne <-  Rtsne(unique(L),perplexity = 30)
#plot(tsne$Y, pch=20, xlab="", ylab="")
x <- tsne$Y[,1]
y <- tsne$Y[,2]

library(RColorBrewer)
for(i in 1:ncol(P)){
  if(length(unique(P[,i]))==1){
    next
  }
  type <- colnames(P)[i]
  p <- P[,i]
  p_levels<-sort(unique(p))
  col_levels<-NULL
  if(is.numeric(p)){
    col_levels<- brewer.pal(length(p_levels), "Spectral")
  }else{
    col_levels<- brewer.pal(length(p_levels), "Set1")
  }
  if(length(p_levels) < length(col_levels)){
    col_levels = col_levels[1:length(p_levels)]
  }
  col<-rep("",length(p))
  for(j in 1:length(p_levels)){
    col[which(p==p_levels[j])]<-col_levels[j]
  }
  pdf(paste(outDir,"/",type, ".pdf", sep=""))
  plot(0, 0, type = "n", xlim = c(min(x), max(x)), ylim = c(min(y), max(y)),xlab = "x", ylab = "y")
  for(j in 1:length(p_levels)){
    points(x[p==p_levels[j]], y[p==p_levels[j]], col = col[p==p_levels[j]], pch=20)
  }
  legend("bottomright", legend = p_levels, pch = 20, col = col_levels)
  dev.off()
}
