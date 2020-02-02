#frame_files <- lapply(sys.frames(), function(x) x$ofile)
#frame_files <- Filter(Negate(is.null), frame_files)
#setwd(dirname(frame_files[[length(frame_files)]]))
#source("myFunction.R")

scriptDir<-dirname(sys.frame(1)$ofile)
source(paste(scriptDir, "/myFunction.R", sep=""))


#args <- commandArgs(trailingOnly=TRUE)
#tumoppArgs <- args[1] 
#outprefix <- args[2]
#itr <- args[3]

#if(!any(objects()=="tumoppArgs") | is.na(tumoppArgs)){
#  tumoppArgs<-"-N 40000 -D 2 -C hex -k 100 -L const"
#}
#if(!any(objects()=="outprefix" | is.na(outprefix))){
# outprefix<-"out"
#}
#if(!any(objects()=="itr") | is.na(itr)){
# itr<-1
#}
#if(!any(objects()=="rand") | is.na(rand)){
#  rand<-TRUE
#}

numberOfRegion<-8

parseTumoppArgs<-function(tumoppArgs){
  tmp<-sub("-", "", unlist(strsplit(tumoppArgs," ")))
  prm<-tmp[(1:length(tmp)) %%2 ==0]
  names(prm)<-tmp[(1:length(tmp)) %%2 ==1]
  return(prm)
}


prm = list(
  D = 3,
  C = "hex",
  N = 10000,
  k = 10**seq(1,6,1),
  L = c("const", "linear", "step"),
  P = c("random", "mindrag")
)

generateRandomTumoppArgs<-function(){
  args<-NULL
  for( i in 1:length(prm)){
    if(length(prm[[i]])==1){
      args <- c(args, paste("-", names(prm)[i], " ", prm[[i]], sep=""))
    }else{
      args <- c(args, paste("-", names(prm)[i], " ", sample(prm[[i]],1), sep=""))
    }
  }
  return(paste(args,collapse = " "))
}


simulate<-function(outprefix){
  if(rand){
    tumoppArgs<-generateRandomTumoppArgs()
  }
  result <- tumopp(tumoppArgs,numberOfRegion)
  vaf_mat<-getVafMat(result)
  png(paste(outprefix,"png",sep="."))
  heatmap(vaf_mat[nrow(vaf_mat):1,], scale="none", col=c("gray95", my.col(100)), Rowv=NA,labRow=NA,distfun=my.dist, hclustfun = my.hclust)
  dev.off()
  vaf_mat<-resizeImageMatrix(vaf_mat, numberOfRegion)
  write.table(vaf_mat, file=paste(outprefix, "tab", sep="."), sep="\t",col.names=FALSE, row.names=FALSE, quote=FALSE)
  prmset<-parseTumoppArgs(tumoppArgs)
  write(rbind(names(prmset),prmset),file=paste(outprefix,"prm", sep="."),ncolumns=2,sep="\t")
}


if(itr == 1){
  simulate(outprefix)
  }else{
 for(i in 1:itr){
   simulate(paste(outprefix,i,sep="."))
  } 
}

