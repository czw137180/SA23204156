#include <Rcpp.h>
using namespace Rcpp;

//' @title Calculate the maximum module length of each column in the matrix
//' @description Calculate the maximum module length of each column in the matrix
//' @param x a matrix
//' @return a constant of maximum module length of each column in the matrix
//' @examples
//' \dontrun{
//' mu <- c(10,10)
//' s <- matrix(c(1,2,2,1,1,3,4,5,6,7,8,9), nrow = 2, ncol = 6)
//' n <- maxcol(s)
//' n
//' }
//' @export
// [[Rcpp::export]]
double maxcol(NumericMatrix x){
  int nr = x.nrow();
  int nc = x.ncol();
  double y, yl;
  y = 0.0;
  NumericVector ycol(nr);
  for(int j = 0; j < nc; j++){
    ycol = x.column(j);
    yl = sum(ycol * ycol);
    if(yl > y) y = yl;
  }
  y = sqrt(y);
  return y;
 }