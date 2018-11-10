/*
Author: Zain Siddiqui
Predicting House Prices using Machine Learning (One-Shot Learning) on historical data.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

double ** inverseMatrix(double **m,int rows, int column);
double ** createMatrix(int rows, int columns);
double ** multiplyMatrix(double **m1, double **m2, int m1_rows, int m2_rows, int m1_cols, int m2_cols);
double ** transposeMatrix(double **m1, int m1_rows, int m1_cols);
void print(double ** m,int rows_m1, int cols_m2);



/* Multiply Matrix Method */
double ** multiplyMatrix(double **m1, double **m2, int m1_rows, int m2_rows, int m1_cols, int m2_cols){
  int ii, jj, kk;
 // int counter = 1;
  double tmp = 0;

  double **res = createMatrix(m1_rows, m2_cols);

for(ii = 0; ii < m1_rows; ii++) {
    for(jj = 0; jj < m2_cols; jj++) {
        for(kk = 0; kk < m1_cols; kk++) {
          tmp = tmp + m1[ii][kk] * m2[kk][jj];
        }
        res[ii][jj] = tmp;
        tmp = 0;
    }
}

    return res;
}

//Transpose matrix
double ** transposeMatrix(double **m1, int m1_rows, int m1_cols){
      double ** tM = createMatrix(m1_cols,m1_rows);


    int rows, columns;
      for(rows = 0; rows < m1_rows; rows++){
       for(columns = 0; columns < m1_cols; columns++){
           tM[columns][rows] = m1[rows][columns];
        }
  }

return tM;
}

//Traverses matrix
void print(double ** m,int rows_m1, int cols_m2){
  int i,j;
  for(i = 0; i < rows_m1; i++){
    for(j = 0; j < cols_m2; j++){
      printf ("%f ", m[i][j]);
    }
    printf("\n");
  }
}

//Dynamically allocates space for the matrix using mallloc
double ** createMatrix(int rows, int columns){
  double ** m = (double**) malloc(rows * sizeof(double*));
  for (int i = 0; i < rows; i++) {
		m[i] = (double*) malloc(columns * sizeof(double));
	}

  return m;
}

int main(int argc, char** argv){
//rows x cols
int numAttr; //cols (not including prices col)
int numEx;  //rows
FILE * fp;
  fp = fopen(argv[1], "r");
 
  fscanf(fp,"%d", &numAttr);
  fscanf(fp,"%d", &numEx);
 
 int cols = numAttr + 1;
 int rows = numEx ;

 double ** totalMatrix = createMatrix(rows, cols);

char *token;
char  help[100000];
int j, i;
i = 0;
while (!feof(fp)){
  if (i == numEx){
    break;
  }
  fscanf(fp, "%s", help);
  token = strtok(help,",");
  j = 0;
 
  while(token != NULL){
    totalMatrix[i][j] = atof(token);
   
    token = strtok(NULL, ",");
    j++;
  }
  i++;
}


//Prices of house as Nx1 matrix 
double ** Y;
 Y = createMatrix(rows,1);
 for(int i =0; i < rows; i++){
   Y[i][0] = totalMatrix[i][cols-1];
 }



//Attributes as Nx(K+1) matrix +1 because we need 1 constant in each row
double ** X;
X = createMatrix(numEx, numAttr+1);
for (int r = 0; r < numEx; r++){
    X[r][0] = 1.0;
  for (int c = 1; c < numAttr+1; c++){
      X[r][c]= totalMatrix[r][c-1];
  }
}

double ** a = transposeMatrix(X,numEx,numAttr+1);

double ** b = multiplyMatrix(a,X,numAttr+1,numEx,numEx,numAttr+1);




//c = (X^T * X)^-1
double ** c = inverseMatrix(b, numAttr+1, numAttr+1);


//c2 = retrieving matrix without augmented matrix
double ** c2 = createMatrix(numAttr+1,numAttr+1);
for(int i = 0; i < numAttr+1; i++){
  int t = numAttr+1;
    for(int j = 0; j < numAttr+1; j++){
        c2[i][j] = c[i][t];
        t++;
    }
  }

//d = (X^T * X)^-1 * (X^T)
double ** d = multiplyMatrix(c2,a,numAttr+1,numAttr+1,numAttr+1,numEx);


//e = ((X^T * X)^-1 * X^T) * Y
//size = numAttr+1 x 1
double ** e = multiplyMatrix(d,Y,numAttr+1,rows,numEx,1);

//w is the weightd matrix
//size = numAttr+1 x 1
double ** w = e;
//W = (X^T * X)^-1 * X^T * Y

 //numAttr 
int numEx2;  //rows
FILE * fp2;
//CHANGE TO 2
  fp2 = fopen(argv[2], "r");
 
  fscanf(fp2,"%d", &numEx2);
 int cols2 = numAttr;
 int rows2 = numEx2 ;

 double ** testMatrix = createMatrix(rows2, cols2);

char *token2;
char  help2[100000];
int j2, i2;
i2 = 0;
while (!feof(fp2)){

  if (i2 == numEx2){
    break;
  }
  fscanf(fp2, "%s", help2);
  token2 = strtok(help2,",");
  j2 = 0;
 
  while(token2 != NULL){
    testMatrix[i2][j2] = atof(token2);
    token2 = strtok(NULL, ",");
    j2++;
  }
  i2++;
}



/* Computing prices of given test data */

//number of test data
int numTest = 0;
while ( numTest < numEx2){

  double price = 0;
  price = w[0][0];
  int b =1;
  for (int i = 0; i < numAttr; i++){
      
      price = price + w[b][0] * testMatrix[numTest][i];
      
      b++;
      
  }
  
 printf("%0.0lf\n",price);

  numTest++;
}

  return 0;
}

/* Compute inverse using Gauss-Jordan Elimination */
double ** inverseMatrix(double **m, int rows, int column){
 // int augRows = rows * 2;
  int augCols = column * 2;

  double ** augM = createMatrix(rows, augCols);

//Initializing augmented matrix
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < column; j++){
      augM[i][j] = m[i][j];
    }
  }


     for (int j = 0; j < rows; j++){
        for (int i = column; i < augCols; i++ ){
            if (i == (j + (augCols/2))){
            augM[j][i] = 1;
          }else{
            augM[j][i] = 0;
          }

        }
     }

int t = 0;
    while (t < column){
       int j = 0;
        double n = augM[t][t];
        int c = 0;

        while (c < augCols){
          if (augM[t][c] != 0){
            
          augM[t][c] = augM[t][c] / n;
          }
          c++;
        }

      j=0;
        while (j < rows){
          if (j != t){
            double n2 = augM[j][t];
            //printf("n2: %f\n", n2);
            //new code
            int z = 0;
            while (z < augCols){
                
               augM[j][z] = augM[j][z] -  (n2 * augM[t][z]);
              
                z++;
          } 
    
          }

          j++;
        }
        
     
t++;
    }




return augM;
}