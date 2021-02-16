//command for compiling
//gcc -cpp -fPIC -shared lib_cvtIntegralImage45.c -lm -o lib_cvtIntegralImage45.so -O3

#include <stdint.h>
void cvtIntegralImage45(double *mat_Z, int height_z, int width_z, double *mat_tmpX){
	#define Z(h,w)    (mat_Z   [(h)*width_z+(w)])
	#define tmpX(h,w) (mat_tmpX[(h)*width_z+(w)])

	for (int J=2; J<width_z; J++){//J in range(2, Z.shape(1)){
	  Z(0,J) = Z(0,J) + Z(1,J-1) + tmpX(0, J-1);
	  for (int I=1; I<height_z-1; I++){//I in range(1, Z.shape(0)-1){
	    Z(I,J) = Z(I,J) + Z(I-1,J-1) + Z(I+1,J-1) - Z(I,J-2) + tmpX(I,J-1);
	  }
	  Z(height_z-1,J) = Z(height_z-1,J) + Z(height_z-2,J-1)  + tmpX(height_z-1,J-1);
	}
  #undef Z
	#undef tmpX

}
