#include <stdio.h>
#include <stdlib.h>
//#include <values.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAXDOUBLE 99.9999999999999999999

int cudaAvailable(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}
 
typedef struct { float x, y; } point_t, *point;
 
/*
inline double dist(point a, point b)
{
        double dx = a->x - b->x, dy = a->y - b->y;
        return dx * dx + dy * dy;
}
 
inline int cmp_dbl(double a, double b)
{
        return a < b ? -1 : a > b ? 1 : 0;
}

int cmp_x(const void *a, const void *b) {
        return cmp_dbl( (*((point*)a))->x, (*((point*)b))->x );
}
 
int cmp_y(const void *a, const void *b) {
        return cmp_dbl( (*((point*)a))->y, (*((point*)b))->y );
}
 


 double brute_force(point* pts, int max_n, point *a, point *b)
{
        int i, j;
        double d, min_d = MAXDOUBLE;
 
        for (i = 0; i < max_n; i++) {
                for (j = i + 1; j < max_n; j++) {
                        d = dist(pts[i], pts[j]);
                        if (d >= min_d ) continue;
                        *a = pts[i];
                        *b = pts[j];
                        min_d = d;
                }
        }
        return min_d;
}
*/

float brute_force(point* pts, int max_n, point *a, point *b){
        int i, j;
        float dx, dy, d, min_d = MAXDOUBLE;
 
        for (i = 0; i < max_n; i++) {
                for (j = i + 1; j < max_n; j++) {
                        //d = dist(pts[i], pts[j]);

                        dx = pts[i]->x - pts[j]->x;
                        dy = pts[i]->y - pts[j]->y;
                        d = dx*dx+dy*dy;
                        
                        if ( !(d >= min_d ) )
                        {
                            *a = pts[i];
                            *b = pts[j];
                            min_d = d;
                        }
                }
        }
        return min_d;
}

__global__ void cuda_brute_force(point_t* pts, int  max_n, point_t *a, point_t *b, float *min_d){

    int j;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        
    float dx, dy, d; 

    while (i < max_n) {
        for (j = i + 1; j < max_n; j++) {
                            
            dx = pts[i].x - pts[j].x;
            dy = pts[i].y - pts[j].y;

            d = dx*dx+dy*dy;
                            
            if ( !(d >= *min_d ) ) {
                *a = pts[i];
                *b = pts[j];
                *min_d = d;
            }
            __syncthreads();
        }
        i += blockDim.x * gridDim.x;
    }
}



/*
double closest(point* sx, int nx, point* sy, int ny, point *a, point *b)
{
        int left, right, i;
        double d, min_d, x0, x1, mid, x;
        point a1, b1;
        point *s_yy;
 
        if (nx <= 8) return brute_force(sx, nx, a, b);
 
        s_yy  = (point*)malloc(sizeof(point) * ny);
        mid = sx[nx/2]->x;

        left = -1; right = ny;
        for (i = 0; i < ny; i++)
                if (sy[i]->x < mid) s_yy[++left] = sy[i];
                else                s_yy[--right]= sy[i];
 

        for (i = ny - 1; right < i; right ++, i--) {
                a1 = s_yy[right]; s_yy[right] = s_yy[i]; s_yy[i] = a1;
        }
 
        min_d = closest(sx, nx/2, s_yy, left + 1, a, b);
        d = closest(sx + nx/2, nx - nx/2, s_yy + left + 1, ny - left - 1, &a1, &b1);
 
        if (d < min_d) { min_d = d; *a = a1; *b = b1; }
        d = sqrt(min_d);
 
        left = -1; right = ny;
        for (i = 0; i < ny; i++) {
                x = sy[i]->x - mid;
                if (x <= -d || x >= d) continue;
 
                if (x < 0) s_yy[++left]  = sy[i];
                else       s_yy[--right] = sy[i];
        }
 
        while (left >= 0) {
                x0 = s_yy[left]->y + d;
 
                while (right < ny && s_yy[right]->y > x0) right ++;
                if (right >= ny) break;
 
                x1 = s_yy[left]->y - d;
                for (i = right; i < ny && s_yy[i]->y > x1; i++)
                        if ((x = dist(s_yy[left], s_yy[i])) < min_d) {
                                min_d = x;
                                d = sqrt(min_d);
                                *a = s_yy[left];
                                *b = s_yy[i];
                        }
 
                left --;
        }
 
        free(s_yy);
        return min_d;
}
*/ 


//#define NP 16
//1000000

int main(int argc, char* argv[])
{
        if (argc != 2) {
            printf("Incorrect no of arguments\n");
            return 1;
        }

        int NP = atoi(argv[1]);

        int i, cudaAble;
        point a;
        point b;
        
        point pts  = (point) malloc(sizeof(point_t) * NP);
        point* s_x = (point*)malloc(sizeof(point) * NP);
        point* s_y = (point*)malloc(sizeof(point) * NP);
        
        //device memory allocation
        point_t* A;
        point_t* B;
        point_t* S_X;
        float* MID_D;

        point_t a1;
        point_t b1;
        float min_d = MAXDOUBLE;
        point_t* s_x1=(point_t*)malloc(sizeof(point_t) * NP);

        cudaAble=cudaAvailable();

        for(i = 0; i < NP; i++) {             
                pts[i].x = 100 * (float) rand()/RAND_MAX;
                pts[i].y = 100 * (float) rand()/RAND_MAX;
                s_x[i] = pts + i;
                s_x1[i]= pts[i];

        }
        printf("\n");

        if(cudaAble){

            int   d;
            cudaDeviceProp prop;
            cudaGetDevice(&d);
            cudaGetDeviceProperties(&prop, d);

            int dBlock = prop.maxThreadsDim[0];
            int mElemnts = prop.totalGlobalMem / (2 * sizeof(float));
            if (NP > mElemnts) {
                NP = mElemnts;
            }

            int dGrid = (NP+dBlock-1)/dBlock;

            cudaMalloc((void**)&A,1*sizeof(point_t));
            cudaMalloc((void**)&B,1*sizeof(point_t));
            cudaMalloc((void**)&S_X,NP*sizeof(point_t));
            cudaMalloc((void**)&MID_D,1*sizeof(float));
            cudaMemcpy(S_X,s_x1,NP*sizeof(point_t),cudaMemcpyHostToDevice);
            cudaMemcpy(MID_D,&min_d,1*sizeof(float),cudaMemcpyHostToDevice);

            //test kernel call
            cuda_brute_force<<<dGrid,dBlock>>>(S_X, NP, A, B, MID_D);

        }

        printf("brute force : %f, ", sqrt(brute_force(s_x, NP, &a, &b)));
        printf("between (%f,%f) and (%f,%f)\n", a->x, a->y, b->x, b->y);  

        if(cudaAble){
            cudaDeviceSynchronize();

            cudaError_t error= cudaGetLastError();

            if(error!= cudaSuccess)
                printf("%s\n",cudaGetErrorString(error));

            cudaMemcpy(&a1,A,1*sizeof(point_t),cudaMemcpyDeviceToHost);
            cudaMemcpy(&b1,B,1*sizeof(point_t),cudaMemcpyDeviceToHost);
            cudaMemcpy(&min_d,MID_D,1*sizeof(float),cudaMemcpyDeviceToHost);

            printf("Using CUDA  : %f, ", sqrt(min_d) );           
            printf("between (%f,%f) and (%f,%f)\n",a1.x,a1.y,b1.x,b1.y);
        }    

        /*memcpy(s_y, s_x, sizeof(point) * NP);
        qsort(s_x, NP, sizeof(point), cmp_x);
        qsort(s_y, NP, sizeof(point), cmp_y);
 
        printf("min: %g; ", sqrt(closest(s_x, NP, s_y, NP, &a, &b)));
        printf("point (%f,%f) and (%f,%f)\n", a->x, a->y, b->x, b->y);*/
 
        //free device memory
        if(cudaAble){
            cudaFree(A);
            cudaFree(B);
            cudaFree(S_X);
            cudaFree(MID_D);
            cudaDeviceReset();
        }

        /* not freeing the memory, let OS deal with it.  Habit. */ 
        free(pts);
        free(s_x);
        free(s_y);
        return 0;
}
