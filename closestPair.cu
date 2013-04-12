#include <stdio.h>
#include <stdlib.h>
#include <values.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

int cudaAvailable(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}
 
typedef struct { double x, y; } point_t, *point;
 
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
__global__ void cuda_brute_force(point* pts, int max_n,point_t *a,point_t *b){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid==0){
        a->x=1.0;
        a->y=2.0;
        b->x=3.0;
        b->y=4.0;
    }

}
double closest(point* sx, int nx, point* sy, int ny, point *a, point *b)
{
        int left, right, i;
        double d, min_d, x0, x1, mid, x;
        point a1, b1;
        point *s_yy;
 
        if (nx <= 8) return brute_force(sx, nx, a, b);
 
        s_yy  = (point*)malloc(sizeof(point) * ny);
        mid = sx[nx/2]->x;
 
        /* adding points to the y-sorted list; if a point's x is less than mid,
           add to the begining; if more, add to the end backwards, hence the
           need to reverse it */
        left = -1; right = ny;
        for (i = 0; i < ny; i++)
                if (sy[i]->x < mid) s_yy[++left] = sy[i];
                else                s_yy[--right]= sy[i];
 
        /* reverse the higher part of the list */
        for (i = ny - 1; right < i; right ++, i--) {
                a1 = s_yy[right]; s_yy[right] = s_yy[i]; s_yy[i] = a1;
        }
 
        min_d = closest(sx, nx/2, s_yy, left + 1, a, b);
        d = closest(sx + nx/2, nx - nx/2, s_yy + left + 1, ny - left - 1, &a1, &b1);
 
        if (d < min_d) { min_d = d; *a = a1; *b = b1; }
        d = sqrt(min_d);
 
        /* get all the points within distance d of the center line */
        left = -1; right = ny;
        for (i = 0; i < ny; i++) {
                x = sy[i]->x - mid;
                if (x <= -d || x >= d) continue;
 
                if (x < 0) s_yy[++left]  = sy[i];
                else       s_yy[--right] = sy[i];
        }
 
        /* compare each left point to right point */
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
 
#define NP 100//1000000
int main()
{
        int i, cudaAble;
        point a;
        point b;
        

        point pts  = (point) malloc(sizeof(point_t) * NP);
        point* s_x = (point*)malloc(sizeof(point) * NP);
        point* s_y = (point*)malloc(sizeof(point) * NP);
        
        //device memory allocation
        point_t* A;
        point_t* B;
        point* S_X;
        dim3 dGrid((NP/512)+1);
        dim3 dBlock(512);

        point_t a1;
        point_t b1;

        cudaAble=cudaAvailable();
        //silence errors

        for(i = 0; i < NP; i++) {             
                pts[i].x = 100 * (double) rand()/RAND_MAX;
                pts[i].y = 100 * (double) rand()/RAND_MAX;
                s_x[i] = pts + i;
        }
        
        if(cudaAble){
            cudaMalloc((void**)&A,1*sizeof(point_t));
            cudaMalloc((void**)&B,1*sizeof(point_t));
            cudaMalloc((void**)&S_X,NP*sizeof(point));
            cudaMemcpy(S_X,s_x,NP*sizeof(point),cudaMemcpyHostToDevice);

            printf("Using CUDA\n");

            //test kernel call
            cuda_brute_force<<<dGrid,dBlock>>>(S_X,NP,A,B);

            cudaMemcpy(&a1,A,1*sizeof(point_t),cudaMemcpyDeviceToHost);
            cudaMemcpy(&b1,B,1*sizeof(point_t),cudaMemcpyDeviceToHost);
            printf("a=%f,%f and b=%f,%f \n",a1.x,a1.y,b1.x,b1.y);
        }

        printf("brute force: %g, ", sqrt(brute_force(s_x, NP, &a, &b)));
        printf("between (%f,%f) and (%f,%f)\n", a->x, a->y, b->x, b->y);        
 
        memcpy(s_y, s_x, sizeof(point) * NP);
        qsort(s_x, NP, sizeof(point), cmp_x);
        qsort(s_y, NP, sizeof(point), cmp_y);
 
        /*printf("min: %g; ", sqrt(closest(s_x, NP, s_y, NP, &a, &b)));
        printf("point (%f,%f) and (%f,%f)\n", a->x, a->y, b->x, b->y);*/
 
        //free device memory
        if(cudaAble){
            cudaFree(A);
            cudaFree(B);
            cudaFree(S_X);
        }

        /* not freeing the memory, let OS deal with it.  Habit. */ 
        free(pts);
        free(s_x);
        free(s_y);
        return 0;
}