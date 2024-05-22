#include "kernel.cuh"

__device__ void bodyInteraction(float *pos, float *vel, int bodies, float step,int gindex){
	float k = 0;
  // getting this thread's position
  float x = pos[gindex*3];
  float y = pos[gindex*3+1];
  float z = pos[gindex*3+2];

  // getting this thread's velocity
  float vx = vel[gindex*3];
  float vy = vel[gindex*3+1];
  float vz = vel[gindex*3+2];
  while (k < 100 ) {

    // getting new acceleration
    float acc[3] = {0};
    float dx, dy, dz, dist, cubedDist, vec;
    for (int i = 0; i < bodies*3; i+=3) {
      dx = pos[i] - x;
      dy = pos[i+1] - y;
      dz = pos[i+2] - z;

      dist = sqrt(dx*dx + dy*dy + dz*dz);
      if (dist != 0) {
        cubedDist = dist*dist*dist;
        vec = 1/cubedDist;

        acc[0] += vec*dx;
        acc[1] += vec*dy;
        acc[2] += vec*dz;
      }
    }

    vx += acc[0]*step;
    vy += acc[1]*step;
    vz += acc[2]*step;

    x += vx*step;
    y += vy*step;
    z += vz*step;

    __syncthreads();

    pos[gindex*3] = x;
    pos[gindex*3+1] = y;
    pos[gindex*3+2] = z;

    vel[gindex*3] = vx;
    vel[gindex*3+1] = vy;
    vel[gindex*3+2] = vz;

    __syncthreads();
    k += step;
  }
}

__device__ void bodyInteractionLocal(float *pos, float *vel, int bodies, float step,int gindex){
	float k = 0;
  extern __shared__ float sharedmem[];
  int numItems = blockDim.x;
  int tnum = blockIdx.x;
  unsigned int loop = bodies/numItems;
  // getting this thread's position
  float x = pos[gindex*3];
  float y = pos[gindex*3+1];
  float z = pos[gindex*3+2];

  // getting this thread's velocity
  float vx = vel[gindex*3];
  float vy = vel[gindex*3+1];
  float vz = vel[gindex*3+2];
  while (k < 100 ) {
    float acc[3] = {0};
    float dx, dy, dz, dist, cubedDist, vec;
    for(int l=0;l<loop;l++){
      int b_id=numItems*l+tnum;
      if(b_id<bodies){
        sharedmem[tnum*3] = pos[tnum*3 + numItems*l];
        sharedmem[tnum*3+1] = pos[tnum*3 + numItems*l+1];
        sharedmem[tnum*3+2] = pos[tnum*3 + numItems*l+2];
      }
      __syncthreads();
      int range=min(numItems,bodies-numItems*l)*3;
      for(int i=0;i<range;i+=3){
        dx = sharedmem[i] - x;
        dy = sharedmem[i+1] - y;
        dz = sharedmem[i+2] - z;

        dist = sqrt(dx*dx + dy*dy + dz*dz);
        if (dist != 0) {
          cubedDist = dist*dist*dist;
          vec = 1/cubedDist;

          acc[0] += vec*dx;
          acc[1] += vec*dy;
          acc[2] += vec*dz;
        }
      }
    }
    vx += acc[0]*step;
    vy += acc[1]*step;
    vz += acc[2]*step;

    x += vx*step;
    y += vy*step;
    z += vz*step;

    __syncthreads();

    pos[gindex*3] = x;
    pos[gindex*3+1] = y;
    pos[gindex*3+2] = z;

    vel[gindex*3] = vx;
    vel[gindex*3+1] = vy;
    vel[gindex*3+2] = vz;

    __syncthreads();
    k += step;
  }
}

__global__ void bodyInteraction1D(float *pos,float *vel, int bodies,float step){
  int gindex = blockDim.x * blockIdx.x + threadIdx.x;
  if (gindex < bodies) {
    bodyInteraction(pos,vel,bodies,step,gindex);
  }
}
__global__ void bodyInteraction2D(float *pos,float *vel, int bodies,float step){
  int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int globalIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
  int gindex=globalIdx_y*blockDim.x*blockIdx.x+globalIdx_x;
  if (gindex<bodies){
    bodyInteraction(pos,vel,bodies,step,gindex);
  }
}

__global__ void bodyInteractionLocal1D(float *pos,float *vel, int bodies,float step){
  int gindex = blockDim.x * blockIdx.x + threadIdx.x;
  if (gindex < bodies) {
    bodyInteractionLocal(pos,vel,bodies,step,gindex);
  }
}