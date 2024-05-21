kernel void bodyInteraction(__global float *pos, __global float *vel, const int bodies, const float step, local float *prods) {
  int gindex = get_global_id(0);
  float k = 0;
  int numItems = get_local_size( 0 ); // # work-items per work-group
  int tnum = get_local_id( 0 ); // thread (i.e., work-item) number in this work-group
  unsigned int loop = bodies/numItems;
  if (gindex < bodies) {
    float x, y, z, vx, vy, vz;
    while (k < 100 ) {

      // getting this thread's position
      x = pos[gindex*3];
      y = pos[gindex*3+1];
      z = pos[gindex*3+2];

      // getting this thread's velocity
      vx = vel[gindex*3];
      vy = vel[gindex*3+1];
      vz = vel[gindex*3+2];

      // getting new acceleration
      float acc[3] = {0};
      float x_com = 0, y_com = 0, z_com = 0, dx, dy, dz, dist, cubedDist, vec;


      for (int l = 0; l < loop; l++) {
        prods[tnum*3] = pos[tnum*3 + numItems*l];
        prods[tnum*3+1] = pos[tnum*3 + numItems*l+1];
        prods[tnum*3+2] = pos[tnum*3 + numItems*l+2];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < numItems; i+=3) {
            dx = prods[i] - x;
            dy = prods[i+1] - y;
            dz = prods[i+2] - z;

            dist = sqrt(dx*dx + dy*dy + dz*dz);
            
            cubedDist = dist*dist*dist + 0.1;
            vec = 1/cubedDist;

            acc[0] += vec*dx;
            acc[1] += vec*dy;
            acc[2] += vec*dz;
            
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      vx += acc[0]*step;
      vy += acc[1]*step;
      vz += acc[2]*step;

      x += vx*step;
      y += vy*step;
      z += vz*step;

      barrier(CLK_GLOBAL_MEM_FENCE);

      pos[gindex*3] = x;
      pos[gindex*3+1] = y;
      pos[gindex*3+2] = z;

      vel[gindex*3] = vx;
      vel[gindex*3+1] = vy;
      vel[gindex*3+2] = vz;

      barrier(CLK_GLOBAL_MEM_FENCE);
      k += step;
    }
    pos[0] = 120;
  }
}
