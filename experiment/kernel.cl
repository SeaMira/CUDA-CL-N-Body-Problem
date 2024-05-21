kernel void bodyInteraction(__global float *pos, __global float *vel, const int bodies, const float step) {
  int gindex = get_global_id(0);
  float k = 0;
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
  }
}
