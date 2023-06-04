defmodule NBodies do
  import GPU
  kernel gpu_nBodies(p,dt,n,softening,[:matrex,:float,:int,:float]) do
    var i int = blockDim.x * blockIdx.x + threadIdx.x
    if (i < n) do
      var fx float= 0.0
      var fy float= 0.0
      var fz float = 0.0
      for j in range(0,n) do
        var dx float= p[6*j] - p[6*i];
        var dy float= p[6*j+1] - p[6*i+1];
        var dz float= p[6*j+2] - p[6*i+2];
        var distSqr  float = dx*dx + dy*dy + dz*dz + softening;
        var invDist float = rsqrtf(distSqr);
        var invDist3  float = invDist * invDist * invDist;

        fx = fx + dx * invDist3;
        fy = fy + dy * invDist3;
        fz = fz + dz * invDist3;
      end
      p[6*i+3] = p[6*i+3]+ dt*fx;
      p[6*i+4] = p[6*i+4]+ dt*fy;
      p[6*i+5] = p[6*i+5]+ dt*fz;
    end

  end

  def nbodies(0,p,_dt,_softening,_n) do
    p
  end
  def nbodies(i,p,dt,softening,n) do
    {fx,fy,fz} = calc_nbodies(n,i,p,softening,0.0,0.0,0.0)
    Matrex.set(p,1,6*i+3,Matrex.at(p,1,6*i+3)+ dt*fx);
    Matrex.set(p,1,6*i+4,Matrex.at(p,1,6*i+4) + dt*fy);
    Matrex.set(p,1,6*i+5,Matrex.at(p,1,6*i+5) + dt*fz);
    nbodies(i-1,p,dt,softening,n)
  end

def calc_nbodies(0,_i,_p,_softening,fx,fy,fz) do
  {fx,fy,fz}
end
def calc_nbodies(j,i,p,softening,fx,fy,fz) do
    dx = Matrex.at(p,1,6*j) - Matrex.at(p,1,6*i);
    dy = Matrex.at(p,1,6*j+1) - Matrex.at(p,1,6*i+1);
    dz = Matrex.at(p,1,6*j+2) - Matrex.at(p,1,6*i+2);
    distSqr = dx*dx + dy*dy + dz*dz + softening;
    invDist = 1/:math.sqrt(distSqr);
    invDist3 = invDist * invDist * invDist;

    fx = fx + dx * invDist3;
    fy = fy + dy * invDist3;
    fz = fz + dz * invDist3;
    calc_nbodies(j-1,i,p,softening,fx,fy,fz)
end
def equality(a, b) do
  if(abs(a-b) < 0.001) do
    true
  else
    false
  end
end

end

nBodies = 3000;
block_size =  128;
nBlocks = (nBodies + block_size - 1) / block_size;
softening = 0.000000001;
dt = 0.01; # time step
size_body = 6

size_matrex = size_body * nBodies


h_buf = Matrex.random(1,size_matrex)





ker=GPU.build('gpu_nBodies')

d_buf =GPU.create_ref(h_buf)

prev = System.monotonic_time()
GPU.spawn(ker,{nBlocks,block_size,1},{1,1,1},[d_buf,dt,nBodies,softening])
GPU.synchronize()
gpu_resp = GPU.get_matrex(d_buf)
next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

IO.puts gpu_resp

prev = System.monotonic_time()
cpu_resp = NBodies.nbodies(nBodies-1,h_buf,dt,softening,nBodies-1)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

IO.puts cpu_resp

IO.inspect cpu_resp
