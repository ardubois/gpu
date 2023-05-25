defmodule MyKernel do
  import GPU
kernel julia(ptr) do
  var dim int = 1000
  var x int = blockIdx.x;
  var y int = blockIdx.y;
  var offset int = x + y * gridDim.x;
#####
  var juliaValue int = 2
  var scale float = 1.5;
  var jx float = scale * (dim/2 - x)/(dim/2);
  var jy float = scale * (dim/2 - y)/(dim/2);

  var cr float = -0.8
  var ci float = 0.156
  var ar float = jx
  var ai float = jy
  for i in range(0,200) do
      ar = (ar*ar - ai*ai) + cr
      ai = (ai*ar + ar*ai) + ci
      if ((ar * ar)+(ai * ai ) > 1000) do
        juliaValue = 0
      end
      if (juliaValue != 0) do
        juliaValue = 1
      end

    end
#####
  ptr[offset*4 + 0] = 255 * juliaValue;
  ptr[offset*4 + 1] = 0;
  ptr[offset*4 + 2] = 0;
  ptr[offset*4 + 3] = 255;

end
end





kernel=GPU.build('julia')
