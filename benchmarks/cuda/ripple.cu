#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#define _bitsperpixel 32
#define _planes 1
#define _compression 0

#define _xpixelpermeter 0x13B //0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x13B//0x130B //2835 , 72 DPI
#define pixel 0xFF
#pragma pack(push,1)
typedef struct{
    uint8_t signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct{
    uint32_t dibheadersize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t ypixelpermeter;
    uint32_t xpixelpermeter;
    uint32_t numcolorspallette;
    uint32_t mostimpcolor;
} bitmapinfoheader;
typedef struct {
    fileheader fileheader;
    bitmapinfoheader bitmapinfoheader;
} bitmap;
#pragma pack(pop)
__global__
void kernel(uint8_t *ptr, uint32_t DIM, int ticks ){
   int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y  = threadIdx.y + blockIdx.y * blockDim.y;
    int  offset  = x + y * blockDim.x * gridDim.x;
            
            float fx = 0.5 *  x - DIM/15;
	    float fy = 0.5* y - DIM/15;
	    float d = sqrtf( fx * fx + fy * fy );
	    unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));
		ptr[offset*4 + 0] = grey;
		ptr[offset*4 + 1] = grey;
		ptr[offset*4 + 2] = grey;
		ptr[offset*4 + 3] = 255;
        
 }
 
 void genBpm (uint32_t height, uint32_t width, uint8_t *pixelbuffer) {
    uint32_t pixelbytesize = height*width*_bitsperpixel/8;
    uint32_t  _filesize =pixelbytesize+sizeof(bitmap);
    FILE *fp = fopen("test.bmp","wb");
    bitmap *pbitmap  = (bitmap*)calloc(1,sizeof(bitmap));


    //strcpy(pbitmap->fileheader.signature,"BM");
    pbitmap->fileheader.signature[0] = 'B';
    pbitmap->fileheader.signature[1] = 'M';
    pbitmap->fileheader.filesize = _filesize;
    pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
    pbitmap->bitmapinfoheader.dibheadersize =sizeof(bitmapinfoheader);
    pbitmap->bitmapinfoheader.width = width;
    pbitmap->bitmapinfoheader.height = height;
    pbitmap->bitmapinfoheader.planes = _planes;
    pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
    pbitmap->bitmapinfoheader.compression = _compression;
    pbitmap->bitmapinfoheader.imagesize = pixelbytesize;
    pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter ;
    pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter ;
    pbitmap->bitmapinfoheader.numcolorspallette = 0;
    fwrite (pbitmap, 1, sizeof(bitmap),fp);
    //memset(pixelbuffer,pixel,pixelbytesize);
    fwrite(pixelbuffer,1,pixelbytesize,fp);
    fclose(fp);
    free(pbitmap);
    free(pixelbuffer);
}


int main( void ) {
   
   uint32_t height = 1000;
    uint32_t width  = 1000;
    uint32_t DIM = 1000;
    cudaError_t j_error;
    
    uint32_t pixelbytesize=  height*width*_bitsperpixel/8;
     
    //printf(" pixel byte size %lu\n",pixelbytesize);
   
     uint8_t *h_pixelbuffer = (uint8_t*)malloc(pixelbytesize);
     uint8_t *d_pixelbuffer;

     ////////
    cudaMalloc( (void**)&d_pixelbuffer, pixelbytesize);
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(j_error));
    ////////

    
    ////////////////////
    dim3 grid(DIM,DIM);

    kernel<<<grid, 1>>>(d_pixelbuffer,height,10); // compute interbody forces
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(j_error));
  ////////



    cudaMemcpy(h_pixelbuffer, d_pixelbuffer, pixelbytesize, cudaMemcpyDeviceToHost); // return results 
    j_error = cudaGetLastError();
    if(j_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(j_error));

    
    genBpm(height,width,h_pixelbuffer);
    
}
