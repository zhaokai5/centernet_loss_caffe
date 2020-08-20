#include <vector>
#include <algorithm>
#include <cfloat>

#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/util/math_functions.cpp"

namespace caffe {
template <typename Dtype>
__global__ void CenterLossForwardGPU(const int nthreads,
        const Dtype* pred_data,const Dtype* label_data,
        Dtype* diff_data,Dtype* loss,float alpha_,float belta_,float N){
      CUDA_KERNEL_LOOP(index, nthreads) {
        Dtype pred=pred_data[index];
        Dtype label=label_data[index];
        if(abs(label-1)<0.00001){
          loss[index]=-powf(1-pred,alpha_)*log(max(pred,Dtype(FLT_MIN)));
          diff_data[index]=(alpha_*powf(1-pred,alpha_-1)*log(max(pred,Dtype(FLT_MIN)))-powf(1-pred,alpha_)/max(pred,Dtype(FLT_MIN)))/N;
        }
        else{
          loss[index]=-powf(1-label,belta_)*powf(pred,alpha_)*log(max(Dtype(FLT_MIN),1-pred));
          diff_data[index]=powf(1-label,belta_)*(powf(pred,alpha_)/max(Dtype(FLT_MIN),1-pred)-\
                            alpha_*powf(pred,alpha_-1)*log(max(Dtype(FLT_MIN),1-pred)))/N;
        }
      }
}
template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

      
  vector<int> shape =  bottom[0]->shape();
  float N=shape[0];
  const int nthreads=bottom[0]->count();
  const Dtype* pred_data=bottom[0]->gpu_data();
  const Dtype* label_data=bottom[1]->gpu_data();
  Dtype* loss_data= bottom[0]->mutable_gpu_diff();
  
  //bottom[0]->mutable_gpu_diff()
  Dtype* diff_data= diff_.mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(),static_cast<Dtype>(0),loss_data);
  caffe_gpu_set(bottom[0]->count(),static_cast<Dtype>(0),diff_data);
  //return;
  //Dtype loss=0;
  CenterLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads,pred_data,label_data,diff_data,loss_data,alpha_,belta_,N);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss/shape[0];
}




template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype* bottom_diff=bottom[0]->mutable_gpu_diff();

    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
      caffe_gpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom_diff);  // b
    //}
    //cudaMemcpy(diff_.mutable_cpu_data(), diff_.gpu_diff(),bottom[0]->count()*sizeof(float), cudaMemcpyDeviceToHost);
    //LOG(INFO) << "Sharing layer " <<diff_.cpu_data()[28*56+28];
    //caffe::caffe_copy(bottom[0]->count(),diff_.gpu_diff(),bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);
}  // namespace caffe
