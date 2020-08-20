#include <vector>
#include <cfloat>
#include <algorithm>

#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/util/math_functions.cpp"

namespace caffe {
template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    alpha_ = this->layer_param_.center_loss_param().alpha();
    belta_ = this->layer_param_.center_loss_param().belta();

}
template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  

  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> shape =  bottom[0]->shape();

  int N=shape[0]; int C=shape[1]; 
  int H=shape[2]; int W=shape[3];
  Dtype* diff_data = diff_.mutable_cpu_data();
  caffe_set(bottom[0]->count(),static_cast<Dtype>(0),diff_data);
  Dtype loss=0;
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();

  for(int id_n=0;id_n<N;id_n++){
    const Dtype* pred_data_channel=pred_data+id_n*C*H*W;
    const Dtype* label_data_channel=label_data+id_n*C*H*W;
    Dtype* diff_data_channel=diff_data+id_n*C*H*W;
    for(int id_c=0;id_c<C;id_c++){
      for(int id_h=0;id_h<H;id_h++){
        for(int id_w=0;id_w<W;id_w++){
           Dtype labeldata=label_data_channel[id_c*H*W+id_h*W+id_w];
           Dtype preddata=pred_data_channel[id_c*H*W+id_h*W+id_w];
           if(abs(labeldata-1)<0.00001){
             loss-=log(std::max(preddata,Dtype(FLT_MIN)))*pow((1-preddata),alpha_);
             diff_data_channel[id_c*H*W+id_h*W+id_w]=(log(std::max(preddata,Dtype(FLT_MIN)))*alpha_*pow((1-preddata),alpha_-1)-pow((1-preddata),alpha_)/preddata)/N;
           }
           else{
             loss-=pow((1-labeldata),belta_)*pow(preddata,alpha_)*log(1-preddata);
             diff_data_channel[id_c*H*W+id_h*W+id_w]=pow((1-labeldata),belta_)*(pow(preddata,alpha_)/(1-preddata)-log(1-preddata)*pow(preddata,alpha_-1)*alpha_)/N;
           }
        }
      }
    }
  }

  top[0]->mutable_cpu_data()[0] = loss/N;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    
    Dtype* bottom_diff=bottom[0]->mutable_cpu_diff();
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_cpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom_diff);  // b
    
    // caffe::caffe_copy(bottom[0]->count(),diff_.cpu_data(),bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe
