
#include <ctc.h>
#include <cstddef>
#include "detail/cpu_ctc.h"


#include<stdio.h>
#include<jni.h>
#include "WarpCTC.h"

extern "C" {
    JNIEXPORT jint JNICALL Java_com_intel_analytics_pipeline_acoustic_training_WarpCTC_computeCTCLoss
    (JNIEnv * env, jobject obj, jfloatArray activations, jfloatArray gradients, jintArray flat_labels, jintArray label_lengths,
    jintArray input_lengths, jint alphabet_size, jint minibatch, jfloatArray costs, jint num_threads) {

   float* _activations=env->GetFloatArrayElements(activations, 0);

    float* _gradients=env->GetFloatArrayElements(gradients, 0);

    float* _costs = env->GetFloatArrayElements(costs, 0);

    int* _flat_labels = env->GetIntArrayElements(flat_labels, 0);

    int* _label_lengths = env->GetIntArrayElements(label_lengths, 0);

    int* _input_lengths = env->GetIntArrayElements(input_lengths, 0);

       ctcOptions info;
       info.loc = CTC_CPU;
       info.num_threads = num_threads;
       info.blank_label = 0;

       size_t size_bytes;
      get_workspace_size(_label_lengths,
                       _input_lengths,
                       alphabet_size,
                       minibatch,
                       info,
                       &size_bytes);

        void* workspace = malloc(size_bytes);

        ctcStatus_t status = compute_ctc_loss(_activations,
                                          _gradients,
                                          _flat_labels,
                                          _label_lengths,
                                          _input_lengths,
                                          alphabet_size,
                                          minibatch,
                                          _costs,
                                          workspace,
                                          info);


          env->ReleaseFloatArrayElements(activations,_activations, 0);
          env->ReleaseFloatArrayElements(gradients,_gradients, 0);
           env->ReleaseFloatArrayElements(costs,_costs,0);
         env->ReleaseIntArrayElements(flat_labels, _flat_labels,0);
           env->ReleaseIntArrayElements(label_lengths, _label_lengths,0);
           env->ReleaseIntArrayElements(input_lengths, _input_lengths,0);


        free(workspace);
        return int(status);
       }


JNIEXPORT void JNICALL Java_com_intel_analytics_pipeline_acoustic_training_WarpCTC_printHello
  (JNIEnv * env, jobject obj) {
    printf("Hello World");
    return;
  }
 }