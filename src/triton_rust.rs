use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference::{ServerLiveRequest, ServerReadyRequest, ModelReadyRequest};
use inference::{InferParameter, ModelInferRequest, ModelInferResponse};
use inference::{ModelMetadataRequest, ModelMetadataResponse};
use inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

use std;
use std::error::Error;
use std::vec::Vec;
use std::collections::HashMap;
use ndarray::{Array, Array3};

include!(concat!(env!("OUT_DIR"), "/shared_memory_binding.rs"));

pub mod inference {
    tonic::include_proto!("inference");
}

pub struct TritonInference {
    client: GrpcInferenceServiceClient<tonic::transport::Channel>
}

pub struct CudaSharedMemoryRegionHandle {
    handle: *mut i32
}

impl TritonInference {
    pub async fn connect(address: &'static str) -> Result<Self, Box<dyn Error>> {

        let mut client = GrpcInferenceServiceClient::connect(address).await?;

        Ok(TritonInference {
            client: client,
        })
    }

    pub async fn is_server_live(&mut self) -> Result<bool,  Box<dyn Error>> {
        let request = tonic::Request::new(ServerLiveRequest {});
        let response = self.client.server_live(request).await?;

        Ok(response.get_ref().live)
    }

    pub async fn is_server_ready(&mut self) -> Result<bool,  Box<dyn Error>> {
        let request = tonic::Request::new(ServerReadyRequest {});
        let response = self.client.server_ready(request).await?;

        Ok(response.get_ref().ready)
    }

    pub async fn is_model_ready(&mut self, model_name: &str, version_number: &str) -> Result<bool,  Box<dyn Error>> {
        let request = tonic::Request::new(ModelReadyRequest {name: model_name.to_string(), version: version_number.to_string()});
        let response = self.client.model_ready(request).await?;

        Ok(response.get_ref().ready)
    }

    pub fn get_infer_input(&mut self, input_name: &str, input_datatype: &str, tensor_shape: &[i64], parameters_map: HashMap<String, InferParameter>) -> InferInputTensor {

        InferInputTensor {
            name: input_name.to_string(),
            datatype: input_datatype.to_string(),
            shape: tensor_shape.to_vec(),
            parameters: parameters_map,
            contents: None
        }
    }

    pub async fn get_model_metadata(&mut self, model_name: &str, model_version: &str) -> Result<ModelMetadataResponse,  Box<dyn Error>> {
        let request = tonic::Request::new(ModelMetadataRequest {name: model_name.to_string(), version: model_version.to_string()});
        let response = self.client.model_metadata(request).await?;

        Ok(response.get_ref().clone())
    }

    pub async fn infer(&mut self, model_name: &str, model_version: &str, request_id: &str, inputs_vec: Vec<InferInputTensor>) -> Result<ModelInferResponse,  Box<dyn Error>> {

        let rand_image = Array3::<f64>::zeros((3, 512, 512));
        let mut tensor_bytes = Vec::<u8>::with_capacity(rand_image.shape().iter().product::<usize>() * 4);
        for float in rand_image.iter() {
            tensor_bytes.extend_from_slice(&float.to_le_bytes());
        }


        let mut input_content = Vec::<Vec<u8>>::with_capacity(1);
        input_content.push(tensor_bytes);

        let request = tonic::Request::new(
            ModelInferRequest {
                model_name: model_name.to_string(),
                model_version: model_version.to_string(),
                id: request_id.to_string(),
                parameters: HashMap::<String, InferParameter>::new(),
                inputs: inputs_vec,
                outputs: Vec::<InferRequestedOutputTensor>::new(),
                raw_input_contents: input_content

            }
        );

        let response = self.client.model_infer(request).await?;

        Ok(response.get_ref().clone())
    }

    pub fn cuda_shared_memory_region_create(&mut self, region_name: &str, region_size: usize, device_id: usize) -> Result<CudaSharedMemoryRegionHandle,  Box<dyn Error>> {
        unsafe {
            let mut handle: *mut i32 = std::ptr::null_mut();
            let handle_ptr: *mut *mut i32 = &mut handle;


            // convert parameters to cstyle
}
