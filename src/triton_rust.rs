#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference::{ServerLiveRequest, ServerReadyRequest, ModelReadyRequest};
use inference::{InferParameter, ModelInferRequest, ModelInferResponse};
use inference::{ModelMetadataRequest, ModelMetadataResponse};
use inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use inference::{CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse};

use std;
use std::error::Error;
use std::vec::Vec;
use std::collections::HashMap;
use ndarray::{Array, Array3};

pub mod shared_memory;

pub mod inference {
    tonic::include_proto!("inference");
}

pub struct TritonInference {
    client: GrpcInferenceServiceClient<tonic::transport::Channel>
}

impl TritonInference {
    pub async fn connect(address: &'static str) -> Result<Self, Box<dyn Error>> {

        let client = GrpcInferenceServiceClient::connect(address).await?;

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

    pub async fn infer(&mut self, model_name: &str, model_version: &str, request_id: &str, inputs_vec: Vec<InferInputTensor>, input_content: Vec<Vec<u8>>) -> Result<ModelInferResponse,  Box<dyn Error>> {

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

    pub fn get_input_content_vector(&mut self, input_array: &Array3<f64>) -> Vec<u8> {

        let mut tensor_bytes = Vec::<u8>::with_capacity(input_array.shape().iter().product::<usize>() * 4);
        for float in input_array.iter() {
            tensor_bytes.extend_from_slice(&float.to_le_bytes());
        }

        tensor_bytes
    }

    pub async fn new_cuda_shared_memory(&mut self, name: &'static str, device_id: i64, size: u64) -> Result<shared_memory::CudaSharedMemoryRegionHandle,  Box<dyn Error>> {

        let mut cuda_handle = shared_memory::CudaSharedMemoryRegionHandle::create(name, size, device_id);
        let cuda_raw_handle = cuda_handle.get_raw_handle();

        let request = tonic::Request::new(
            CudaSharedMemoryRegisterRequest {
                name: name.to_string(),
                raw_handle: cuda_raw_handle,
                device_id: device_id.try_into().unwrap(),
                byte_size: size
            }
        );

        let response = self.client.cuda_shared_memory_register(request).await?;
        Ok(cuda_handle)
    }
}
