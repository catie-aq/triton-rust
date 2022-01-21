#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference::{ServerLiveRequest, ServerReadyRequest, ModelReadyRequest};
use inference::{InferParameter, ModelInferRequest, ModelInferResponse, infer_parameter};
use inference::{ModelMetadataRequest, ModelMetadataResponse};
use inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use inference::{CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse};
use inference::{CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse};
use inference::{CudaSharedMemoryUnregisterResponse, CudaSharedMemoryUnregisterRequest};
use inference::{SystemSharedMemoryRegisterRequest, SystemSharedMemoryRegisterResponse};
use inference::{SystemSharedMemoryStatusRequest, SystemSharedMemoryStatusResponse};
use inference::{SystemSharedMemoryUnregisterRequest, SystemSharedMemoryUnregisterResponse};

use std;
use std::error::Error;
use std::vec::Vec;
use std::collections::HashMap;
use std::mem;
use ndarray::{Array, Array3, Array4};

pub mod cuda_shared_memory;
pub mod system_shared_memory;

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

    pub fn get_infer_output(&mut self, input_name: &str, parameters_map: HashMap<String, InferParameter>) -> InferRequestedOutputTensor {

        InferRequestedOutputTensor {
            name: input_name.to_string(),
            parameters: parameters_map
        }
    }

    pub async fn get_model_metadata(&mut self, model_name: &str, model_version: &str) -> Result<ModelMetadataResponse,  Box<dyn Error>> {
        let request = tonic::Request::new(ModelMetadataRequest {name: model_name.to_string(), version: model_version.to_string()});
        let response = self.client.model_metadata(request).await?;

        Ok(response.get_ref().clone())
    }

    pub async fn infer(&mut self, model_name: &str, model_version: &str, request_id: &str, inputs_vec: Vec<InferInputTensor>, outputs_vec: Vec<InferRequestedOutputTensor>, input_content: Vec<Vec<u8>>) -> Result<ModelInferResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            ModelInferRequest {
                model_name: model_name.to_string(),
                model_version: model_version.to_string(),
                id: request_id.to_string(),
                parameters: HashMap::<String, InferParameter>::new(),
                inputs: inputs_vec,
                outputs: outputs_vec,
                raw_input_contents: input_content

            }
        );

        let response = self.client.model_infer(request).await?;

        Ok(response.get_ref().clone())
    }

    pub fn get_input_content_vector(&mut self, input_array: &Array4<f32>) -> Vec<u8> {

        let mut tensor_bytes = Vec::<u8>::with_capacity(input_array.shape().iter().product::<usize>() * mem::size_of::<f32>());
        for float in input_array.iter() {
            tensor_bytes.extend_from_slice(&float.to_ne_bytes());
        }

        tensor_bytes
    }

    pub async fn create_cuda_shared_memory(&mut self, name: &'static str, size: u64, device_id: i64) -> Result<cuda_shared_memory::CudaSharedMemoryRegionHandle,  Box<dyn Error>> {

        let mut cuda_handle = cuda_shared_memory::CudaSharedMemoryRegionHandle::create(name, size, device_id);
        let cuda_raw_handle = cuda_handle.get_raw_handle();

        let request = tonic::Request::new(
            CudaSharedMemoryRegisterRequest {
                name: name.to_string(),
                raw_handle: cuda_raw_handle,
                device_id: device_id,
                byte_size: size
            }
        );

        let response = self.client.cuda_shared_memory_register(request).await?;
        Ok(cuda_handle)
    }

    pub async fn cuda_shared_memory_status(&mut self, name: &'static str) -> Result<CudaSharedMemoryStatusResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            CudaSharedMemoryStatusRequest {
                name: name.to_string()
            }
        );

        let response = self.client.cuda_shared_memory_status(request).await?;

        Ok(response.get_ref().clone())
    }

    pub async fn unregister_cuda_shared_memory(&mut self, name: &'static str) -> Result<CudaSharedMemoryUnregisterResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            CudaSharedMemoryUnregisterRequest {
                name: name.to_string()
            }
        );

        let response = self.client.cuda_shared_memory_unregister(request).await?;

        Ok(response.get_ref().clone())
    }

    pub async fn create_system_shared_memory(&mut self, name: &'static str, key: &'static str, size: u64) -> Result<system_shared_memory::SystemSharedMemoryRegionHandle,  Box<dyn Error>> {

        let mut shm_handle = system_shared_memory::SystemSharedMemoryRegionHandle::create(name, key, size);

        let request = tonic::Request::new(
            SystemSharedMemoryRegisterRequest {
                name: name.to_string(),
                key: key.to_string(),
                offset: 0,
                byte_size: size
            }
        );

        let response = self.client.system_shared_memory_register(request).await?;
        Ok(shm_handle)
    }

    pub async fn system_shared_memory_status(&mut self, name: &'static str) -> Result<SystemSharedMemoryStatusResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            SystemSharedMemoryStatusRequest {
                name: name.to_string()
            }
        );

        let response = self.client.system_shared_memory_status(request).await?;

        Ok(response.get_ref().clone())
    }

    pub async fn unregister_system_shared_memory(&mut self, name: &'static str) -> Result<SystemSharedMemoryUnregisterResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            SystemSharedMemoryUnregisterRequest {
                name: name.to_string()
            }
        );

        let response = self.client.system_shared_memory_unregister(request).await?;

        Ok(response.get_ref().clone())
    }

    pub fn get_system_shared_memory_params(&mut self, name: &'static str, size: u64, offset: u64) -> HashMap<String, InferParameter> {
        let params = HashMap::from([
            ("shared_memory_region".to_string(), InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::StringParam(name.to_string())) }),
            ("shared_memory_byte_size".to_string(), InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::Int64Param(size.try_into().unwrap())) }),
            ("shared_memory_offset".to_string(), InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::Int64Param(offset.try_into().unwrap())) })
        ]);

        params
    }
}
